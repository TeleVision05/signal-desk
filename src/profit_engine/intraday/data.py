"""Binance GLOBAL spot 5m klines: normalise, cache to parquet, load (holdout-guarded).

Two entry points:

* :func:`sync_cache` builds/updates ``data/cache/intraday/<SYM>_5m.parquet`` with every
  CLOSED 5m bar up to now (seed file -> monthly SHA256-verified archives -> REST top-up)
  and returns a :class:`DataReport` (counts and gaps only, no prices).
* :func:`load_klines` reads the cache for research. Its default range is
  ``[TRAIN_START, HOLDOUT_START)``; asking for anything at or after ``HOLDOUT_START``
  raises unless inside ``open_holdout``.

Frame contract: index ``open_time`` (tz-aware UTC bar OPEN time, strictly increasing, on
the 5-minute grid, no fabricated bars) and columns ``config.KLINE_COLUMNS``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from profit_engine.intraday import binance, config
from profit_engine.intraday.holdout import assert_pre_holdout

GRID_MS = int(config.BAR.total_seconds() * 1000)
FLOAT_COLUMNS = [c for c in config.KLINE_COLUMNS if c != "trades"]


@dataclass
class DataReport:
    symbol: str
    rows: int
    first_bar: str
    last_bar: str
    expected_bars: int
    missing_bars: int
    n_gaps: int
    largest_gaps: list[dict] = field(default_factory=list)
    off_grid_dropped: int = 0
    duplicates_dropped: int = 0
    in_progress_dropped: int = 0
    zero_volume_bars: int = 0
    source: str = ""

    def line(self) -> str:
        return (
            f"{self.symbol}: {self.rows:,} closed 5m bars {self.first_bar} .. {self.last_bar} | "
            f"missing {self.missing_bars:,}/{self.expected_bars:,} "
            f"({self.missing_bars / max(self.expected_bars, 1):.3%}) in {self.n_gaps} gaps | "
            f"dropped off-grid={self.off_grid_dropped} dup={self.duplicates_dropped} "
            f"in-progress={self.in_progress_dropped} | zero-volume bars={self.zero_volume_bars}"
        )


def normalize_epoch(values: pd.Series) -> pd.Series:
    """Epoch integers -> milliseconds. Archives switched ms -> µs on 2025-01-01."""
    x = values.astype("int64")
    return x.where(x < 10**14, x // 1000)


def _utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def normalize_raw(raw: pd.DataFrame, now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Raw kline rows -> frame-contract DataFrame, plus counts of what was dropped.

    Drops (never repairs): bars still in progress (``close_time >= now``), bars whose open
    time is off the 5-minute grid, and duplicate open times (last one wins).
    """
    now_ms = int((now or _utcnow()).value // 1_000_000)
    open_ms = normalize_epoch(raw["open_time"])
    close_ms = normalize_epoch(raw["close_time"])
    in_progress = (close_ms >= now_ms).to_numpy()
    off_grid = ((open_ms % GRID_MS) != 0).to_numpy() & ~in_progress
    keep = ~in_progress & ~off_grid
    df = raw.loc[keep, list(config.KLINE_COLUMNS)].copy()
    df[FLOAT_COLUMNS] = df[FLOAT_COLUMNS].astype("float64")
    df["trades"] = df["trades"].astype("float64").astype("int64")
    stamps = pd.to_datetime(open_ms[keep].to_numpy(), unit="ms", utc=True)
    df.index = pd.DatetimeIndex(stamps, name="open_time").as_unit("ns")
    dups = int(df.index.duplicated(keep="last").sum())
    df = df[~df.index.duplicated(keep="last")].sort_index()
    stats = {
        "in_progress_dropped": int(in_progress.sum()),
        "off_grid_dropped": int(off_grid.sum()),
        "duplicates_dropped": dups,
    }
    return df, stats


def find_gaps(df: pd.DataFrame, bar: pd.Timedelta = config.BAR) -> pd.DataFrame:
    """One row per hole in the grid: last bar before it, first bar after, bars missing."""
    idx = df.index
    if len(idx) < 2:
        return pd.DataFrame(columns=["after", "resumes", "missing_bars"])
    step = idx[1:] - idx[:-1]
    holes = np.flatnonzero(step > bar)
    return pd.DataFrame(
        {
            "after": idx[holes],
            "resumes": idx[holes + 1],
            "missing_bars": (step[holes] // bar - 1).astype("int64"),
        }
    )


def build_report(symbol: str, df: pd.DataFrame, stats: dict, source: str) -> DataReport:
    gaps = find_gaps(df)
    expected = int((df.index[-1] - df.index[0]) // config.BAR) + 1 if len(df) else 0
    top = gaps.sort_values("missing_bars", ascending=False).head(5)
    return DataReport(
        symbol=symbol,
        rows=len(df),
        first_bar=str(df.index[0]) if len(df) else "",
        last_bar=str(df.index[-1]) if len(df) else "",
        expected_bars=expected,
        missing_bars=expected - len(df),
        n_gaps=len(gaps),
        largest_gaps=[{k: str(v) for k, v in row.items()} for row in top.to_dict("records")],
        zero_volume_bars=int((df["volume"] == 0).sum()) if len(df) else 0,
        source=source,
        **stats,
    )


def cache_path(symbol: str, cache_dir: Path | None = None) -> Path:
    return Path(cache_dir or config.CACHE_DIR) / f"{symbol}_5m.parquet"


def _seed_raw(symbol: str, seed_dir: Path) -> pd.DataFrame | None:
    """Raw frame from an earlier verified dump, or None if absent / schema does not fit."""
    path = Path(seed_dir) / f"binance_global_{symbol}_5m.csv.gz"
    if not path.exists():
        return None
    raw = pd.read_csv(path, dtype=str)
    needed = [c for c in binance.RAW_COLUMNS if c != "ignore"]
    if not set(needed).issubset(raw.columns):
        return None
    return raw[needed]


def _download(symbol: str, start: pd.Timestamp, now: pd.Timestamp, session: requests.Session) -> pd.DataFrame:
    """Raw rows for bars opening in [start, now): monthly archives for whole months, REST otherwise."""
    first_full = (start + pd.offsets.MonthBegin(0)).normalize()
    this_month = now.normalize().replace(day=1)
    frames: list[pd.DataFrame] = []
    rest_from = start
    if first_full < this_month:
        if start < first_full:
            frames.append(binance.fetch_rest(symbol, "5m", start.value // 10**6, session, first_full.value // 10**6))
        last = (this_month - pd.offsets.MonthBegin(1)).to_pydatetime()
        for y, m in binance.month_range((first_full.year, first_full.month), (last.year, last.month)):
            url = binance.monthly_url(symbol, "5m", y, m)
            try:
                frames.append(binance.fetch_archive(url, session))
                rest_from = pd.Timestamp(year=y, month=m, day=1, tz="UTC") + pd.offsets.MonthBegin(1)
            except binance.ArchiveMissing:
                rest_from = pd.Timestamp(year=y, month=m, day=1, tz="UTC")
                break  # newest month not published yet; REST covers it
    frames.append(binance.fetch_rest(symbol, "5m", rest_from.value // 10**6, session))
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=binance.RAW_COLUMNS)


def _write_atomic(df: pd.DataFrame, path: Path, report: DataReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)
    path.with_suffix(".json").write_text(json.dumps(asdict(report), indent=2))


def sync_cache(
    symbol: str,
    now: pd.Timestamp | None = None,
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
    seed_dir: Path | None = None,
    rebuild: bool = False,
) -> DataReport:
    """Bring the parquet cache up to the latest CLOSED bar and return a data report."""
    now = now or _utcnow()
    session = session or requests.Session()
    path = cache_path(symbol, cache_dir)
    stats = {"in_progress_dropped": 0, "off_grid_dropped": 0, "duplicates_dropped": 0}
    if path.exists() and not rebuild:
        base, source = pd.read_parquet(path), "cache"
        manifest = path.with_suffix(".json")
        if manifest.exists():  # carry forward cumulative drop counts
            prev = json.loads(manifest.read_text())
            stats = {k: int(prev.get(k, 0)) for k in stats}
    else:
        raw = _seed_raw(symbol, seed_dir or config.SEED_DIR)
        source = "seed"
        if raw is None:
            raw, source = _download(symbol, config.TRAIN_START - pd.Timedelta(days=31), now, session), "archive"
        base, stats = normalize_raw(raw, now)
    top_raw = _download(symbol, base.index[-1] + config.BAR, now, session)
    if len(top_raw):
        top, top_stats = normalize_raw(top_raw, now)
        stats = {k: stats[k] + top_stats[k] for k in stats}
        base = pd.concat([base, top])
        base = base[~base.index.duplicated(keep="last")].sort_index()
        source += "+rest"
    report = build_report(symbol, base, stats, source)
    _write_atomic(base, path, report)
    return report


def load_klines(
    symbol: str,
    start: pd.Timestamp | str = config.TRAIN_START,
    end: pd.Timestamp | str = config.HOLDOUT_START,
    cache_dir: Path | None = None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Closed 5m bars with open time in ``[start, end)`` from the cache (run sync_cache first)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start
    end = end.tz_localize("UTC") if end.tzinfo is None else end
    assert_pre_holdout(end - pd.Timedelta(1, "ns"), f"load_klines({symbol}) end")
    path = cache_path(symbol, cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run profit_engine.intraday.data.sync_cache('{symbol}')")
    df = pd.read_parquet(path)
    df = df[(df.index >= start) & (df.index < end)]
    closed = df.index + config.BAR <= (now or _utcnow())
    return df[closed]
