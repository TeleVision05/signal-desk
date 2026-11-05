"""(c) resample completeness assertion; (d) in-progress bars dropped; epoch normalisation;
checksum verification; cache round trip."""

from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from profit_engine.intraday import binance, data
from profit_engine.intraday.resample import IncompleteBarError, resample

MS = 300_000


def _raw(open_ms: list[int], close_offset: int = MS - 1, scale: int = 1) -> pd.DataFrame:
    rows = []
    for o in open_ms:
        rows.append([o * scale, "1", "2", "0.5", "1.5", "10", (o + close_offset) * scale, "15", "7", "4", "6", "0"])
    return pd.DataFrame(rows, columns=list(binance.RAW_COLUMNS)).astype(str)


# ---------------------------------------------------------------- (c) resample


def test_c_resample_trips_on_missing_sub_bar(bars):
    full = bars.iloc[:48]  # 4 whole hours starting at midnight
    out = resample(full, "1h")
    assert len(out) == 4 and (out["n_sub"] == 12).all()
    holed = full.drop(full.index[17])  # one 5m bar missing inside hour #2
    with pytest.raises(IncompleteBarError):
        resample(holed, "1h")
    with pytest.raises(AssertionError):  # IncompleteBarError is an AssertionError
        resample(holed, "1h", incomplete="raise")
    dropped = resample(holed, "1h", incomplete="drop")
    assert len(dropped) == 3 and pd.Timestamp("2021-03-01 01:00", tz="UTC") not in dropped.index
    flagged = resample(holed, "1h", incomplete="flag")
    assert flagged["complete"].tolist() == [True, False, True, True]
    assert flagged.loc["2021-03-01 01:00", "n_sub"].item() == 11


def test_c_resample_ohlcv_and_left_labels(bars):
    out = resample(bars.iloc[:48], "15min")
    first = bars.iloc[:3]
    assert out.index[0] == bars.index[0]
    assert out["open"].iloc[0] == first["open"].iloc[0]
    assert out["close"].iloc[0] == first["close"].iloc[-1]
    assert out["high"].iloc[0] == first["high"].max()
    assert out["volume"].iloc[0] == pytest.approx(first["volume"].sum())


def test_c_4h_buckets_anchor_to_utc_midnight(bar_factory):
    df = bar_factory(n=200, start="2021-03-01 01:30")
    out = resample(df, "4h", incomplete="flag")
    assert all(ts.hour % 4 == 0 and ts.minute == 0 for ts in out.index)
    assert not out["complete"].iloc[0]  # 01:30 start => first 4h bucket partial
    assert not out["complete"].iloc[-1]  # truncated end => last bucket partial


# ---------------------------------------------------------------- (d) in-progress + normalisation


def test_d_in_progress_bar_dropped():
    base = 1_700_000_100_000 // MS * MS
    raw = _raw([base, base + MS, base + 2 * MS])
    now = pd.Timestamp(base + 2 * MS + 120_000, unit="ms", tz="UTC")  # 2 min into the last bar
    df, stats = data.normalize_raw(raw, now)
    assert len(df) == 2
    assert stats["in_progress_dropped"] == 1
    assert df.index[-1] == pd.Timestamp(base + MS, unit="ms", tz="UTC")
    # exactly at close_time is still in progress (close_time >= now)
    now_edge = pd.Timestamp(base + 3 * MS - 1, unit="ms", tz="UTC")
    assert len(data.normalize_raw(raw, now_edge)[0]) == 2
    assert len(data.normalize_raw(raw, now_edge + pd.Timedelta(1, "ms"))[0]) == 3


def test_d_load_klines_drops_unclosed_bars(isolated, bar_factory):
    df = bar_factory(n=10, start="2022-01-01")
    path = data.cache_path("BTCUSDT")
    path.parent.mkdir(parents=True)
    df.to_parquet(path)
    now = df.index[-1] + pd.Timedelta(minutes=3)
    out = data.load_klines("BTCUSDT", start="2021-01-01", end="2023-01-01", now=now)
    assert len(out) == 9


def test_microsecond_epochs_from_2025_normalised():
    base = 1_735_689_600_000  # 2025-01-01 00:00 UTC in ms
    raw = _raw([base, base + MS], scale=1000)  # archives switched to microseconds
    df, _ = data.normalize_raw(raw, pd.Timestamp("2030-01-01", tz="UTC"))
    assert df.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert df.index.freq is None and (np.diff(df.index.asi8) == MS * 10**6).all()


def test_off_grid_and_duplicate_bars_dropped():
    base = 1_700_000_100_000 // MS * MS
    raw = pd.concat([_raw([base, base + MS]), _raw([base + 7_000]), _raw([base + MS])])
    df, stats = data.normalize_raw(raw, pd.Timestamp("2030-01-01", tz="UTC"))
    assert len(df) == 2
    assert stats["off_grid_dropped"] == 1 and stats["duplicates_dropped"] == 1


def test_gap_report(bar_factory):
    df = bar_factory(n=50).drop(bar_factory(n=50).index[10:13])
    gaps = data.find_gaps(df)
    assert len(gaps) == 1 and gaps["missing_bars"].iloc[0] == 3


def test_checksum_verified():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.csv", "1500000000000,1,2,0.5,1.5,10,1500000299999,15,7,4,6,0\n")
    payload = buf.getvalue()
    good = hashlib.sha256(payload).hexdigest() + "  x.zip"
    binance.verify_checksum(payload, good, "x.zip")
    assert binance.parse_zip(payload)["open_time"].iloc[0] == "1500000000000"
    with pytest.raises(binance.ChecksumError):
        binance.verify_checksum(payload + b"x", good, "x.zip")


def test_c_harness_timeframe_names(bars):
    assert len(resample(bars.iloc[:48], "15m")) == 16  # "m" means minutes here, not months
