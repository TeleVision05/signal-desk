"""``signal-desk-live train lgbm`` and ``signal-desk-live retrain``: keep the LightGBM models current.

Data. Training reads closed Binance 5m bars from ``paths.market_data_dir()``
(``<app folder>/market_data/<SYMBOL>_5m.parquet``), kept up to date with the harness's own
``intraday.data.sync_cache`` (monthly SHA256-verified archives, then REST). It is kept apart
from the research cache in ``data/cache/intraday`` so the live runner never changes research
inputs; the first sync copies that research cache, when present, instead of downloading 9
years of bars again.

Cutoff. The model for a month is trained on every closed bar from ``TRAIN_START`` up to the
first instant of that month (UTC), which is the research walk-forward fold for that month.
Training refuses to run unless the data reaches past the cutoff, so a failed download can never
quietly produce a model trained on less data.

``retrain`` trains each LightGBM book's model for the current month unless a USABLE one exists
(an artifact that fails to load, say after a library upgrade or a damaged file, is retrained,
which is what the book's "Retrain it" message promises). A failed training publishes nothing,
so the month's bars wait (the book keeps its position) and the next hourly run tries again.
It keeps the previous artifact in use when no threshold makes money (research would trade flat that month; the live
book keeps the last model instead, a marker records it so the hourly job does not train the
month again, and the event says so). It is safe to run any number of times: an up-to-date
model is not retrained, so the launchd job runs it every hour (``macos.RETRAIN_TIMES``) and a
new UTC month's model is ready before that month's first 4h bar closes.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from profit_engine.intraday import config, data
from profit_engine.live import paths
from profit_engine.live.registry import ConfigError, load_config
from profit_engine.live.store import EventSpec, Store, iso
from profit_engine.live.strategies import lgbm_artifacts as art
from profit_engine.live.strategies.lgbm import DEFAULT_CONFIG, LgbmStrategy

log = logging.getLogger("signal_desk.training")
SYMBOLS = tuple(art.OTHER)


class DataNotReady(RuntimeError):
    """The 5m history does not reach the cutoff yet (or could not be downloaded)."""


def _seed_from_research(symbol: str, data_dir: Path) -> None:
    target = data.cache_path(symbol, data_dir)
    source = data.cache_path(symbol, config.CACHE_DIR)
    if target.exists() or not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    log.info("seeded live market data from the research cache", extra={"symbol": symbol})


def load_5m(data_dir: Path, sync: bool = True, now: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    """Closed 5m bars of both symbols, synced to now first unless ``sync`` is False."""
    now = now or pd.Timestamp.now(tz="UTC")
    frames = {}
    for symbol in SYMBOLS:
        path = data.cache_path(symbol, data_dir)
        if sync:
            _seed_from_research(symbol, data_dir)
            data.sync_cache(symbol, now=now, cache_dir=data_dir, seed_dir=data_dir)
        if not path.exists():
            raise DataNotReady(f"no {symbol} history at {path}")
        frame = pd.read_parquet(path)
        frames[symbol] = frame[frame.index + config.BAR <= now]
    return frames


def train_and_publish(
    frames: dict[str, pd.DataFrame], symbol: str, cutoff: pd.Timestamp, models_dir: Path,
    cfg_name: str = DEFAULT_CONFIG, now: pd.Timestamp | None = None,
) -> Path:  # fmt: skip
    """Train the model for the month starting at ``cutoff`` and publish it; returns its folder."""
    for s in (symbol, art.OTHER[symbol]):
        if frames[s].empty or frames[s].index[-1] < cutoff:
            last = "nothing" if frames[s].empty else f"{frames[s].index[-1]:%Y-%m-%d %H:%M}"
            raise DataNotReady(f"{s} history ends at {last} UTC, before {cutoff:%Y-%m-%d}")
    try:
        trained = art.train(frames, symbol, cutoff, cfg_name)
    except art.NoThreshold:
        art.mark_no_threshold(models_dir, cfg_name, symbol, f"{cutoff:%Y-%m}", now or pd.Timestamp.now(tz="UTC"))
        raise
    return art.publish(trained, models_dir, now)


def month_ready(models_dir: Path, cfg_name: str, symbol: str, cutoff: pd.Timestamp) -> bool:
    """Whether the month starting at ``cutoff`` has a model that loads (or is known to have none)."""
    month = f"{cutoff:%Y-%m}"
    same = [p for p in art.versions(models_dir, cfg_name, symbol) if art.trades_month(p) == month]
    if not same:
        return art.has_no_threshold(models_dir, cfg_name, symbol, month)
    try:
        art.load(same[-1])
    except (art.ArtifactError, ValueError, OSError) as exc:
        log.warning("this month's model is unusable; retraining", extra={"symbol": symbol, "error": str(exc)})
        return False
    return True


def lgbm_targets() -> list[tuple[str, str, list[str]]]:
    """(config, symbol, book ids) for every LightGBM book in books.json; both coins if it won't load."""
    try:
        books = [b for b in load_config().books if isinstance(b.strategy, LgbmStrategy)]
    except (ConfigError, ValueError, OSError):
        return [(DEFAULT_CONFIG, s, []) for s in SYMBOLS]
    out: dict[tuple[str, str], list[str]] = {}
    for b in books:
        out.setdefault((b.strategy.config, b.symbol), []).append(b.book_id)
    return [(c, s, ids) for (c, s), ids in sorted(out.items())]


def _retrain_one(
    cfg_name: str, symbol: str, now: pd.Timestamp, models_dir: Path, frames
) -> tuple[str, str]:  # fmt: skip
    coin, cutoff = symbol.removesuffix("USDT"), art.month_start(now)
    if month_ready(models_dir, cfg_name, symbol, cutoff):
        if art.has_no_threshold(models_dir, cfg_name, symbol, f"{cutoff:%Y-%m}"):
            return "up_to_date", (
                f"The {coin} LightGBM model for {cutoff:%B %Y} found no setting that made money, so the book "
                "keeps its previous model."
            )
        return "up_to_date", f"The {coin} LightGBM model for {cutoff:%B %Y} is already trained."
    try:
        folder = train_and_publish(frames(), symbol, cutoff, models_dir, cfg_name, now)
    except art.NoThreshold:
        return "kept_old", (
            f"Retraining the {coin} LightGBM model for {cutoff:%B %Y} found no setting that made money in "
            "training, so the book keeps using its previous model (the backtest would have stayed in cash)."
        )
    except Exception:  # any failure publishes nothing; the detail is logged (retrain.jsonl)
        log.exception("retrain failed", extra={"symbol": symbol, "cutoff": iso(cutoff)})
        return "failed", (
            f"Retraining the {coin} LightGBM model for {cutoff:%B %Y} did not finish, so the book makes no new "
            f"decisions for {cutoff:%B} and keeps its position. It will try again within the hour."
        )
    return "trained", f"Trained a new {coin} LightGBM model for {cutoff:%B %Y} ({folder.name})."


def retrain(
    now: pd.Timestamp | None = None, models_dir: Path | None = None, data_dir: Path | None = None,
    sync: bool = True, db: Path | None = None,
) -> list[dict]:  # fmt: skip
    """Bring every LightGBM book's model up to the current month; record the outcome as events."""
    now = now or pd.Timestamp.now(tz="UTC")
    cutoff = art.month_start(now)
    models_dir, data_dir = Path(models_dir or paths.models_dir()), Path(data_dir or paths.market_data_dir())
    cache: dict = {}

    def frames() -> dict[str, pd.DataFrame]:
        if "f" not in cache:
            cache["f"] = load_5m(data_dir, sync, now)
        return cache["f"]

    results = []
    store = Store(db or paths.db_path())
    try:
        for cfg_name, symbol, book_ids in lgbm_targets():
            status, text = _retrain_one(cfg_name, symbol, now, models_dir, frames)
            results.append({"symbol": symbol, "config": cfg_name, "status": status, "message": text})
            if status != "up_to_date":
                event = EventSpec(f"model_{status}", text, {}, f"{cutoff:%Y-%m}:{now:%Y%m%d}")  # once a day
                for book_id in book_ids or [""]:
                    store.add_event(book_id, event, now)
    finally:
        store.close()
    return results
