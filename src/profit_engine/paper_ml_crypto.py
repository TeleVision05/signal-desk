"""Paper-only driver for BTCUSDT walk-forward ML (24/7 second book).

Uses Binance *public* klines for prices and simulates fills in PaperLedger.
Never places live exchange orders.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .daily_backtest import StopConfig
from .data import fetch_binance_klines
from .ledger import PaperLedger
from .ml_walkforward import WFConfig
from .paper_ml import missing_inputs, publish, score_latest
from .paper_step import BookSpec, step

OUT = Path(__file__).resolve().parents[2] / "output"

RUN_ID = "ml-btc-hgb-paper-v1"
STRATEGY = "ml_BTCUSDT_hgb_h5_t0.55"
ASSET = "BTCUSDT"

PROMOTED = WFConfig(
    ticker="BTCUSDT",
    model="hgb",
    horizon=5,
    proba_threshold=0.55,
    hard_pct=0.08,
    trailing_pct=0.12,
    cost_bps=20.0,
    start="2018-01-01",
    embargo=5,
)
SPEC = BookSpec(
    run_id=RUN_ID,
    book="btc",
    strategy=STRATEGY,
    asset=ASSET,
    cost_bps=PROMOTED.cost_bps,
    stop=StopConfig(hard_pct=PROMOTED.hard_pct, trailing_pct=PROMOTED.trailing_pct),
    mark_source="binance",
    threshold=float(PROMOTED.proba_threshold),
    config=PROMOTED.__dict__,
)


def closed_days(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Drop the UTC day still in progress: its close is not final until midnight UTC."""
    now = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("UTC").tz_localize(None)
    return df.loc[df.index + pd.Timedelta(days=1) <= now]


def sync_crypto_paper(
    run_id: str = RUN_ID,
    *,
    initial_cash: float = 1000.0,
    db_path=None,
) -> dict:
    cfg = PROMOTED
    spec = SPEC if run_id == SPEC.run_id else BookSpec(**{**SPEC.__dict__, "run_id": run_id})
    btc = closed_days(fetch_binance_klines("BTCUSDT", start=cfg.start))
    try:
        eth = closed_days(fetch_binance_klines("ETHUSDT", start=cfg.start))["Close"]
    except Exception as exc:
        print(f"[crypto] ETH download failed: {exc}", flush=True)
        eth = None
    extras = {"eth": eth}
    problem = missing_inputs(btc, extras)
    want, p = (None, None) if problem else score_latest(btc, extras, cfg)
    ledger = PaperLedger(db_path) if db_path else PaperLedger()
    info = step(spec, btc, want, p, ledger=ledger, initial_cash=initial_cash, problem=problem)
    publish(info, OUT / "ml_crypto_paper_last.json")
    return info
