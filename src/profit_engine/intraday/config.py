"""Constants and paths shared by the intraday research harness.

Every date here is UTC. Ranges are half-open: ``[start, end)``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
BASE_INTERVAL = "5m"
BAR = pd.Timedelta(minutes=5)
TIMEFRAMES: tuple[str, ...] = ("5m", "15m", "1h", "4h")

# Earliest bar a model may train on (Binance spot 5m history starts 2017-08-17).
TRAIN_START = pd.Timestamp("2017-09-01", tz="UTC")
# Walk-forward out-of-sample evaluation range: 2020-01-01 .. 2026-06-30 inclusive.
EVAL_START = pd.Timestamp("2020-01-01", tz="UTC")
# Sealed holdout. Nothing at or after this instant is loaded, scored or evaluated
# unless the caller is inside ``holdout.open_holdout(reason=...)``.
HOLDOUT_START = pd.Timestamp("2026-07-01", tz="UTC")
EVAL_END = HOLDOUT_START  # exclusive

BASE_COST_BPS = 5.0
STRESS_COST_BPS = 10.0
PERIODS_PER_YEAR = 365  # crypto trades every calendar day

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "cache" / "intraday"
RESEARCH_DIR = REPO_ROOT / "research" / "intraday"
HOLDOUT_AUDIT_LOG = RESEARCH_DIR / "holdout_access.jsonl"
TRIALS_DIR = RESEARCH_DIR / "trials"

# Already SHA256-verified 5m dumps from an earlier research session; used to seed the
# cache so a first run does not re-download ~110 monthly zips per symbol.
SEED_DIR = Path(
    "/private/tmp/claude-501/-Users-tarunvadapalli-Documents-Git-profit-engine/"
    "fc846fab-d5a7-4eb4-a54d-70da1b8ae7f4/scratchpad/research/data"
)

KLINE_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
)
