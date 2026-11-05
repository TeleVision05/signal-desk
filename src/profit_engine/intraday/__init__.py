"""Intraday long/flat spot crypto research harness (BTCUSDT, ETHUSDT).

Usage guide and rules: ``docs/intraday_harness.md``. Modules:

* ``data`` / ``binance`` / ``resample`` - verified 5m klines, parquet cache, safe resampling
* ``features`` / ``labels`` - causal features, next-open labels with end times
* ``splits`` - purged walk-forward and inner purged CV
* ``backtest`` / ``metrics`` / ``overfitting`` - next-open simulator, metrics, DSR/PBO/permutation
* ``trials`` / ``promotion`` / ``holdout`` - trial registry, promotion gate, holdout guard
"""

from profit_engine.intraday.config import (
    BASE_COST_BPS,
    EVAL_END,
    EVAL_START,
    HOLDOUT_START,
    STRESS_COST_BPS,
    SYMBOLS,
    TIMEFRAMES,
    TRAIN_START,
)
from profit_engine.intraday.holdout import HoldoutViolation, open_holdout

__all__ = [
    "BASE_COST_BPS",
    "EVAL_END",
    "EVAL_START",
    "HOLDOUT_START",
    "STRESS_COST_BPS",
    "SYMBOLS",
    "TIMEFRAMES",
    "TRAIN_START",
    "HoldoutViolation",
    "open_holdout",
]
