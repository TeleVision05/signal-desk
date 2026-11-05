"""Helpers for the LightGBM book tests: synthetic 4h bars and small fake trained models."""

from __future__ import annotations

from functools import cache

import lightgbm as lgb
import numpy as np
import pandas as pd
from live_helpers import make_5m

from profit_engine.intraday.resample import resample
from profit_engine.live.strategies import lgbm_artifacts as art
from profit_engine.live.strategies import lgbm_core as core

CONFIG = "lgbm_4h_h30_pt2_sl1"
FOUR_H = pd.Timedelta("4h")


@cache
def synthetic_4h() -> dict[str, pd.DataFrame]:
    return {s: resample(make_5m(48 * 400, seed=i, vol=0.003), "4h") for i, s in enumerate(("BTCUSDT", "ETHUSDT"))}


@cache
def dated_4h(last_open: str, n: int = 400) -> dict[str, pd.DataFrame]:
    """``n`` 4h bars of both coins whose newest bar opens at ``last_open`` (UTC)."""
    start = pd.Timestamp(last_open, tz="UTC") - (n - 1) * FOUR_H
    return {
        s: resample(make_5m(48 * n, seed=i, vol=0.003, start=str(start)), "4h")
        for i, s in enumerate(("BTCUSDT", "ETHUSDT"))
    }


@cache
def columns() -> list[str]:
    b = synthetic_4h()
    return list(core.feature_frame(b["BTCUSDT"], b["ETHUSDT"], "4h")[1].columns)


def fake_trained(symbol: str, month: str, threshold: float = 0.5) -> art.Trained:
    rng = np.random.default_rng(len(month))
    x = pd.DataFrame(rng.normal(size=(800, len(columns()))), columns=columns())
    params = {"objective": "binary", "verbose": -1, "num_leaves": 4, "seed": 1, "deterministic": True}
    booster = lgb.train(params, lgb.Dataset(x, rng.integers(0, 2, 800)), num_boost_round=5)
    meta = {"config": CONFIG, "symbol": symbol, "other_symbol": art.OTHER[symbol], "timeframe": "4h",
            "trades_month": month, "threshold": threshold, "threshold_quantile": 0.5, "feature_columns": columns(),
            "train_window": {"n_rows": 800}}  # fmt: skip
    return art.Trained(booster, threshold, 0.5, columns(), meta)


def frames_5m(end: str) -> dict[str, pd.DataFrame]:
    """Trivial 5m history of both coins up to ``end`` (for retrain tests that fake the training)."""
    idx = pd.date_range("2026-01-01", end, freq="5min", tz="UTC", inclusive="left")
    return {s: pd.DataFrame({"open": 1.0}, index=idx) for s in art.OTHER}
