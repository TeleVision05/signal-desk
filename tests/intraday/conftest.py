"""Shared fixtures: synthetic 5m bars and isolated research/cache paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from profit_engine.intraday import config


def make_bars(
    n: int = 2000,
    seed: int = 0,
    start: str = "2021-03-01",
    vol: float = 0.002,
    gapless: bool = True,
) -> pd.DataFrame:
    """Random-walk 5m bars on the grid. ``gapless`` => open[t] == close[t-1]."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC", name="open_time")
    close = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    open_ = np.concatenate([[100.0], close[:-1]]) if gapless else close * np.exp(rng.normal(0, vol, n))
    wick = np.abs(rng.normal(0, vol, n))
    high = np.maximum(open_, close) * np.exp(wick)
    low = np.minimum(open_, close) * np.exp(-wick)
    volume = rng.gamma(2.0, 5.0, n)
    taker = volume * rng.uniform(0.2, 0.8, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": rng.integers(1, 500, n),
            "taker_buy_base": taker,
            "taker_buy_quote": taker * close,
        },
        index=idx,
    )


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the audit log, trial registry and cache at a temp dir."""
    monkeypatch.setattr(config, "HOLDOUT_AUDIT_LOG", tmp_path / "holdout_access.jsonl")
    monkeypatch.setattr(config, "TRIALS_DIR", tmp_path / "trials")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    return tmp_path


@pytest.fixture
def bar_factory():
    return make_bars
