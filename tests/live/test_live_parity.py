"""Parity: the runner on a replayed bar stream == the harness backtest on the same bars.

The reference strategy goes through ``intraday.backtest.run_backtest`` once, and through the
live runner tick by tick (every 15 minutes, as a Mac that never sleeps would). Fill times,
prices, targets and costs must match exactly; equity to float precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from live_helpers import FakeSource, book, config, make_5m, rows

from profit_engine.intraday.backtest import run_backtest
from profit_engine.intraday.resample import resample
from profit_engine.live.runner import tick
from profit_engine.live.store import iso
from profit_engine.live.strategy import SmaTrendDemo, target_series


@pytest.mark.parametrize("cost_bps", [5.0, 10.0])
def test_runner_matches_harness_backtest(store, quiet, cost_bps):
    bars5 = make_5m(12 * 24 * 7, seed=11, vol=0.004)  # 7 days of 5m bars
    strategy = SmaTrendDemo("BTCUSDT", "1h", fast=3, slow=8)
    hourly = resample(bars5, "1h", incomplete="drop")

    harness = run_backtest(hourly, target_series(strategy, hourly), cost_bps)
    held = harness.bars["position"].to_numpy()
    changed = np.flatnonzero(np.diff(np.concatenate([[0.0], held])) != 0)
    changed = changed[changed < len(hourly) - 1]  # the harness force-exits on its last bar; live does not
    expected = [(iso(hourly.index[j]), held[j], hourly["open"].iloc[j]) for j in changed]
    assert len(expected) >= 6  # the replay must actually trade

    cfg = config(book(strategy, cost_bps_per_side=cost_bps, max_daily_loss=0.0))
    src = FakeSource({"BTCUSDT": bars5}, bars5.index[0])
    end = hourly.index[-1]  # the harness force-exits here, so the replay stops just before it
    for now in pd.date_range(bars5.index[0] + pd.Timedelta("15min"), end - pd.Timedelta("5min"), freq="15min"):
        src.now = now + pd.Timedelta("30s")
        result = tick(cfg, store, src, quiet)
        assert result["ok"], result

    fills = rows(store, "SELECT * FROM fills WHERE reason='signal' ORDER BY fill_bar_ts")
    got = [(f["fill_bar_ts"], f["target"], f["price"]) for f in fills]
    assert got == expected

    start = 10_000.0
    for f, j in zip(fills, changed, strict=True):
        assert f["cost"] / f["equity_before"] == pytest.approx(harness.bars["cost"].iloc[j], rel=1e-9, abs=1e-15)
        assert f["equity_before"] == pytest.approx(start * harness.bars["equity"].iloc[j - 1], rel=1e-9)
        assert f["equity_after"] == pytest.approx(
            start * harness.bars["equity"].iloc[j - 1] * (1 - harness.bars["cost"].iloc[j]), rel=1e-9
        )
    decided = rows(store, "SELECT bar_open_ts, target FROM decisions ORDER BY bar_open_ts")
    want = target_series(strategy, hourly).dropna()
    want = want[want.index < end - pd.Timedelta("1h")]  # every bar the replay could decide
    assert [(d["bar_open_ts"], d["target"]) for d in decided] == [(iso(t), x) for t, x in want.items()]
