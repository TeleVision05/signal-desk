"""Runner-level parity for the vol-targeted trend book on real cached bars.

The trend book goes through ``tick()`` once per hour with a fake Binance that replays the
cached 5m bars (2025-12-27..2026-01-21, a busy stretch for the band), and its target series
goes through ``intraday.backtest.run_backtest`` on the same 1h bars. Fractional targets make
the weight drift between fills, so this also checks the drifted-weight cost. Fill times,
prices and targets must match exactly; costs and equity to 1e-9. The runner feeds the
strategy its own last target from the store, ``target_series`` from its own output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from live_helpers import FakeSource, book, config, rows
from parity_helpers import raw_5m

from profit_engine.intraday.backtest import run_backtest
from profit_engine.intraday.resample import resample
from profit_engine.live.runner import tick
from profit_engine.live.store import iso
from profit_engine.live.strategies.voltarget import VolTargetTrend
from profit_engine.live.strategy import target_series

END = pd.Timestamp("2026-01-21", tz="UTC")


def test_trend_book_through_the_runner_matches_the_harness(store, quiet):
    strategy = VolTargetTrend("BTCUSDT")
    n = strategy.warmup_bars
    five = raw_5m("BTCUSDT")
    five = five[(five.index >= END - pd.Timedelta(hours=n + 24 * 26)) & (five.index < END)]
    hourly = resample(five, "1h", incomplete="drop")
    targets = target_series(strategy, hourly)  # starts flat at the first full window, like a new book
    harness = run_backtest(hourly, targets, 5.0)
    held = harness.bars["position"].to_numpy()
    changed = np.flatnonzero(np.diff(np.concatenate([[0.0], held])) != 0)
    changed = changed[changed < len(hourly) - 1]  # the harness force-exits on its last bar; live does not
    expected = [(iso(hourly.index[j]), held[j], hourly["open"].iloc[j]) for j in changed]
    assert len(expected) >= 8 and len(set(held[changed])) >= 5  # fractional moves both ways

    cfg = config(book(strategy, cost_bps_per_side=5.0, max_daily_loss=0.0))
    src = FakeSource({"BTCUSDT": five}, five.index[0])
    first = hourly.index[n - 1]  # the first bar with a full window
    for bar in hourly.index[hourly.index >= first][:-1]:
        src.now = bar + pd.Timedelta("1h30s")  # just after the bar closed; the next bar has opened
        result = tick(cfg, store, src, quiet)
        assert result["ok"], result

    fills = rows(store, "SELECT * FROM fills WHERE reason='signal' ORDER BY fill_bar_ts")
    assert [(f["fill_bar_ts"], f["target"], f["price"]) for f in fills] == expected
    start = 10_000.0
    for f, j in zip(fills, changed, strict=True):
        assert f["cost"] / f["equity_before"] == pytest.approx(harness.bars["cost"].iloc[j], rel=1e-9, abs=1e-15)
        assert f["equity_before"] == pytest.approx(start * harness.bars["equity"].iloc[j - 1], rel=1e-9)
    decided = rows(store, "SELECT bar_open_ts, target FROM decisions ORDER BY bar_open_ts")
    want = targets.dropna()
    want = want[want.index < hourly.index[-1]]
    assert [(d["bar_open_ts"], d["target"]) for d in decided] == [(iso(t), x) for t, x in want.items()]
