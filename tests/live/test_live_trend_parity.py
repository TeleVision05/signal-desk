"""Vol-targeted trend: the live strategy, replayed bar by bar, equals the research target EXACTLY.

Research target: ``experiments/intraday/voltarget_trend.py`` ``vt_target`` for config
``vt_1h_slow4_z_s30`` on the full cached 1h history. Live: ``VolTargetTrend.decide`` on the
last ``warmup_bars`` closed bars, fed its OWN previous target (as the runner feeds it from
the store), never the research one, except once as the starting state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from parity_helpers import bars, research

from profit_engine.live.strategies import voltarget as live
from profit_engine.live.strategies.voltarget import VolTargetTrend
from profit_engine.live.strategy import decision_window, target_series

CONFIG = "vt_1h_slow4_z_s30"
SYMBOLS = ("BTCUSDT", "ETHUSDT")


def _research_target(symbol: str) -> pd.Series:
    vt = research("voltarget_trend")
    return vt.vt_target(bars(symbol, "1h"), "1h", vt.GRID_BY_NAME[CONFIG]["params"])


def test_live_parameters_are_the_research_grid_entry():
    vt = research("voltarget_trend")
    assert live.CONFIGS[CONFIG] == {"tf": vt.GRID_BY_NAME[CONFIG]["tf"], "params": vt.GRID_BY_NAME[CONFIG]["params"]}
    assert live.STEP == vt.STEP and live.BARS_PER_DAY == vt.BARS_PER_DAY


def test_one_hysteresis_step_at_a_time_is_the_research_loop():
    vt = research("voltarget_trend")
    rng = np.random.default_rng(3)
    raw = rng.uniform(-0.1, 1.2, 20_000)
    raw[rng.random(raw.size) < 0.05] = np.nan
    raw[:50] = np.nan  # flat before the first value
    cur, out = 0.0, []
    for x in raw:
        cur = live.hysteresis_step(x, cur, 0.2)
        out.append(cur)
    assert np.array_equal(np.array(out), vt.hysteresis(raw, 0.2))


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_replay_2025_to_2026_06_equals_research_exactly(symbol):
    """Every 1h bar of 2025-01..2026-06: same target, and the raw exposure within 1e-9."""
    vt = research("voltarget_trend")
    hourly, want = bars(symbol, "1h"), _research_target(symbol)
    want_raw = vt.raw_exposure(hourly, "1h", vt.GRID_BY_NAME[CONFIG]["params"])
    strategy = VolTargetTrend(symbol)
    first = int(hourly.index.searchsorted(pd.Timestamp("2025-01-01", tz="UTC")))
    prev = float(want.iloc[first - 1])  # the starting state only; from here the live output is fed back
    got, worst = [], 0.0
    for i in range(first, len(hourly)):
        window = decision_window(hourly.iloc[: i + 1], strategy)
        d = strategy.decide(window, None, prev_target=prev)
        prev = d.target
        got.append(prev)
        worst = max(worst, abs(d.details["raw"] / want_raw.iloc[i] - 1) if want_raw.iloc[i] else abs(d.details["raw"]))
    got = pd.Series(got, index=hourly.index[first:])
    assert len(got) > 13_000 and got.index[-1] >= pd.Timestamp("2026-06-30 23:00", tz="UTC")
    assert (got.diff().fillna(0) != 0).sum() >= 40  # the window actually trades
    assert got.equals(want.iloc[first:])
    assert worst < 1e-9, f"warmup too short: worst relative raw-exposure difference {worst:.2e}"


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_replay_from_the_first_bar_equals_research(symbol):
    """From the first cached bar, flat, with no research state at all (2017-09..2019-12, every
    bar). Until ``warmup_bars`` bars exist the window is the whole history so far, which is what
    the research had; after that it is the last ``warmup_bars`` bars, as the runner passes."""
    hourly = bars(symbol, "1h")
    span = hourly[hourly.index < pd.Timestamp("2020-01-01", tz="UTC")]
    strategy = VolTargetTrend(symbol)
    prev, got = 0.0, []
    for i in range(len(span)):
        prev = strategy.decide(span.iloc[max(0, i + 1 - strategy.warmup_bars) : i + 1], None, prev_target=prev).target
        got.append(prev)
    got = pd.Series(got, index=span.index)
    assert got.equals(_research_target(symbol).loc[span.index])
    assert (got.diff().fillna(0) != 0).sum() >= 40


def test_target_series_feeds_back_the_strategys_own_target():
    """``strategy.target_series`` (the backtest of the live code path) carries the state."""
    hourly = bars("BTCUSDT", "1h")
    strategy = VolTargetTrend("BTCUSDT")
    end = int(hourly.index.searchsorted(pd.Timestamp("2026-01-21", tz="UTC")))
    span = hourly.iloc[end - strategy.warmup_bars - 24 * 30 : end]  # a busy month for the band
    got = target_series(strategy, span).dropna()
    prev, want = 0.0, []
    for i in range(strategy.warmup_bars - 1, len(span)):
        prev = strategy.decide(span.iloc[i + 1 - strategy.warmup_bars : i + 1], None, prev_target=prev).target
        want.append(prev)
    assert got.tolist() == want and len(set(want)) > 2
