"""(a) peeking is not profitable through the simulator; (b) a decision never earns its own bar;
(g) cost accounting; plus constraint and benchmark behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from profit_engine.intraday.backtest import buy_and_hold, run_backtest, vol_matched_buy_and_hold
from profit_engine.intraday.metrics import t_stat


def _jump_bars(n: int = 300, seed: int = 1) -> pd.DataFrame:
    """Bars whose price moves only BETWEEN bars (open == high == low == close inside a bar).

    So close[t] -> close[t+1] is exactly open[t] -> open[t+1]: the move that a decision made at
    t's close can only earn if the simulator wrongly fills at t's own close/open.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="5min", tz="UTC")
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0}, index=idx)


def test_a_peeking_at_next_close_is_not_profitable():
    df = _jump_bars(n=5000)
    peek = (df["close"].shift(-1) > df["close"]).astype(float)  # uses the FUTURE close
    res = run_backtest(df, peek, cost_bps_per_side=5)
    gross = res.bars["gross_ret"][res.bars["position"] > 0].to_numpy()
    # Correct fills (next open) leave the peek with no edge: gross per-bar mean is noise ...
    assert abs(t_stat(gross)) < 3
    # ... and after costs the strategy loses money.
    assert res.bars["equity"].iloc[-1] < 1.0
    # Control: the same signal filled at the decision bar's own price would make a fortune,
    # so the assertions above only hold if fills really happen at t+1's open.
    wrong = np.prod(1 + peek.to_numpy()[:-1] * (df["close"].shift(-1) / df["close"] - 1).to_numpy()[:-1])
    assert wrong > 1e6


def test_b_decision_at_t_never_earns_the_gap_into_t_plus_1():
    idx = pd.date_range("2022-01-01", periods=10, freq="5min", tz="UTC")
    px = np.array([100.0] * 5 + [110.0] * 5)  # gap up at bar 5's open
    df = pd.DataFrame({"open": px, "high": px, "low": px, "close": px}, index=idx)
    late = pd.Series(0.0, index=idx)
    late.iloc[4] = 1.0  # decided at bar 4's close, after which the price gaps
    res = run_backtest(df, late, cost_bps_per_side=0)
    assert res.bars["position"].iloc[4] == 0.0
    assert res.bars["position"].iloc[5] == 1.0
    assert res.bars["equity"].iloc[-1] == pytest.approx(1.0, abs=1e-15)

    early = pd.Series(0.0, index=idx)
    early.iloc[3] = 1.0  # decided at bar 3's close -> held over [open4, open5), which has the gap
    res = run_backtest(df, early, cost_bps_per_side=0)
    assert res.bars["equity"].iloc[-1] == pytest.approx(1.10, abs=1e-12)


def test_b_random_series_position_uncorrelated_with_own_bar_return(bar_factory):
    df = bar_factory(n=4000, seed=3)
    own = df["close"] / df["open"] - 1
    target = (own > 0).astype(float)  # perfectly knows its own bar's move at the close
    res = run_backtest(df, target, cost_bps_per_side=0)
    gross = res.bars["gross_ret"][res.bars["position"] > 0].to_numpy()
    assert abs(t_stat(gross)) < 3


def test_g_one_round_trip_on_flat_price_costs_10bps():
    idx = pd.date_range("2022-01-01", periods=6, freq="5min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=idx)
    target = pd.Series([0, 1, 1, 0, 0, 0], index=idx, dtype=float)
    res = run_backtest(df, target, cost_bps_per_side=5)
    assert res.bars["equity"].iloc[-1] == pytest.approx((1 - 5e-4) ** 2, abs=1e-15)
    assert len(res.trades) == 1
    assert res.trades["net_bps"].iloc[0] == pytest.approx(-10.0, abs=0.01)
    assert res.trades["gross_bps"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert res.trades["holding_bars"].iloc[0] == 2
    assert res.daily.sum() == pytest.approx((1 - 5e-4) ** 2 - 1, abs=1e-12)


def test_g_stress_cost_doubles_the_bill():
    idx = pd.date_range("2022-01-01", periods=6, freq="5min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=idx)
    res = run_backtest(df, pd.Series([0, 1, 0, 0, 0, 0], index=idx, dtype=float), cost_bps_per_side=10)
    assert res.trades["net_bps"].iloc[0] == pytest.approx(-20.0, abs=0.02)


def test_long_flat_only():
    idx = pd.date_range("2022-01-01", periods=4, freq="5min", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "close": 1.0}, index=idx)
    with pytest.raises(ValueError):
        run_backtest(df, pd.Series([0, -1, 0, 0], index=idx, dtype=float))
    with pytest.raises(ValueError):
        run_backtest(df, pd.Series([0, 1.5, 0, 0], index=idx, dtype=float))


def test_open_position_is_liquidated_on_last_bar():
    idx = pd.date_range("2022-01-01", periods=4, freq="5min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "close": 100.0}, index=idx)
    res = run_backtest(df, 1.0, cost_bps_per_side=5)
    assert res.bars["position"].iloc[-1] == 0.0
    assert len(res.trades) == 1
    assert res.trades["exit_time"].iloc[0] == idx[-1]


def test_min_hold_and_round_trip_cap():
    idx = pd.date_range("2022-01-01", periods=12, freq="5min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "close": 100.0}, index=idx)
    flicker = pd.Series([1, 0] * 6, index=idx, dtype=float)
    held = run_backtest(df, flicker, min_hold_bars=3).bars["position"].to_numpy()
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], held > 0, [0]]).astype(int))))[::2]
    assert (runs[:-1] >= 3).all()
    capped = run_backtest(df, flicker, max_round_trips_per_day=2)
    assert len(capped.trades) == 2


def test_daily_series_is_24_7_and_includes_flat_days(bar_factory):
    df = bar_factory(n=288 * 5, seed=2)
    df = df.drop(df.index[288 * 2 : 288 * 3])  # a whole missing day
    res = run_backtest(df, 0.0)
    assert len(res.daily) == 5
    assert (res.daily == 0).all()


def test_benchmarks(bar_factory):
    df = bar_factory(n=288 * 30, seed=4)
    bh = buy_and_hold(df, cost_bps_per_side=5)
    assert len(bh.trades) == 1
    expected = df["open"].iloc[-1] / df["open"].iloc[1] * (1 - 5e-4) ** 2
    assert bh.bars["equity"].iloc[-1] == pytest.approx(expected, rel=1e-12)
    half = bh.daily * 0.5
    vm, w = vol_matched_buy_and_hold(df, half, cost_bps_per_side=5)
    assert w == pytest.approx(0.5, rel=0.02)
    assert vm.bars["position"].max() == pytest.approx(w)
