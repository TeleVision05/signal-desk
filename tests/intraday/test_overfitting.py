"""(h) DSR / PBO / permutation test behave on synthetic noise and on a planted edge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from profit_engine.intraday.backtest import run_backtest
from profit_engine.intraday.overfitting import (
    _segments,
    deflated_sharpe_ratio,
    dsr_from_daily,
    expected_max_sharpe,
    pbo_cscv,
    permutation_test,
    probabilistic_sharpe,
)

T, N = 2048, 40


def _noise(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=T, freq="D", tz="UTC")
    return pd.DataFrame(rng.normal(0, 0.01, (T, N)), index=idx)


def test_h_pure_noise_trials_low_dsr_and_pbo_near_half():
    m = _noise()
    srs = m.mean() / m.std()
    best = m[srs.idxmax()]
    dsr = dsr_from_daily(best, n_trials=N, var_trial_sharpe=float(srs.var()))
    # Undeflated, the best of 40 coin flips clears a 0.95 bar; deflated, it does not.
    assert probabilistic_sharpe(srs.max(), 0.0, T, 0.0, 3.0) > 0.95
    assert dsr < 0.95
    pbo = pbo_cscv(m)["pbo"]
    assert 0.3 <= pbo <= 0.7


def test_h_noise_dsr_and_pbo_are_about_half_on_average():
    pbos, dsrs = [], []
    for seed in range(6):
        m = _noise(seed)
        srs = m.mean() / m.std()
        pbos.append(pbo_cscv(m)["pbo"])
        dsrs.append(dsr_from_daily(m[srs.idxmax()], n_trials=N, var_trial_sharpe=float(srs.var())))
    assert 0.35 <= float(np.mean(pbos)) <= 0.65
    assert float(np.mean(dsrs)) < 0.8 and max(dsrs) < 0.95


def test_h_planted_edge_low_pbo_high_dsr():
    m = _noise(1)
    m[7] = m[7] + 0.002  # daily SR ~0.2 (annualised ~3.8)
    srs = m.mean() / m.std()
    assert srs.idxmax() == 7
    assert pbo_cscv(m)["pbo"] < 0.05
    assert dsr_from_daily(m[7], n_trials=N, var_trial_sharpe=float(srs.var())) > 0.95


def test_dsr_reference_values():
    # E[max] grows with the number of trials; one trial means no deflation.
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01) > 0
    assert deflated_sharpe_ratio(0.1, 1, 0.01, 1000, 0.0, 3.0) == pytest.approx(
        probabilistic_sharpe(0.1, 0.0, 1000, 0.0, 3.0)
    )
    # E[max of 100 standard normals] ~= 0.4228 * z(0.99) + 0.5772 * z(1 - 1/(100e)) = 2.5306
    assert expected_max_sharpe(100, 1.0) == pytest.approx(2.5306, abs=1e-3)


def test_h_permutation_test_planted_timing_edge(bar_factory):
    df = bar_factory(n=288 * 60, seed=5)
    fwd = df["open"].shift(-2) / df["open"].shift(-1) - 1  # the move the simulator will credit
    rng = np.random.default_rng(0)
    informed = ((fwd > 0) & (rng.random(len(df)) < 0.3)).astype(float)  # partial foresight
    res = run_backtest(df, informed, cost_bps_per_side=0)
    out = permutation_test(res, n_perm=200, seed=0)
    assert out["p_value"] < 0.05

    random_pos = pd.Series((rng.random(len(df)) < 0.3).astype(float), index=df.index)
    res = run_backtest(df, random_pos, cost_bps_per_side=0)
    out = permutation_test(res, n_perm=200, seed=0)
    assert out["p_value"] > 0.05


def test_permutation_preserves_exposure_and_trades(bar_factory):
    df = bar_factory(n=500, seed=6)
    target = pd.Series((np.random.default_rng(1).random(500) < 0.2).astype(float), index=df.index)
    held = run_backtest(df, target).bars["position"].to_numpy()
    trades, gaps = _segments(held)
    assert sum(map(len, trades)) + sum(gaps) == len(held)
    assert len(gaps) == len(trades) + 1 and min(gaps) >= 1
