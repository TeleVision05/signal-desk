"""Selection-bias statistics: Deflated Sharpe Ratio, PBO via CSCV, random-timing permutation test.

All Sharpe ratios here are per-period (daily, NOT annualised), as the formulas require.

References: Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"; Bailey, Borwein,
López de Prado & Zhu (2016), "The Probability of Backtest Overfitting".
"""

from __future__ import annotations

from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd

from profit_engine.intraday.backtest import BacktestResult, bar_pnl
from profit_engine.intraday.holdout import assert_pre_holdout

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def expected_max_sharpe(n_trials: int, var_trial_sharpe: float) -> float:
    """E[max SR] of ``n_trials`` zero-skill trials whose SRs have variance ``var_trial_sharpe``."""
    if n_trials <= 1 or var_trial_sharpe <= 0:
        return 0.0
    a = _N.inv_cdf(1 - 1 / n_trials)
    b = _N.inv_cdf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(var_trial_sharpe) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b))


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int, skew: float, kurt: float) -> float:
    """P(true SR > sr_benchmark) given an observed per-period ``sr`` (kurt is NON-excess)."""
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if n_obs < 2 or denom <= 0:
        return 0.0
    return _N.cdf((sr - sr_benchmark) * np.sqrt(n_obs - 1) / np.sqrt(denom))


def deflated_sharpe_ratio(
    sr: float, n_trials: int, var_trial_sharpe: float, n_obs: int, skew: float, kurt: float
) -> float:
    """DSR = PSR against the expected maximum Sharpe of ``n_trials`` unskilled trials."""
    return probabilistic_sharpe(sr, expected_max_sharpe(n_trials, var_trial_sharpe), n_obs, skew, kurt)


def dsr_from_daily(daily: pd.Series, n_trials: int, var_trial_sharpe: float) -> float:
    assert_pre_holdout(daily.index, "overfitting.dsr_from_daily")
    x = daily.to_numpy(dtype="float64")
    sd = x.std(ddof=1)
    sr = x.mean() / sd if sd > 0 else 0.0
    return deflated_sharpe_ratio(
        sr, n_trials, var_trial_sharpe, len(x), float(pd.Series(x).skew()), float(pd.Series(x).kurt() + 3)
    )


def _sharpe_from_sums(s1: np.ndarray, s2: np.ndarray, k: int) -> np.ndarray:
    var = (s2 - s1**2 / k) / (k - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (s1 / k) / np.sqrt(var)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def pbo_cscv(returns: pd.DataFrame | np.ndarray, n_blocks: int = 16, chunk: int = 2048) -> dict:
    """Probability of Backtest Overfitting by Combinatorially Symmetric Cross-Validation.

    ``returns``: T x N matrix of per-period returns, one column per trial, same dates.
    Returns ``{"pbo", "logits", "n_combinations", "n_trials"}``. PBO is the share of IS/OOS
    splits where the in-sample best trial ranks at or below the OOS median.
    """
    if isinstance(returns, pd.DataFrame):
        assert_pre_holdout(returns.index, "overfitting.pbo_cscv")
    m = np.asarray(returns, dtype="float64")
    t, n = m.shape
    if n < 2 or n_blocks % 2 or t < 2 * n_blocks:
        raise ValueError("need >= 2 trials, an even n_blocks, and >= 2 rows per block")
    m = m[t % n_blocks :]
    k = len(m) // n_blocks
    blocks = m.reshape(n_blocks, k, n)
    b1, b2 = blocks.sum(axis=1), (blocks**2).sum(axis=1)  # n_blocks x N
    combos = np.array(list(combinations(range(n_blocks), n_blocks // 2)))
    half = k * n_blocks // 2
    logits = []
    for start in range(0, len(combos), chunk):
        sel = np.zeros((len(combos[start : start + chunk]), n_blocks))
        np.put_along_axis(sel, combos[start : start + chunk], 1.0, axis=1)
        is_sr = _sharpe_from_sums(sel @ b1, sel @ b2, half)
        oos_sr = _sharpe_from_sums((1 - sel) @ b1, (1 - sel) @ b2, half)
        best = is_sr.argmax(axis=1)
        chosen = oos_sr[np.arange(len(best)), best][:, None]
        rank = (oos_sr < chosen).sum(axis=1) + 0.5 * ((oos_sr == chosen).sum(axis=1) - 1) + 1
        omega = rank / (n + 1)
        logits.append(np.log(omega / (1 - omega)))
    lam = np.concatenate(logits)
    return {"pbo": float((lam <= 0).mean()), "logits": lam, "n_combinations": len(lam), "n_trials": n}


def _segments(held: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    """Split a position path into in-market runs and the flat gaps around them."""
    in_mkt = np.concatenate([[False], held > 0, [False]])
    edges = np.flatnonzero(np.diff(in_mkt.astype(int)))
    starts, ends = edges[::2], edges[1::2]
    trades = [held[s:e] for s, e in zip(starts, ends, strict=True)]
    bounds = np.concatenate([[0], np.ravel(np.column_stack([starts, ends])), [len(held)]])
    gaps = list(np.diff(bounds)[::2])
    return trades, gaps


def _daily_sharpe(held: np.ndarray, price: np.ndarray, c: float, day: np.ndarray, n_days: int) -> float:
    """Daily Sharpe of ``held`` under exactly the simulator's P&L model (:func:`bar_pnl`)."""
    _, gross, cost = bar_pnl(held, price, c)
    ln = np.log1p(gross) + np.log1p(-cost)
    daily = np.expm1(np.bincount(day, weights=ln, minlength=n_days))
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd) if sd > 0 else 0.0


def permutation_test(result: BacktestResult, n_perm: int = 500, seed: int = 0) -> dict:
    """Random-timing null: same trades (lengths, sizes), same flat gaps, shuffled order.

    Exposure, trade count, holding periods and turnover (hence costs) are preserved exactly;
    only WHEN the trades happen changes. Statistic: daily net Sharpe. Returns
    ``{"p_value", "sharpe_daily", "null_mean", "null_p95", "n_perm"}``.
    """
    bars = result.bars
    assert_pre_holdout(bars.index, "overfitting.permutation_test")
    held, ret = bars["position"].to_numpy(), bars["ret"].to_numpy()
    price = np.concatenate([[1.0], np.cumprod(1 + ret[:-1])])  # proportional to the opens
    days = bars.index.floor("D")
    day = ((days - days[0]) // pd.Timedelta("1D")).to_numpy().astype("int64")
    n_days, c = int(day[-1]) + 1, result.cost_bps_per_side / 1e4
    actual = _daily_sharpe(held, price, c, day, n_days)
    trades, gaps = _segments(held)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        order_t, order_g = rng.permutation(len(trades)), rng.permutation(len(gaps))
        pieces = [np.zeros(gaps[order_g[0]])]
        for j, ti in enumerate(order_t):
            pieces += [trades[ti], np.zeros(gaps[order_g[j + 1]])]
        null[i] = _daily_sharpe(np.concatenate(pieces), price, c, day, n_days)
    return {
        "p_value": float((1 + (null >= actual).sum()) / (1 + n_perm)),
        "sharpe_daily": actual,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "n_perm": n_perm,
    }
