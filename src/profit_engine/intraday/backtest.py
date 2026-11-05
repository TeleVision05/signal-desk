"""Position-based long/flat simulator for 24/7 spot markets.

Timing contract (the whole point of this module):

* ``target[t]`` in [0, 1] is decided at bar ``t``'s CLOSE.
* It is filled at bar ``t+1``'s OPEN and held until bar ``t+2``'s open, so the position
  that earns the open-to-open move ``open[j] -> open[j+1]`` is ``target[j-1]``.
* A target never earns the move of the bar on which it was decided.
* Cost: ``cost_bps_per_side / 1e4 * |Δposition| * equity`` at every fill.
* The last bar has no next open: the book is forced flat there (exit cost charged).

``position`` is a fraction of equity re-set at each fill; between fills a fractional
position is treated as constant-weight (drift is not charged). For 0/1 targets this is exact.
NaN targets mean flat. Days are UTC calendar days; every day in the span appears in the
daily series (0 when flat or when the exchange had no bars).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from profit_engine.intraday import config
from profit_engine.intraday.holdout import assert_pre_holdout


@dataclass
class BacktestResult:
    bars: pd.DataFrame  # per bar: position, ret, gross_ret, cost, net_ret, equity
    trades: pd.DataFrame  # per trade ledger
    daily: pd.Series  # daily NET simple returns
    daily_gross: pd.Series  # daily GROSS simple returns
    cost_bps_per_side: float
    params: dict = field(default_factory=dict)


def _validate_target(df: pd.DataFrame, target: pd.Series | float) -> np.ndarray:
    if np.isscalar(target):
        target = pd.Series(float(target), index=df.index)
    tgt = pd.Series(target, dtype="float64").reindex(df.index).to_numpy()
    finite = tgt[np.isfinite(tgt)]
    if finite.size and (finite.min() < 0 or finite.max() > 1):
        raise ValueError(f"long/flat only: target must be in [0, 1], got [{finite.min()}, {finite.max()}]")
    return np.nan_to_num(tgt, nan=0.0)


def _apply_constraints(held: np.ndarray, day: np.ndarray, max_rt: int | None, min_hold: int) -> np.ndarray:
    """Sequentially enforce min holding period and a per-UTC-day entry cap on executed positions."""
    out = held.copy()
    prev, bars_in, entries, cur_day = 0.0, 0, 0, None
    last = len(out) - 1
    for j in range(len(out)):
        if day[j] != cur_day:
            cur_day, entries = day[j], 0
        want = out[j]
        if prev == 0 and want > 0:
            if max_rt is not None and entries >= max_rt:
                want = 0.0
            else:
                entries += 1
        elif prev > 0 and want == 0 and bars_in < min_hold and j != last:
            want = prev
        bars_in = bars_in + 1 if want > 0 else 0
        out[j] = prev = want
    return out


def _trade_ledger(
    idx: pd.DatetimeIndex, opens: np.ndarray, held: np.ndarray, lg: np.ndarray, ln: np.ndarray, cost: np.ndarray
) -> pd.DataFrame:
    in_mkt = (held > 0).astype(int)
    edges = np.diff(np.concatenate([[0], in_mkt]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    cg, cn = np.concatenate([[0.0], np.cumsum(lg)]), np.concatenate([[0.0], np.cumsum(ln)])
    gross = np.expm1(cg[ends] - cg[starts])
    net = np.exp(cn[ends] - cn[starts]) * (1 - cost[ends]) - 1
    return pd.DataFrame(
        {
            "entry_time": idx[starts],
            "entry_price": opens[starts],
            "exit_time": idx[ends],
            "exit_price": opens[ends],
            "holding_bars": ends - starts,
            "avg_position": [held[s:e].mean() for s, e in zip(starts, ends, strict=True)],
            "gross_ret": gross,
            "net_ret": net,
            "gross_bps": gross * 1e4,
            "net_bps": net * 1e4,
        }
    )


def _daily(idx: pd.DatetimeIndex, log_growth: np.ndarray) -> pd.Series:
    days = idx.floor("D")
    s = pd.Series(log_growth, index=days).groupby(level=0).sum()
    full = pd.date_range(days[0], days[-1], freq="D", tz=days.tz, name="date")
    return np.expm1(s.reindex(full, fill_value=0.0))


def run_backtest(
    df: pd.DataFrame,
    target: pd.Series | float,
    cost_bps_per_side: float = config.BASE_COST_BPS,
    max_round_trips_per_day: int | None = None,
    min_hold_bars: int = 0,
) -> BacktestResult:
    """Simulate ``target`` (decided at each bar's close) on ``df``'s ``open`` prices."""
    if len(df) < 2:
        raise ValueError("need at least two bars")
    assert_pre_holdout(df.index, "run_backtest data")
    idx = df.index
    opens = df["open"].to_numpy(dtype="float64")
    held = np.concatenate([[0.0], _validate_target(df, target)[:-1]])
    held[-1] = 0.0  # no next open after the last bar: liquidate
    if max_round_trips_per_day is not None or min_hold_bars > 0:
        day = idx.floor("D").asi8
        held = _apply_constraints(held, day, max_round_trips_per_day, min_hold_bars)
    ret = np.zeros(len(df))
    ret[:-1] = opens[1:] / opens[:-1] - 1
    cost = cost_bps_per_side / 1e4 * np.abs(np.diff(np.concatenate([[0.0], held])))
    gross = held * ret
    lg, ln = np.log1p(gross), np.log1p(gross) + np.log1p(-cost)
    bars = pd.DataFrame(
        {
            "position": held,
            "ret": ret,
            "gross_ret": gross,
            "cost": cost,
            "net_ret": np.expm1(ln),
            "equity": np.exp(np.cumsum(ln)),
        },
        index=idx,
    )
    params = {"max_round_trips_per_day": max_round_trips_per_day, "min_hold_bars": min_hold_bars}
    return BacktestResult(
        bars=bars,
        trades=_trade_ledger(idx, opens, held, lg, ln, cost),
        daily=_daily(idx, ln),
        daily_gross=_daily(idx, lg),
        cost_bps_per_side=cost_bps_per_side,
        params=params,
    )


def buy_and_hold(df: pd.DataFrame, cost_bps_per_side: float = config.BASE_COST_BPS) -> BacktestResult:
    """Fully long from the second bar's open to the last bar's open (one entry, one exit)."""
    return run_backtest(df, 1.0, cost_bps_per_side)


def vol_matched_buy_and_hold(
    df: pd.DataFrame,
    strategy_daily: pd.Series,
    cost_bps_per_side: float = config.BASE_COST_BPS,
) -> tuple[BacktestResult, float]:
    """Constant-weight buy-and-hold scaled to the strategy's daily volatility (weight capped at 1).

    Returns the result and the weight used. The weight uses the full-period vol of both
    series, i.e. it is a benchmark, not a tradable strategy.
    """
    bh = buy_and_hold(df, cost_bps_per_side)
    common = bh.daily.index.intersection(strategy_daily.index)
    bh_sd = bh.daily.loc[common].std()
    w = float(np.clip(strategy_daily.loc[common].std() / bh_sd, 0.0, 1.0)) if bh_sd > 0 else 0.0
    return run_backtest(df, w, cost_bps_per_side), w
