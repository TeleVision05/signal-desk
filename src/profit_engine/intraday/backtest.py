"""Position-based long/flat simulator for 24/7 spot markets.

Timing contract (the whole point of this module):

* ``target[t]`` in [0, 1] is decided at bar ``t``'s CLOSE.
* It is filled at bar ``t+1``'s OPEN and held until bar ``t+2``'s open, so the position
  that earns the open-to-open move ``open[j] -> open[j+1]`` is ``target[j-1]``.
* A target never earns the move of the bar on which it was decided.
* A fill happens only when the target CHANGES. Between fills the book holds its shares, so
  a fractional position drifts with the price: ``w' = w (1 + r) / (1 + w r)`` (column
  ``weight``). A fill re-sets the weight to the new target and is charged
  ``cost_bps_per_side / 1e4 * |target - drifted weight| * equity``. For 0/1 targets the
  drift is nil and this is exactly ``|Δposition|``. A constant 0.5 target is a true 50% hold.
* The last bar has no next open: the book is forced flat there (exit cost charged).

``target`` is a float or a Series on the bars' timestamps: every target timestamp inside the
bars' span must be a bar open time (a tz-naive, shifted or integer index raises instead of
silently becoming a flat book). Timestamps outside the span are ignored; bars with no target
or a NaN target are flat. An infinite target raises (it is not "flat" and not in [0, 1]).

Opens must be finite and positive (ValueError otherwise): drop missing bars, never reindex
onto a full grid, or the NaN bars would hide the move and the gap flag across each hole.

Gaps: ``span_bars`` is the open-to-next-open distance of each bar in units of the frame's
bar length (inferred as the median step unless ``bar`` is given). A position held over a
bar with ``span_bars > 1`` crossed a data gap: the trade ledger flags it (``crosses_gap``,
``holding_time``) and ``params`` counts it. P&L across a gap is the real pre-gap open to
post-gap open move (no stale fill).

Halted bars: a bar with ``volume == 0`` (Binance prints the stale previous close as its open
during an exchange halt) is not tradeable. A fill that would land on it is deferred to the
next bar with volume, and the position held before it is carried meanwhile, so it earns the
halt move it was really exposed to. ``params`` counts ``n_stale_bars`` and ``fills_deferred``;
the ledger flags trades held over a halted bar (``crosses_stale``). The forced exit on the last
bar happens even if it is halted (``forced_exit_on_stale``). A frame without a ``volume``
column has no halt detection (``params["stale_check"]`` is False). ``max_round_trips_per_day``
and ``min_hold_bars`` apply to the fills that actually happen: a deferred entry counts on the
day it fills and its holding period starts there.

The bar index must be a UTC DatetimeIndex, strictly increasing (ValueError otherwise): an
unsorted or duplicated frame would compute P&L across time-travelling "next opens", and a
non-UTC zone would bucket the daily series by local days.

Days are UTC calendar days; every day in the span appears in the daily series (0 when flat
or when the exchange had no bars).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from profit_engine.intraday import config
from profit_engine.intraday.holdout import assert_pre_holdout


@dataclass
class BacktestResult:
    bars: pd.DataFrame  # per bar: position, weight, span_bars, ret, gross_ret, cost, net_ret, equity
    trades: pd.DataFrame  # per trade ledger
    daily: pd.Series  # daily NET simple returns
    daily_gross: pd.Series  # daily GROSS simple returns
    cost_bps_per_side: float
    params: dict = field(default_factory=dict)


def _check_alignment(idx: pd.DatetimeIndex, ti: pd.Index) -> None:
    if not isinstance(ti, pd.DatetimeIndex) or (ti.tz is None) != (idx.tz is None):
        raise ValueError(
            "target must be indexed by the bars' DatetimeIndex (same tz-awareness); "
            f"got {type(ti).__name__} tz={getattr(ti, 'tz', None)}"
        )
    on_grid = ti.isin(idx)
    stray = (ti >= idx[0]) & (ti <= idx[-1]) & ~on_grid
    if stray.any():
        raise ValueError(f"{int(stray.sum())} target timestamps are not bar open times, first {ti[stray][0]}")
    if not on_grid.any():
        raise ValueError("target shares no timestamp with the bars")


def _validate_target(df: pd.DataFrame, target: pd.Series | float) -> np.ndarray:
    if np.isscalar(target):
        target = pd.Series(float(target), index=df.index)
    if not isinstance(target, pd.Series):
        raise TypeError("target must be a float or a pd.Series indexed by the bars' open times")
    _check_alignment(df.index, target.index)
    tgt = target.astype("float64").reindex(df.index).to_numpy()
    if np.isinf(tgt).any():
        first = df.index[np.isinf(tgt).argmax()]
        raise ValueError(f"target must be finite or NaN (flat); got inf at {first} (e.g. vol targeting with rv == 0)")
    known = tgt[~np.isnan(tgt)]
    if known.size and (known.min() < 0 or known.max() > 1):
        raise ValueError(f"long/flat only: target must be in [0, 1], got [{known.min()}, {known.max()}]")
    return np.nan_to_num(tgt, nan=0.0)


def _check_index(idx: pd.Index) -> None:
    if not isinstance(idx, pd.DatetimeIndex) or str(idx.tz) != "UTC":
        raise ValueError(
            f"bars need a UTC DatetimeIndex (daily P&L is bucketed by UTC day), got {type(idx).__name__} "
            f"tz={getattr(idx, 'tz', None)}; use .tz_convert('UTC')"
        )
    if not idx.is_monotonic_increasing or not idx.is_unique:
        raise ValueError("bars index must be strictly increasing (sorted, no duplicate timestamps)")


def _check_opens(df: pd.DataFrame) -> np.ndarray:
    """Fill prices must be real: a NaN open (a frame reindexed onto a full grid) would hide the
    move across a data gap and the gap flags with it."""
    opens = df["open"].to_numpy(dtype="float64")
    bad = ~np.isfinite(opens) | (opens <= 0)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} bars have a missing or non-positive open, first at {df.index[bad.argmax()]}. "
            "Drop missing bars instead of reindexing onto a full grid: the simulator flags gaps itself."
        )
    return opens


def _drifted(h: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Weight of a book that set weight ``h`` and then saw the asset move by ``rel``."""
    return h * rel / ((1 - h) + h * rel)


def bar_pnl(held: np.ndarray, price: np.ndarray, cost_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shares-held P&L of executed targets ``held`` on open prices ``price``.

    Returns ``(weight, gross, cost)`` per bar: the drifted weight held over bar ``j``, its gross
    return ``weight * (price[j+1] / price[j] - 1)``, and the fill cost (a fraction of equity)
    at bar ``j``'s open, charged only where ``held`` changes, on ``|held - drifted weight|``.
    """
    n = len(held)
    fill = np.ones(n, dtype=bool)
    fill[1:] = held[1:] != held[:-1]
    anchor = np.maximum.accumulate(np.where(fill, np.arange(n), 0))
    weight = _drifted(held, price / price[anchor])
    pre = np.zeros(n)
    pre[1:] = _drifted(held[:-1], price[1:] / price[anchor[:-1]])
    ret = np.zeros(n)
    ret[:-1] = price[1:] / price[:-1] - 1
    cost = np.where(fill, cost_frac * np.abs(held - pre), 0.0)
    return weight, weight * ret, cost


def _spans(idx: pd.DatetimeIndex, bar: pd.Timedelta | None) -> tuple[np.ndarray, pd.Timedelta]:
    """Open-to-next-open distance of each bar in bar lengths (1 = no gap; last bar = 1)."""
    step = np.diff(pd.DatetimeIndex(idx).as_unit("ns").asi8)
    bar_ns = pd.Timedelta(bar).value if bar is not None else int(np.median(step))
    span = np.ones(len(idx))
    span[:-1] = step / bar_ns
    return span, pd.Timedelta(bar_ns, "ns")


def _execute(
    wanted: np.ndarray, stale: np.ndarray, day: np.ndarray, max_rt: int | None, min_hold: int
) -> tuple[np.ndarray, int]:
    """Executed positions, bar by bar: no fill on a halted bar (the position is carried), then a
    per-UTC-day entry cap and a minimum holding period on the fills that actually happen.

    One sequential pass, so a deferred entry counts against the day it fills on and the holding
    period starts at the real entry bar. Returns ``(held, fills_deferred)``.
    """
    out = np.empty_like(wanted)
    prev, bars_in, entries, cur_day, deferred = 0.0, 0, 0, None, 0
    last = len(out) - 1
    for j in range(len(out)):
        if day[j] != cur_day:
            cur_day, entries = day[j], 0
        want = wanted[j]
        if stale[j] and 0 < j < last:  # halted: nothing fills here (the last bar always exits)
            deferred += int(want != prev)
            want = prev
        elif prev == 0 and want > 0:
            if max_rt is not None and entries >= max_rt:
                want = 0.0
            else:
                entries += 1
        elif prev > 0 and want == 0 and bars_in < min_hold and j != last:
            want = prev
        bars_in = bars_in + 1 if want > 0 else 0
        out[j] = prev = want
    return out, deferred


def _trade_ledger(
    idx: pd.DatetimeIndex, opens: np.ndarray, held: np.ndarray, lg: np.ndarray, ln: np.ndarray, extra: dict
) -> pd.DataFrame:
    cost, gap, stale = extra["cost"], extra["span"] > 1 + 1e-9, extra["stale"]
    in_mkt = (held > 0).astype(int)
    edges = np.diff(np.concatenate([[0], in_mkt]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    cg, cn = np.concatenate([[0.0], np.cumsum(lg)]), np.concatenate([[0.0], np.cumsum(ln)])
    gross = np.expm1(cg[ends] - cg[starts])
    net = np.exp(cn[ends] - cn[starts]) * (1 - cost[ends]) - 1
    pairs = list(zip(starts, ends, strict=True))
    return pd.DataFrame(
        {
            "entry_time": idx[starts],
            "entry_price": opens[starts],
            "exit_time": idx[ends],
            "exit_price": opens[ends],
            "holding_bars": ends - starts,
            "holding_time": idx[ends] - idx[starts],
            "crosses_gap": [bool(gap[s:e].any()) for s, e in pairs],
            "crosses_stale": [bool(stale[s : e + 1].any()) for s, e in pairs],
            "avg_position": [held[s:e].mean() for s, e in pairs],
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


def _executed(
    df: pd.DataFrame, target: pd.Series | float, max_rt: int | None, min_hold: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Executed position per bar (target shifted one bar, halts deferred, then constrained)."""
    wanted = np.concatenate([[0.0], _validate_target(df, target)[:-1]])
    wanted[-1] = 0.0  # no next open after the last bar: liquidate
    has_vol = "volume" in df.columns
    stale = (df["volume"].to_numpy(dtype="float64") <= 0) if has_vol else np.zeros(len(df), dtype=bool)
    held, deferred = _execute(wanted, stale, df.index.floor("D").asi8, max_rt, min_hold)
    info = {
        "stale_check": has_vol,
        "n_stale_bars": int(stale.sum()),
        "fills_deferred": deferred,
        "bars_held_across_stale": int((stale & (held > 0)).sum()),
        "forced_exit_on_stale": bool(stale[-1] and held[-2] > 0),
    }
    return held, stale, info


def run_backtest(
    df: pd.DataFrame,
    target: pd.Series | float,
    cost_bps_per_side: float = config.BASE_COST_BPS,
    max_round_trips_per_day: int | None = None,
    min_hold_bars: int = 0,
    bar: pd.Timedelta | None = None,
) -> BacktestResult:
    """Simulate ``target`` (decided at each bar's close) on ``df``'s ``open`` prices."""
    if len(df) < 2:
        raise ValueError("need at least two bars")
    _check_index(df.index)
    assert_pre_holdout(df.index, "run_backtest data")
    idx = df.index
    opens = _check_opens(df)
    held, stale, stale_info = _executed(df, target, max_round_trips_per_day, min_hold_bars)
    weight, gross, cost = bar_pnl(held, opens, cost_bps_per_side / 1e4)
    span, bar_len = _spans(idx, bar)
    ret = np.zeros(len(df))
    ret[:-1] = opens[1:] / opens[:-1] - 1
    lg, ln = np.log1p(gross), np.log1p(gross) + np.log1p(-cost)
    if not np.isfinite(ln).all():  # backstop: _daily's groupby-sum would turn NaN P&L into 0
        raise ValueError(f"non-finite bar P&L at {idx[~np.isfinite(ln)][0]}; check prices and target")
    bars = pd.DataFrame(
        {
            "position": held,
            "weight": weight,
            "span_bars": span,
            "ret": ret,
            "gross_ret": gross,
            "cost": cost,
            "net_ret": np.expm1(ln),
            "equity": np.exp(np.cumsum(ln)),
        },
        index=idx,
    )
    gap = span > 1 + 1e-9
    params = {
        "max_round_trips_per_day": max_round_trips_per_day,
        "min_hold_bars": min_hold_bars,
        "bar": str(bar_len),
        "n_gaps": int(gap.sum()),
        "bars_held_across_gap": int((gap & (held > 0)).sum()),
        **stale_info,
    }
    return BacktestResult(
        bars=bars,
        trades=_trade_ledger(idx, opens, held, lg, ln, {"cost": cost, "span": span, "stale": stale}),
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
    """Buy-and-hold of a fixed fraction ``w`` (shares held, so the weight drifts) scaled to the
    strategy's daily volatility (``w`` capped at 1).

    Returns the result and the weight used. The weight uses the full-period vol of both
    series, i.e. it is a benchmark, not a tradable strategy.
    """
    bh = buy_and_hold(df, cost_bps_per_side)
    common = bh.daily.index.intersection(strategy_daily.index)
    bh_sd = bh.daily.loc[common].std()
    w = float(np.clip(strategy_daily.loc[common].std() / bh_sd, 0.0, 1.0)) if bh_sd > 0 else 0.0
    return run_backtest(df, w, cost_bps_per_side), w
