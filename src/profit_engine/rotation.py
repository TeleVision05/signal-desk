"""Weekly cross-sectional momentum rotation with stop-loss (real ETF data)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .daily_backtest import StopConfig, _stop_hit
from .data import fetch_ohlcv


UNIVERSE = ["QQQ", "SPY", "IWM", "EFA", "EEM", "TLT", "GLD", "XLE", "XLF", "SMH"]


@dataclass
class RotationResult:
    equity: pd.Series
    trades: pd.DataFrame
    holdings: pd.Series


def run_rotation(
    *,
    start: str = "2012-01-01",
    lookback: int = 63,
    rebalance_every: int = 5,  # trading days
    top_n: int = 1,
    initial_cash: float = 100_000.0,
    cost_bps: float = 10.0,
    hard_pct: float = 0.08,
    trailing_pct: float = 0.12,
    abs_filter: bool = True,
) -> RotationResult:
    panels = {t: fetch_ohlcv(t, start=start) for t in UNIVERSE}
    # Align closes
    closes = pd.DataFrame({t: panels[t]["Close"] for t in UNIVERSE}).dropna(how="any")
    dates = closes.index
    half = cost_bps / 10_000 / 2
    stop = StopConfig(hard_pct=hard_pct, trailing_pct=trailing_pct)

    cash = float(initial_cash)
    shares = 0.0
    asset: str | None = None
    entry = peak = 0.0
    equity_rows = []
    hold_rows = []
    trade_rows = []

    # First valid day after lookback
    start_i = lookback
    for i in range(start_i, len(dates)):
        dt = dates[i]
        # Stop check using underlying OHLC
        if asset and shares > 0:
            bar = panels[asset].loc[dt]
            peak = max(peak, float(bar["High"]))
            hit, exit_px = _stop_hit(bar, entry, peak, stop)
            if hit and exit_px is not None:
                cost = shares * exit_px * half
                cash += shares * exit_px - cost
                trade_rows.append(
                    {"date": dt, "side": "SELL", "reason": "stop", "asset": asset, "shares": shares, "price": exit_px, "cost": cost}
                )
                shares = 0.0
                asset = None
                entry = peak = 0.0

        # Rebalance on schedule
        if (i - start_i) % rebalance_every == 0:
            window = closes.iloc[i - lookback : i]
            rets = window.iloc[-1] / window.iloc[0] - 1.0
            ranked = rets.sort_values(ascending=False)
            if abs_filter:
                ranked = ranked[ranked > 0]
            target = ranked.index[0] if len(ranked) else None
            if top_n > 1 and len(ranked) >= top_n:
                # equal-weight top_n — keep simple: still pick #1 for v1
                target = ranked.index[0]

            # Switch if needed
            if target != asset:
                if asset and shares > 0:
                    px = float(panels[asset].loc[dt, "Open"])
                    cost = shares * px * half
                    cash += shares * px - cost
                    trade_rows.append(
                        {"date": dt, "side": "SELL", "reason": "rebalance", "asset": asset, "shares": shares, "price": px, "cost": cost}
                    )
                    shares = 0.0
                    asset = None
                if target is not None and cash > 0:
                    px = float(panels[target].loc[dt, "Open"])
                    cost = cash * half
                    shares = (cash - cost) / px
                    cash = 0.0
                    asset = target
                    entry = peak = px
                    trade_rows.append(
                        {"date": dt, "side": "BUY", "reason": "rebalance", "asset": asset, "shares": shares, "price": px, "cost": cost}
                    )

        mark = cash
        if asset and shares > 0:
            mark += shares * float(panels[asset].loc[dt, "Close"])
        equity_rows.append({"date": dt, "equity": mark})
        hold_rows.append({"date": dt, "asset": asset or "CASH"})

    eq = pd.DataFrame(equity_rows).set_index("date")["equity"]
    holds = pd.DataFrame(hold_rows).set_index("date")["asset"]
    trades = pd.DataFrame(trade_rows)
    return RotationResult(equity=eq, trades=trades, holdings=holds)
