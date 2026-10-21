"""Volatility-targeted exposure on a single ETF (classic risk-managed beta)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import fetch_ohlcv


def run_vol_target(
    ticker: str = "QQQ",
    *,
    start: str = "2012-01-01",
    target_vol: float = 0.12,
    vol_lookback: int = 20,
    max_leverage: float = 2.0,
    initial_cash: float = 100_000.0,
    cost_bps: float = 5.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Daily: leverage = clip(target_vol / realized_vol, 0, max_leverage).
    Rebalance when leverage changes by >10% relative, pay costs on turnover.
    """
    df = fetch_ohlcv(ticker, start=start)
    close = df["Close"].astype(float)
    ret = close.pct_change()
    realized = ret.rolling(vol_lookback).std() * np.sqrt(252)
    lev = (target_vol / realized).clip(lower=0, upper=max_leverage)
    lev = lev.fillna(0)

    cash = float(initial_cash)
    shares = 0.0
    equity = []
    trades = []
    half = cost_bps / 10_000 / 2
    prev_lev = 0.0

    for dt, row in df.iterrows():
        if dt not in lev.index or pd.isna(lev.loc[dt]):
            equity.append((dt, cash + shares * float(row["Close"])))
            continue
        px = float(row["Close"])
        eq = cash + shares * px
        target_lev = float(lev.loc[dt])
        # Rebalance if material change
        if prev_lev == 0 or abs(target_lev - prev_lev) / max(prev_lev, 1e-6) > 0.10:
            target_notional = eq * target_lev
            cur_notional = shares * px
            delta = target_notional - cur_notional
            if abs(delta) > 1.0:
                cost = abs(delta) * half * 2  # approx one-way on delta
                # pay cost from equity
                eq -= cost
                shares = (eq * target_lev) / px if px else 0.0
                cash = eq - shares * px
                trades.append({"date": dt, "lev": target_lev, "shares": shares, "cost": cost})
            prev_lev = target_lev
        equity.append((dt, cash + shares * float(row["Close"])))

    eq_s = pd.Series({d: e for d, e in equity}).sort_index()
    return eq_s, pd.DataFrame(trades)
