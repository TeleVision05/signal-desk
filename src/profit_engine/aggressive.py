"""Aggressive weekly/daily long-only signal generators (with stops applied in backtester)."""

from __future__ import annotations

import pandas as pd


def sma_trend(close: pd.Series, window: int = 50) -> pd.Series:
    """1 when close > SMA, else 0. Signal at close."""
    sma = close.rolling(window).mean()
    return (close > sma).astype(int).rename("sma_trend")


def dual_sma(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    return ((close > f) & (f > s)).astype(int).rename("dual_sma")


def weekly_momentum(close: pd.Series, lookback_days: int = 20, hold_above: float = 0.0) -> pd.Series:
    """Long when N-day return > hold_above."""
    ret = close.pct_change(lookback_days)
    return (ret > hold_above).astype(int).rename("mom")


def breakout(close: pd.Series, lookback: int = 20) -> pd.Series:
    """Long when close makes N-day high."""
    hh = close.rolling(lookback).max()
    return (close >= hh).astype(int).rename("breakout")


def rsi_pullback(close: pd.Series, period: int = 14, buy_below: float = 35, exit_above: float = 55) -> pd.Series:
    """
    Sticky RSI regime: enter when RSI crosses below buy_below, stay long until RSI > exit_above.
    """
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / down.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    state = 0
    out = []
    for v in rsi.fillna(50):
        if state == 0 and v <= buy_below:
            state = 1
        elif state == 1 and v >= exit_above:
            state = 0
        out.append(state)
    return pd.Series(out, index=close.index, name="rsi")
