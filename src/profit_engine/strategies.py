"""Pure signal generators — no look-ahead beyond month-end close."""

from __future__ import annotations

import pandas as pd


def gem_dual_momentum(
    month_closes: pd.DataFrame,
    *,
    lookback: int = 12,
    absolute_cutoff: float = 0.0,
    risk_on: tuple[str, str] = ("SPY", "VEU"),
    risk_off: str = "AGG",
) -> pd.Series:
    """
    Classic GEM dual momentum (Antacci-style):
    1) Relative: pick stronger of risk_on assets by lookback total return
    2) Absolute: hold winner only if lookback return > absolute_cutoff, else risk_off
    Signal is known at month-end close; execution is next open (handled by backtester).
    """
    a, b = risk_on
    rets = month_closes.pct_change(lookback)
    signals = []
    for dt, row in rets.iterrows():
        if pd.isna(row[a]) or pd.isna(row[b]) or pd.isna(row.get(risk_off, float("nan"))):
            signals.append(None)
            continue
        winner = a if row[a] >= row[b] else b
        asset = winner if row[winner] > absolute_cutoff else risk_off
        signals.append(asset)
    out = pd.Series(signals, index=month_closes.index, name="gem")
    return out.dropna()


def faber_sma(
    month_closes: pd.DataFrame,
    *,
    asset: str = "SPY",
    cash: str = "BIL",
    sma_months: int = 10,
) -> pd.Series:
    """Faber 10-month SMA timing: risk asset if price > SMA else cash proxy."""
    px = month_closes[asset]
    sma = px.rolling(sma_months).mean()
    signals = []
    for dt in month_closes.index:
        if pd.isna(sma.loc[dt]):
            signals.append(None)
            continue
        signals.append(asset if px.loc[dt] > sma.loc[dt] else cash)
    out = pd.Series(signals, index=month_closes.index, name="faber")
    return out.dropna()


def buy_and_hold(month_closes: pd.DataFrame, asset: str = "SPY") -> pd.Series:
    return pd.Series(asset, index=month_closes.index, name="bh")