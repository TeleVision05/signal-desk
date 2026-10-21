"""Real-price feature engineering for walk-forward ML (no look-ahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(ohlc: pd.DataFrame, *, extras: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """
    All features use only information available at that bar's close.
    `extras` are other close series aligned by date (e.g. SPY, VIX).
    """
    c = ohlc["Close"].astype(float)
    h = ohlc["High"].astype(float)
    l = ohlc["Low"].astype(float)
    v = ohlc["Volume"].astype(float)
    o = ohlc["Open"].astype(float)

    feat = pd.DataFrame(index=ohlc.index)
    for n in (1, 2, 3, 5, 10, 20, 60):
        feat[f"ret_{n}"] = c.pct_change(n)
    feat["vol_10"] = c.pct_change().rolling(10).std()
    feat["vol_20"] = c.pct_change().rolling(20).std()
    feat["vol_ratio"] = feat["vol_10"] / feat["vol_20"].replace(0, np.nan)
    for n in (10, 20, 50, 100, 200):
        sma = c.rolling(n).mean()
        feat[f"dist_sma_{n}"] = c / sma - 1.0
    feat["rsi_14"] = _rsi(c, 14) / 100.0
    feat["rsi_7"] = _rsi(c, 7) / 100.0
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    feat["atr_pct"] = atr / c
    feat["hl_range"] = (h - l) / c
    feat["oc_ret"] = c / o - 1.0
    feat["vol_z_20"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    # Donchian position
    for n in (20, 55):
        hh = h.rolling(n).max()
        ll = l.rolling(n).min()
        feat[f"donch_{n}"] = (c - ll) / (hh - ll).replace(0, np.nan)

    if extras:
        for name, series in extras.items():
            s = series.reindex(ohlc.index).ffill()
            feat[f"x_{name}_ret_5"] = s.pct_change(5)
            feat[f"x_{name}_ret_20"] = s.pct_change(20)
            feat[f"x_{name}_corr_20"] = c.pct_change().rolling(20).corr(s.pct_change())

    return feat.replace([np.inf, -np.inf], np.nan)


def make_labels(
    close: pd.Series,
    *,
    horizon: int = 5,
    mode: str = "excess_vol",
) -> pd.Series:
    """
    Binary label known only after `horizon` days — used for training with shift.
    mode:
      - sign: forward return > 0
      - excess_vol: forward return > 0.5 * realized vol * sqrt(horizon)
    """
    fwd = close.shift(-horizon) / close - 1.0
    if mode == "sign":
        y = (fwd > 0).astype(float)
    else:
        vol = close.pct_change().rolling(20).std()
        thresh = 0.35 * vol * np.sqrt(horizon)
        y = (fwd > thresh).astype(float)
    y.name = "y"
    # Last `horizon` rows are unknown
    y.iloc[-horizon:] = np.nan
    return y
