"""Causal feature library.

Every function is a pure function of the bar frame: the value stamped at bar ``t`` uses only
bars ``<= t`` (including bar ``t``'s own close, which is known when the decision is made at
``t``'s close; the simulator fills at ``t+1``'s open). Windows and horizons are in BARS of
whatever timeframe the frame has, so scale them when you change timeframe (e.g. 288 bars =
1 day at 5m, 24 at 1h).

``tests/intraday/test_features_causal.py`` recomputes every function in :data:`FEATURES` on
truncated frames and requires the overlapping rows to be identical. A new feature function
must be added to :data:`FEATURES` so that test covers it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def _log_close(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"])


def log_returns(df: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 12, 48)) -> pd.DataFrame:
    """``ret_h`` = log(close_t / close_{t-h})."""
    lc = _log_close(df)
    return pd.DataFrame({f"ret_{h}": lc.diff(h) for h in horizons}, index=df.index)


def realized_vol(df: pd.DataFrame, windows: tuple[int, ...] = (12, 48, 288)) -> pd.DataFrame:
    """``rv_w`` = rolling std of 1-bar log returns over the last ``w`` bars."""
    r1 = _log_close(df).diff()
    return pd.DataFrame({f"rv_{w}": r1.rolling(w, min_periods=w).std() for w in windows}, index=df.index)


def return_zscore(df: pd.DataFrame, horizon: int = 12, window: int = 288) -> pd.DataFrame:
    """``z_h_w`` = (h-bar log return - its rolling mean) / its rolling std over ``window`` bars."""
    rh = _log_close(df).diff(horizon)
    roll = rh.rolling(window, min_periods=window)
    z = (rh - roll.mean()) / roll.std()
    return pd.DataFrame({f"z_{horizon}_{window}": z}, index=df.index)


def taker_imbalance(df: pd.DataFrame, windows: tuple[int, ...] = (1, 12, 48)) -> pd.DataFrame:
    """``tib_w`` = 2 * taker-buy base volume / total base volume - 1 over ``w`` bars (in [-1, 1])."""
    out = {}
    for w in windows:
        buy = df["taker_buy_base"].rolling(w, min_periods=w).sum()
        vol = df["volume"].rolling(w, min_periods=w).sum()
        out[f"tib_{w}"] = (2 * buy / vol.where(vol > 0) - 1).clip(-1, 1)
    return pd.DataFrame(out, index=df.index)


def vwap_deviation(df: pd.DataFrame, windows: tuple[int, ...] = (12, 48, 288)) -> pd.DataFrame:
    """``vwapdev_w`` = log(close / rolling VWAP over ``w`` bars), VWAP = sum(quote) / sum(base)."""
    out = {}
    for w in windows:
        quote = df["quote_volume"].rolling(w, min_periods=w).sum()
        base = df["volume"].rolling(w, min_periods=w).sum()
        vwap = quote / base.where(base > 0)
        out[f"vwapdev_{w}"] = np.log(df["close"] / vwap)
    return pd.DataFrame(out, index=df.index)


def volume_ratio(df: pd.DataFrame, short: int = 12, long: int = 288) -> pd.DataFrame:
    """``volratio_s_l`` = log(mean volume over ``short`` bars / mean volume over ``long`` bars)."""
    s = df["volume"].rolling(short, min_periods=short).mean()
    lo = df["volume"].rolling(long, min_periods=long).mean()
    ratio = np.log(s.where(s > 0) / lo.where(lo > 0))
    return pd.DataFrame({f"volratio_{short}_{long}": ratio}, index=df.index)


def range_location(df: pd.DataFrame) -> pd.DataFrame:
    """``hl_range`` = log(high/low); ``close_loc`` = (close-low)/(high-low), 0.5 on a flat bar."""
    span = df["high"] - df["low"]
    loc = ((df["close"] - df["low"]) / span.where(span > 0)).fillna(0.5)
    return pd.DataFrame({"hl_range": np.log(df["high"] / df["low"]), "close_loc": loc}, index=df.index)


def calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical hour-of-day and day-of-week of the bar's OPEN time (UTC)."""
    idx = df.index
    hour = idx.hour + idx.minute / 60.0
    dow = idx.dayofweek + hour / 24.0
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        },
        index=idx,
    )


def cross_asset(df: pd.DataFrame, other: pd.DataFrame, horizons: tuple[int, ...] = (1, 12, 48)) -> pd.DataFrame:
    """Other symbol's ``x_ret_h`` and the spread ``rel_ret_h`` = own - other, on ``df``'s index.

    ``other`` is reindexed to ``df.index`` without filling, so a bar missing in ``other`` gives
    NaN rather than a stale or future value. Use :func:`make_feature_frame` to align first.
    """
    own = _log_close(df)
    oth = _log_close(other).reindex(df.index)
    out = {}
    for h in horizons:
        out[f"x_ret_{h}"] = oth.diff(h)
        out[f"rel_ret_{h}"] = own.diff(h) - oth.diff(h)
    return pd.DataFrame(out, index=df.index)


# Single-frame feature functions (name -> callable(df)). The causality test iterates this.
FEATURES: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "log_returns": log_returns,
    "realized_vol": realized_vol,
    "return_zscore": return_zscore,
    "taker_imbalance": taker_imbalance,
    "vwap_deviation": vwap_deviation,
    "volume_ratio": volume_ratio,
    "range_location": range_location,
    "calendar": calendar,
}


def align(primary: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to the bars they share (inner join on open time)."""
    idx = primary.index.intersection(other.index)
    return primary.loc[idx], other.loc[idx]


def make_feature_frame(
    primary: pd.DataFrame,
    other: pd.DataFrame | None = None,
    params: dict[str, dict] | None = None,
    cross_horizons: tuple[int, ...] = (1, 12, 48),
) -> pd.DataFrame:
    """All features for ``primary`` (plus cross-asset ones vs ``other``) on a common index.

    ``params`` maps a :data:`FEATURES` name to keyword overrides, e.g.
    ``{"realized_vol": {"windows": (6, 24)}}``; mapping a name to ``None`` skips it.
    Rows are NOT dropped for warm-up NaNs: the caller decides (LightGBM accepts NaN).
    """
    params = params or {}
    if other is not None:
        primary, other = align(primary, other)
    parts = [
        fn(primary, **(params.get(name) or {}))
        for name, fn in FEATURES.items()
        if name not in params or params[name] is not None
    ]
    if other is not None:
        parts.append(cross_asset(primary, other, cross_horizons))
    return pd.concat(parts, axis=1)
