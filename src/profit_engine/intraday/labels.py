"""Labels measured the way the simulator trades.

A decision made at bar ``t``'s close is filled at bar ``t+1``'s OPEN, so every label starts
at ``open[t+1]``. Each label also returns ``t1``: the open time of the first bar at which
the label is fully known. Purging (``splits.py``) drops training rows whose ``t1`` reaches
into a test block. Horizons are in bars; rows whose label would need bars past the end of
the frame are NaN (``t1`` NaT), never truncated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _shift(values: np.ndarray, k: int, fill: float = np.nan) -> np.ndarray:
    """values[t + k] at position t (k >= 0), ``fill`` past the end."""
    out = np.full(len(values), fill, dtype="float64")
    if k < len(values):
        out[: len(values) - k] = values[k:]
    return out


def _shift_index(index: pd.DatetimeIndex, k: int) -> pd.Series:
    """index[t + k] at position t, NaT past the end."""
    return pd.Series(index, index=index).shift(-k)


def forward_return(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """``ret`` = log(open[t+1+h] / open[t+1]); ``t1`` = open time of bar t+1+h."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1 bar")
    o = df["open"].to_numpy(dtype="float64")
    ret = np.log(_shift(o, 1 + horizon) / _shift(o, 1))
    return pd.DataFrame({"ret": ret, "t1": _shift_index(df.index, 1 + horizon)}, index=df.index)


def ex_ante_sigma(df: pd.DataFrame, window: int, horizon: int) -> pd.Series:
    """Std of 1-bar log close returns over the last ``window`` bars (known at t), x sqrt(h)."""
    r1 = np.log(df["close"]).diff()
    return r1.rolling(window, min_periods=window).std() * np.sqrt(horizon)


def triple_barrier(
    df: pd.DataFrame,
    horizon: int,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    vol_window: int = 288,
) -> pd.DataFrame:
    """Triple-barrier labels for a long entry at ``open[t+1]``.

    Barriers: ``entry * exp(+pt_mult * sigma_t)`` and ``entry * exp(-sl_mult * sigma_t)``
    with ``sigma_t`` from :func:`ex_ante_sigma`; vertical barrier at ``open[t+1+h]``.
    Bars ``t+1 .. t+h`` are scanned with their high/low. If one bar touches both barriers the
    stop is assumed first (conservative). A stop gapped through at a later bar's open exits at
    that open. Columns: ``label`` (+1 profit-take, -1 stop, 0 vertical), ``ret`` (log return
    realised), ``barrier`` ("pt"/"sl"/"vb"), ``sigma``, ``t1``.
    """
    n = len(df)
    o, hi, lo = (df[c].to_numpy(dtype="float64") for c in ("open", "high", "low"))
    sigma = ex_ante_sigma(df, vol_window, horizon).to_numpy()
    entry = _shift(o, 1)
    upper, lower = entry * np.exp(pt_mult * sigma), entry * np.exp(-sl_mult * sigma)
    label = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    end_pos = np.full(n, -1, dtype="int64")
    done = ~np.isfinite(upper) | ~np.isfinite(lower)
    for k in range(1, horizon + 1):
        h_k, l_k, o_k = _shift(hi, k), _shift(lo, k), _shift(o, k)
        live = ~done & np.isfinite(h_k)
        sl = live & (l_k <= lower)
        pt = live & ~sl & (h_k >= upper)
        exit_sl = np.minimum(lower, o_k)  # o_1 is the entry, so this only bites on gaps
        label[sl], ret[sl] = -1.0, np.log(exit_sl[sl] / entry[sl])
        label[pt], ret[pt] = 1.0, np.log(upper[pt] / entry[pt])
        end_pos[sl | pt] = np.flatnonzero(sl | pt) + k + 1  # info complete at next bar's open
        done |= sl | pt
    vb = ~done & np.isfinite(_shift(o, 1 + horizon))
    label[vb], ret[vb] = 0.0, np.log(_shift(o, 1 + horizon)[vb] / entry[vb])
    end_pos[vb] = np.flatnonzero(vb) + 1 + horizon
    valid = (end_pos >= 0) & (end_pos < n)
    label[~valid], ret[~valid] = np.nan, np.nan
    t1 = pd.Series(pd.NaT, index=df.index, dtype=df.index.dtype)
    t1[valid] = df.index[end_pos[valid]]
    barrier = np.select([label == 1, label == -1, label == 0], ["pt", "sl", "vb"], default="")
    return pd.DataFrame({"label": label, "ret": ret, "barrier": barrier, "sigma": sigma, "t1": t1}, index=df.index)
