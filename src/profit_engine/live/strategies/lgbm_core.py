"""LightGBM triple-barrier model, ported from ``experiments/intraday/lgbm.py`` for live use.

Everything here mirrors one research function (named in each docstring) with the same
expressions, so a model trained here on the same bars is the research fold's model, and its
threshold the research fold's threshold (``tests/live/test_live_lgbm_parity.py``).

Why a port and not an import: ``experiments/`` is not a package, and the harness label and
inner-CV functions (``labels.triple_barrier``, ``labels.forward_return``,
``splits.purged_kfold``) refuse any bar at or after ``HOLDOUT_START``, which is right for
research and wrong for a production model trained on last month's bars. The ports below are
the same code without that research guard; ``splits.purge_and_embargo`` and
``features.make_feature_frame`` are unguarded and are imported as they are.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from profit_engine.intraday import config, features
from profit_engine.intraday.splits import DEFAULT_EMBARGO, purge_and_embargo

SEED = 7
BARS_PER_DAY = {"15m": 96, "1h": 24, "4h": 6}
LABEL_MIN_RET = 2 * config.BASE_COST_BPS / 1e4  # y = 1 iff the barrier return beats one round trip
INNER_SPLITS = 4
MIN_TRAIN_ROWS = 2000
QUANTILES = np.round(np.arange(0.20, 0.951, 0.05), 2)
MODEL_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 500,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 10.0,
    "reg_alpha": 1.0,
    "min_split_gain": 0.0,
}
CONFIGS: dict[str, dict] = {"lgbm_4h_h30_pt2_sl1": {"tf": "4h", "h": 30, "pt": 2.0, "sl": 1.0}}


# --------------------------------------------------------------------------- features


def feature_params(tf: str) -> tuple[dict, tuple[int, ...]]:
    """Mirrors ``lgbm.feature_params``: feature windows scaled to the timeframe."""
    d = BARS_PER_DAY[tf]
    q = max(1, d // 6)
    params = {
        "log_returns": {"horizons": tuple(sorted({1, q, d, 7 * d, 30 * d}))},
        "realized_vol": {"windows": (d, 7 * d)},
        "return_zscore": {"horizon": q, "window": 7 * d},
        "taker_imbalance": {"windows": (1, d)},
        "vwap_deviation": {"windows": (d, 7 * d)},
        "volume_ratio": {"short": max(2, q), "long": 7 * d},
    }
    return params, (1, d, 7 * d)


def feature_frame(df: pd.DataFrame, other: pd.DataFrame, tf: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The feature half of ``lgbm.build_xy``: (bars on the shared index, features)."""
    d = BARS_PER_DAY[tf]
    fparams, cross = feature_params(tf)
    x = features.make_feature_frame(df, other, params=fparams, cross_horizons=cross)
    df = df.loc[x.index]
    close = df["close"]
    for n in (7 * d, 30 * d):  # causal: adjust=False EWM uses bars <= t only
        x[f"emadist_{n}"] = np.log(close / close.ewm(span=n, adjust=False, min_periods=n).mean())
    x["rvratio"] = np.log(x[f"rv_{d}"] / x[f"rv_{7 * d}"].where(x[f"rv_{7 * d}"] > 0))
    return df, x.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- labels


def _shift(values: np.ndarray, k: int, fill: float = np.nan) -> np.ndarray:
    """Mirrors ``labels._shift``: values[t + k] at position t, ``fill`` past the end."""
    out = np.full(len(values), fill, dtype="float64")
    if k < len(values):
        out[: len(values) - k] = values[k:]
    return out


def forward_return_1(df: pd.DataFrame) -> np.ndarray:
    """Mirrors ``labels.forward_return(df, 1)['ret']``: log(open[t+2] / open[t+1])."""
    o = df["open"].to_numpy(dtype="float64")
    return np.log(_shift(o, 2) / _shift(o, 1))


def triple_barrier(df: pd.DataFrame, horizon: int, pt_mult: float, sl_mult: float, vol_window: int) -> pd.DataFrame:
    """Mirrors ``labels.triple_barrier`` (columns label, ret, t1), without the holdout guard."""
    n = len(df)
    o, hi, lo = (df[c].to_numpy(dtype="float64") for c in ("open", "high", "low"))
    r1 = np.log(df["close"]).diff()
    sigma = (r1.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(horizon)).to_numpy()
    entry = _shift(o, 1)
    upper, lower = entry * np.exp(pt_mult * sigma), entry * np.exp(-sl_mult * sigma)
    label, ret = np.full(n, np.nan), np.full(n, np.nan)
    end_pos = np.full(n, -1, dtype="int64")
    done = ~np.isfinite(upper) | ~np.isfinite(lower)
    for k in range(1, horizon + 1):
        h_k, l_k, o_k = _shift(hi, k), _shift(lo, k), _shift(o, k)
        live = ~done & np.isfinite(h_k)
        sl = live & (l_k <= lower)
        pt = live & ~sl & (h_k >= upper)
        exit_sl = np.minimum(lower, o_k)
        label[sl], ret[sl] = -1.0, np.log(exit_sl[sl] / entry[sl])
        label[pt], ret[pt] = 1.0, np.log(upper[pt] / entry[pt])
        end_pos[sl | pt] = np.flatnonzero(sl | pt) + k + 1
        done |= sl | pt
    vb = ~done & np.isfinite(_shift(o, 1 + horizon))
    label[vb], ret[vb] = 0.0, np.log(_shift(o, 1 + horizon)[vb] / entry[vb])
    end_pos[vb] = np.flatnonzero(vb) + 1 + horizon
    valid = (end_pos >= 0) & (end_pos < n)
    label[~valid], ret[~valid] = np.nan, np.nan
    t1 = pd.Series(pd.NaT, index=df.index, dtype=df.index.dtype)
    t1[valid] = df.index[end_pos[valid]]
    return pd.DataFrame({"label": label, "ret": ret, "t1": t1}, index=df.index)


def build_xy(df: pd.DataFrame, other: pd.DataFrame, cfg: dict) -> dict:
    """Mirrors ``lgbm.build_xy``: features, labels, weight inputs and 1-bar return."""
    d = BARS_PER_DAY[cfg["tf"]]
    df, x = feature_frame(df, other, cfg["tf"])
    tb = triple_barrier(df, cfg["h"], cfg["pt"], cfg["sl"], vol_window=7 * d)
    end_pos = df.index.get_indexer(tb["t1"])
    return {
        "df": df,
        "x": x,
        "y": (tb["ret"] > LABEL_MIN_RET).astype(int).to_numpy(),
        "t1": tb["t1"],
        "end_pos": end_pos,
        "r1": np.nan_to_num(forward_return_1(df)),
        "has_x": x.notna().all(axis=1).to_numpy(),
        "has_y": tb["label"].notna().to_numpy() & (end_pos >= 0),
    }


def uniqueness_weights(rows: np.ndarray, end_pos: np.ndarray, n: int) -> np.ndarray:
    """Mirrors ``lgbm.uniqueness_weights``: mean 1/concurrency over each label's span."""
    start, stop = rows + 1, np.maximum(end_pos[rows], rows + 2)
    conc = np.zeros(n + 1)
    np.add.at(conc, start, 1.0)
    np.add.at(conc, np.minimum(stop, n), -1.0)
    conc = np.cumsum(conc)[:n]
    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1.0), 0.0)
    cum = np.concatenate([[0.0], np.cumsum(inv)])
    stop = np.minimum(stop, n)
    w = (cum[stop] - cum[start]) / np.maximum(stop - start, 1)
    return w / w.mean()


# --------------------------------------------------------------------------- model + threshold


def make_model() -> lgb.LGBMClassifier:
    """Mirrors ``lgbm.make_model``: fixed seed, deterministic, 4 threads."""
    return lgb.LGBMClassifier(
        **MODEL_PARAMS,
        objective="binary",
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        n_jobs=4,
        verbose=-1,
    )


def block_pnl(p: np.ndarray, r1: np.ndarray, block: np.ndarray, thr: float, cost: float) -> float:
    """Mirrors ``lgbm.block_pnl``: summed net log P&L of ``p > thr``, each block starting flat."""
    pos = (p > thr).astype(float)
    turn = np.abs(np.diff(pos, prepend=0.0))
    starts = np.flatnonzero(np.diff(block, prepend=block[0] - 1) != 0)
    turn[starts] = pos[starts]
    return float((pos * r1).sum() - cost * turn.sum())


def tune_threshold(oof: np.ndarray, r1: np.ndarray, block: np.ndarray) -> tuple[float, float]:
    """Mirrors ``lgbm.tune_threshold``: best OOF quantile threshold, inf if none makes money."""
    cost = config.BASE_COST_BPS / 1e4
    cands = np.quantile(oof, QUANTILES)
    pnl = np.array([block_pnl(oof, r1, block, t, cost) for t in cands])
    k = int(np.argmax(pnl))
    if pnl[k] <= 0:
        return np.inf, float("nan")
    return float(cands[k]), float(QUANTILES[k])


def purged_kfold(index: pd.DatetimeIndex, t1: pd.Series, n_splits: int, block_start: pd.Timestamp) -> list:
    """Mirrors ``splits.purged_kfold(..., within=fold)`` without the holdout guard: contiguous
    folds, purged and embargoed; raises if a row or label end reaches ``block_start``."""
    ends = pd.Series(t1).reindex(index)
    late = (index >= block_start) | ends.isna().to_numpy() | (ends >= block_start).to_numpy()
    if late.any():
        raise ValueError(f"{int(late.sum())} training rows reach the month the model will trade ({block_start})")
    n = len(index)
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    out = []
    for k in range(n_splits):
        test = np.zeros(n, dtype=bool)
        test[bounds[k] : bounds[k + 1]] = True
        start = index[bounds[k]]
        end = index[bounds[k + 1]] if bounds[k + 1] < n else index[-1] + pd.Timedelta(1, "ns")
        out.append((purge_and_embargo(index, t1, start, end, DEFAULT_EMBARGO) & ~test, test))
    return out


def fit_fold(xy: dict, train: np.ndarray, block_start: pd.Timestamp) -> tuple[lgb.LGBMClassifier, float, float]:
    """Mirrors ``lgbm.fit_fold``: inner purged CV on the training rows picks the threshold;
    then the model is refit on all of them."""
    rows = np.flatnonzero(train)
    idx = xy["df"].index[rows]
    t1 = pd.Series(xy["t1"].to_numpy()[rows], index=idx)
    x_tr, y_tr, r1_tr = xy["x"].iloc[rows], xy["y"][rows], xy["r1"][rows]
    w = uniqueness_weights(rows, xy["end_pos"], len(xy["df"]))
    oof, block = np.full(len(rows), np.nan), np.full(len(rows), -1)
    for k, (itr, iva) in enumerate(purged_kfold(idx, t1, INNER_SPLITS, block_start)):
        m = make_model().fit(x_tr.iloc[itr], y_tr[itr], sample_weight=w[itr])
        oof[iva], block[iva] = m.predict_proba(x_tr.iloc[iva])[:, 1], k
    thr, q = tune_threshold(oof, r1_tr, block)
    final = make_model().fit(x_tr, y_tr, sample_weight=w)
    return final, thr, q


def training_rows(xy: dict, cutoff: pd.Timestamp) -> np.ndarray:
    """Mirrors the training mask of ``lgbm.oos_positions`` for the month starting at ``cutoff``
    (``splits.walk_forward_splits`` expanding from TRAIN_START, purged, 1-day embargo)."""
    index = xy["df"].index
    t = pd.DatetimeIndex(index).as_unit("ns").asi8
    end = cutoff + pd.DateOffset(months=1)
    tr = purge_and_embargo(index, xy["t1"], cutoff, end, DEFAULT_EMBARGO)
    tr &= (t >= config.TRAIN_START.value) & (t < cutoff.value)
    if tr.sum() < MIN_TRAIN_ROWS:
        raise ValueError(f"only {int(tr.sum())} training bars before {cutoff:%Y-%m-%d}; need {MIN_TRAIN_ROWS}")
    train = tr & xy["has_x"] & xy["has_y"]
    if (xy["t1"].to_numpy()[train] >= cutoff).any():
        raise RuntimeError(f"training label reaches the month starting {cutoff:%Y-%m-%d}")
    return train
