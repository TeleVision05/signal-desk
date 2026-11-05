"""Walk-forward and purged cross-validation splits.

Samples are rows of a bar-indexed frame; each has a label end time ``t1`` (from
``labels.py``). A training row is usable for a test block ``[b, e)`` only if

* it is not in the block,
* its label does not reach the block: ``t1 < b`` for rows before it (rows with unknown
  ``t1`` are purged), and rows inside ``[b, e)`` are never training rows,
* it is not in the embargo window ``[e, e + embargo)`` after the block.

The outer walk-forward only trains on the past, so the embargo matters for the inner
:func:`purged_kfold`, where training folds sit on both sides of the validation fold.
Tune hyper-parameters and thresholds ONLY with the inner splitters, on rows of the outer
fold's training mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from profit_engine.intraday import config
from profit_engine.intraday.holdout import assert_pre_holdout

DEFAULT_EMBARGO = pd.Timedelta("1D")


@dataclass(frozen=True)
class Fold:
    number: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive
    train: np.ndarray  # boolean mask over the index
    test: np.ndarray  # boolean mask over the index

    @property
    def n_train(self) -> int:
        return int(self.train.sum())

    @property
    def n_test(self) -> int:
        return int(self.test.sum())


def _ns(index: pd.DatetimeIndex) -> np.ndarray:
    """Epoch nanoseconds regardless of the index's stored resolution (pandas 3 keeps ms/us)."""
    return pd.DatetimeIndex(index).as_unit("ns").asi8


def _t1_array(index: pd.DatetimeIndex, t1: pd.Series | None) -> np.ndarray:
    """Label end times as int64 ns; NaT -> max int (always purged); None -> the row's own time."""
    if t1 is None:
        return _ns(index).copy()
    aligned = pd.Series(t1).reindex(index)
    vals = _ns(pd.DatetimeIndex(aligned)).copy()
    vals[aligned.isna().to_numpy()] = np.iinfo("int64").max
    return vals


def purge_and_embargo(
    index: pd.DatetimeIndex,
    t1: pd.Series | None,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    embargo: pd.Timedelta,
) -> np.ndarray:
    """Boolean mask of rows that may train a model evaluated on ``[test_start, test_end)``."""
    t = _ns(index)
    ends = _t1_array(index, t1)
    b, e = test_start.value, test_end.value
    before = (t < b) & (ends < b)
    after = t >= e + embargo.value
    return before | after


def walk_forward_splits(
    index: pd.DatetimeIndex,
    t1: pd.Series | None = None,
    train: str = "expanding",
    train_window: pd.DateOffset | None = None,
    test_months: int = 1,
    eval_start: pd.Timestamp = config.EVAL_START,
    eval_end: pd.Timestamp = config.EVAL_END,
    train_start: pd.Timestamp = config.TRAIN_START,
    embargo: pd.Timedelta = DEFAULT_EMBARGO,
    min_train_rows: int = 1,
) -> list[Fold]:
    """Monthly (``test_months``) test blocks over ``[eval_start, eval_end)``, training before each.

    ``train="expanding"`` trains on ``[train_start, block)``; ``"rolling"`` on
    ``[block - train_window, block)`` (never before ``train_start``). Training rows are purged
    by ``t1``. Folds with fewer than ``min_train_rows`` training rows are skipped.
    """
    if train not in ("expanding", "rolling"):
        raise ValueError("train must be 'expanding' or 'rolling'")
    if train == "rolling" and train_window is None:
        raise ValueError("rolling training needs train_window, e.g. pd.DateOffset(months=12)")
    assert_pre_holdout(pd.Timestamp(eval_end) - pd.Timedelta(1, "ns"), "walk_forward_splits eval_end")
    t = _ns(index)
    folds: list[Fold] = []
    b = pd.Timestamp(eval_start)
    while b < eval_end:
        e = min(b + pd.DateOffset(months=test_months), pd.Timestamp(eval_end))
        lo = train_start if train == "expanding" else max(train_start, b - train_window)
        test = (t >= b.value) & (t < e.value)
        tr = purge_and_embargo(index, t1, b, e, embargo) & (t >= lo.value) & (t < b.value)
        if test.any() and tr.sum() >= min_train_rows:
            folds.append(Fold(len(folds), b, e, tr, test))
        b = e
    return folds


def purged_kfold(
    index: pd.DatetimeIndex,
    t1: pd.Series | None,
    n_splits: int = 5,
    embargo: pd.Timedelta = DEFAULT_EMBARGO,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous K-fold over ``index`` (pass only the outer training rows) with purge + embargo."""
    n = len(index)
    if n_splits < 2 or n < n_splits:
        raise ValueError("need n_splits >= 2 and at least n_splits rows")
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    out = []
    for k in range(n_splits):
        test = np.zeros(n, dtype=bool)
        test[bounds[k] : bounds[k + 1]] = True
        start = index[bounds[k]]
        end = index[bounds[k + 1]] if bounds[k + 1] < n else index[-1] + pd.Timedelta(1, "ns")
        out.append((purge_and_embargo(index, t1, start, end, embargo) & ~test, test))
    return out


def purged_tail_split(
    index: pd.DatetimeIndex,
    t1: pd.Series | None,
    valid_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Train on the first ``1 - valid_frac`` of rows, validate on the rest, purged at the seam."""
    n = len(index)
    cut = int(n * (1 - valid_frac))
    if not 0 < cut < n:
        raise ValueError("valid_frac leaves an empty side")
    valid = np.zeros(n, dtype=bool)
    valid[cut:] = True
    train = purge_and_embargo(index, t1, index[cut], index[-1] + pd.Timedelta(1, "ns"), pd.Timedelta(0))
    return train & ~valid, valid
