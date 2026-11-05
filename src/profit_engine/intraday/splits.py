"""Walk-forward and purged cross-validation splits.

Samples are rows of a bar-indexed frame; each has a label end time ``t1`` (from
``labels.py``). A training row is usable for a test block ``[b, e)`` only if

* it is not in the block,
* BEFORE the block, its label does not reach the block: ``t1 < b`` (rows with unknown
  ``t1`` are purged),
* AFTER the block, it starts at or after the block's label end plus the embargo:
  ``t >= max(e, max t1 of the block's rows) + embargo``. A block row with unknown ``t1``
  makes that end unbounded, so nothing after the block trains.

The outer walk-forward only trains on the past, so the after-side rule matters for the inner
:func:`purged_kfold`, where training folds sit on both sides of the validation fold.
Tune hyper-parameters and thresholds ONLY with the inner splitters, on rows of the outer
fold's training mask (pass only labelled rows: an unlabelled row has ``t1`` NaT).

``t1`` is REQUIRED everywhere. Passing ``None`` raises unless ``allow_unpurged=True``, which
treats every row's label as ending at its own bar (only true for labels that need no
future bar, which is almost never the case).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from profit_engine.intraday import config
from profit_engine.intraday.holdout import assert_pre_holdout

DEFAULT_EMBARGO = pd.Timedelta("1D")
_INT_MAX = np.iinfo("int64").max


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


def _t1_array(index: pd.DatetimeIndex, t1: pd.Series | None, allow_unpurged: bool = False) -> np.ndarray:
    """Label end times as int64 ns; NaT -> max int (always purged); None only if allowed."""
    if t1 is None:
        if not allow_unpurged:
            raise ValueError(
                "t1 is required: pass the label end times (labels.forward_return(...)['t1']). "
                "Pass t1=None with allow_unpurged=True only for labels known at their own bar."
            )
        return _ns(index).copy()
    aligned = pd.Series(t1).reindex(index)
    vals = _ns(pd.DatetimeIndex(aligned)).copy()
    vals[aligned.isna().to_numpy()] = _INT_MAX
    return vals


def purge_and_embargo(
    index: pd.DatetimeIndex,
    t1: pd.Series | None,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    embargo: pd.Timedelta,
    allow_unpurged: bool = False,
) -> np.ndarray:
    """Boolean mask of rows that may train a model evaluated on ``[test_start, test_end)``."""
    t = _ns(index)
    ends = _t1_array(index, t1, allow_unpurged)
    b, e = test_start.value, test_end.value
    before = (t < b) & (ends < b)
    in_block = (t >= b) & (t < e)
    label_end = max(e, int(ends[in_block].max())) if in_block.any() else e
    if label_end >= _INT_MAX - embargo.value:  # unknown label end inside the block
        return before
    return before | (t >= label_end + embargo.value)


def walk_forward_splits(
    index: pd.DatetimeIndex,
    t1: pd.Series | None,
    train: str = "expanding",
    train_window: pd.DateOffset | None = None,
    test_months: int = 1,
    eval_start: pd.Timestamp = config.EVAL_START,
    eval_end: pd.Timestamp = config.EVAL_END,
    train_start: pd.Timestamp = config.TRAIN_START,
    embargo: pd.Timedelta = DEFAULT_EMBARGO,
    min_train_rows: int = 1,
    allow_unpurged: bool = False,
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
    assert_pre_holdout(index, "walk_forward_splits index")
    _t1_array(index, t1, allow_unpurged)  # fail fast on a missing t1
    t = _ns(index)
    folds: list[Fold] = []
    b = pd.Timestamp(eval_start)
    while b < eval_end:
        e = min(b + pd.DateOffset(months=test_months), pd.Timestamp(eval_end))
        lo = train_start if train == "expanding" else max(train_start, b - train_window)
        test = (t >= b.value) & (t < e.value)
        tr = purge_and_embargo(index, t1, b, e, embargo, allow_unpurged) & (t >= lo.value) & (t < b.value)
        if test.any() and tr.sum() >= min_train_rows:
            folds.append(Fold(len(folds), b, e, tr, test))
        b = e
    return folds


def purged_kfold(
    index: pd.DatetimeIndex,
    t1: pd.Series | None,
    n_splits: int = 5,
    embargo: pd.Timedelta = DEFAULT_EMBARGO,
    allow_unpurged: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous K-fold over ``index`` (pass only the outer training rows) with purge + embargo."""
    assert_pre_holdout(index, "purged_kfold index")
    _t1_array(index, t1, allow_unpurged)
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
        out.append((purge_and_embargo(index, t1, start, end, embargo, allow_unpurged) & ~test, test))
    return out


def purged_tail_split(
    index: pd.DatetimeIndex,
    t1: pd.Series | None,
    valid_frac: float = 0.2,
    allow_unpurged: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Train on the first ``1 - valid_frac`` of rows, validate on the rest, purged at the seam."""
    assert_pre_holdout(index, "purged_tail_split index")
    _t1_array(index, t1, allow_unpurged)
    n = len(index)
    cut = int(n * (1 - valid_frac))
    if not 0 < cut < n:
        raise ValueError("valid_frac leaves an empty side")
    valid = np.zeros(n, dtype=bool)
    valid[cut:] = True
    end = index[-1] + pd.Timedelta(1, "ns")
    train = purge_and_embargo(index, t1, index[cut], end, pd.Timedelta(0), allow_unpurged)
    return train & ~valid, valid
