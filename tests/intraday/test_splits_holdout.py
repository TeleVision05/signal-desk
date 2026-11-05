"""(e) purge + embargo; (f) holdout guard raises and open_holdout writes an audit record."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from profit_engine.intraday import config, data
from profit_engine.intraday.backtest import run_backtest
from profit_engine.intraday.holdout import HoldoutViolation, assert_pre_holdout, open_holdout
from profit_engine.intraday.labels import forward_return
from profit_engine.intraday.metrics import summarize
from profit_engine.intraday.promotion import holdout_check
from profit_engine.intraday.splits import purged_kfold, purged_tail_split, walk_forward_splits

# ---------------------------------------------------------------- (e) purge / embargo


def _hourly(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")


def test_e_walk_forward_purges_labels_reaching_into_test():
    idx = _hourly("2019-10-01", "2020-04-01")
    h = 24  # labels end 25 bars later
    t1 = pd.Series(idx, index=idx).shift(-(h + 1))
    folds = walk_forward_splits(
        idx, t1, eval_start=pd.Timestamp("2020-01-01", tz="UTC"), eval_end=pd.Timestamp("2020-04-01", tz="UTC")
    )
    assert [f.test_start.month for f in folds] == [1, 2, 3]
    for f in folds:
        train_t1 = t1[f.train]
        assert (train_t1 < f.test_start).all()
        assert not (f.train & f.test).any()
        # exactly the 25 rows before the block are purged (label reaches the block), no more
        before = (idx < f.test_start) & (idx >= config.TRAIN_START)
        assert before.sum() - f.train.sum() == h + 1
        assert (idx[f.test] >= f.test_start).all() and (idx[f.test] < f.test_end).all()


def test_e_unknown_label_end_is_purged():
    idx = _hourly("2019-12-01", "2020-02-01")
    t1 = pd.Series(idx, index=idx).shift(-2)
    t1.iloc[5] = pd.NaT  # unknown label end, far from the test block
    folds = walk_forward_splits(idx, t1, eval_end=pd.Timestamp("2020-02-01", tz="UTC"))
    assert not folds[0].train[5]
    assert folds[0].train[6]


def test_e_rolling_window():
    idx = _hourly("2019-01-01", "2020-03-01")
    folds = walk_forward_splits(
        idx,
        None,
        train="rolling",
        train_window=pd.DateOffset(months=2),
        eval_end=pd.Timestamp("2020-03-01", tz="UTC"),
        allow_unpurged=True,
    )
    first_train = idx[folds[0].train]
    assert first_train[0] == pd.Timestamp("2019-11-01", tz="UTC")


def test_e_purged_kfold_purge_and_embargo():
    idx = _hourly("2021-01-01", "2021-03-01")
    t1 = pd.Series(idx, index=idx).shift(-6)
    embargo = pd.Timedelta("12h")
    splits = purged_kfold(idx, t1, n_splits=4, embargo=embargo)
    for train, test in splits:
        b, e = idx[test][0], idx[test][-1] + pd.Timedelta("1h")
        tr_idx, tr_t1 = idx[train], t1[train]
        before = tr_idx < b
        assert (tr_t1[before] < b).all()  # purge
        after = tr_idx >= e
        assert (tr_idx[after] >= e + embargo).all()  # embargo
        if test[-1] == 0:  # a block with rows after it: embargo removed exactly 12 rows
            assert ((idx >= e) & (idx < e + embargo)).sum() == 12
            assert not train[(idx >= e) & (idx < e + embargo)].any()


def test_e_tail_split():
    idx = _hourly("2021-01-01", "2021-01-11")
    t1 = pd.Series(idx, index=idx).shift(-3)
    train, valid = purged_tail_split(idx, t1, 0.25)
    assert (t1[train] < idx[valid][0]).all() and train.sum() == 180 - 3


def test_e_forward_return_label_times(bars):
    lab = forward_return(bars, 3)
    assert lab["t1"].iloc[0] == bars.index[4]
    assert lab["ret"].iloc[0] == pytest.approx(np.log(bars["open"].iloc[4] / bars["open"].iloc[1]))
    assert lab["ret"].iloc[-4:].isna().all() and lab["t1"].iloc[-4:].isna().all()


# ---------------------------------------------------------------- (f) holdout


def _post_holdout_bars(bar_factory) -> pd.DataFrame:
    return bar_factory(n=600, start="2026-06-30 23:00")


def test_f_guard_raises_everywhere(isolated, bar_factory):
    df = _post_holdout_bars(bar_factory)
    with pytest.raises(HoldoutViolation):
        data.load_klines("BTCUSDT", end="2026-07-02")
    with pytest.raises(HoldoutViolation):
        t1 = pd.Series(df.index, index=df.index).shift(-4)  # not via labels: that raises first
        walk_forward_splits(df.index, t1, eval_end=pd.Timestamp("2026-08-01", tz="UTC"))
    with pytest.raises(HoldoutViolation):
        run_backtest(df, 1.0)
    with pytest.raises(HoldoutViolation):
        holdout_check({})
    assert_pre_holdout(config.HOLDOUT_START - pd.Timedelta(1, "ns"), "edge")  # last legal instant
    with pytest.raises(HoldoutViolation):
        assert_pre_holdout(config.HOLDOUT_START, "edge")
    assert not config.HOLDOUT_AUDIT_LOG.exists()


def test_f_open_holdout_logs_and_unseals_only_inside(isolated, bar_factory):
    df = _post_holdout_bars(bar_factory)
    with pytest.raises(ValueError):
        open_holdout(reason="peek")  # too short to be a reason
    with open_holdout(reason="unit test of the audit trail"):
        res = run_backtest(df, 1.0)
        assert summarize(res)["n_trades"] == 1
    with pytest.raises(HoldoutViolation):
        run_backtest(df, 1.0)
    lines = config.HOLDOUT_AUDIT_LOG.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["reason"] == "unit test of the audit trail"
    assert "test_splits_holdout.py" in rec["caller"]
    assert "test_f_open_holdout_logs_and_unseals_only_inside" in rec["caller"]
    assert {"ts", "git_sha", "user"} <= rec.keys()


def test_f_holdout_check_runs_inside_and_records_result(isolated, bar_factory):
    df = bar_factory(n=288 * 3, start="2026-07-01")
    with open_holdout(reason="unit test of holdout_check"):
        res = run_backtest(df, 1.0)
        table = holdout_check({"BTCUSDT": res})
    assert set(table["symbol"]) == {"BTCUSDT", "ETHUSDT"}
    assert not table.loc[table["symbol"] == "ETHUSDT", "pass"].any()
    events = [json.loads(x) for x in config.HOLDOUT_AUDIT_LOG.read_text().splitlines()]
    assert [e.get("event") for e in events] == [None, "holdout_check"]
