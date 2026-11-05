"""Audit round 1: holdout guard coverage, inner-CV purging, t1 required, seed refresh,
and simulator honesty (gap flags, weight drift, target alignment)."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from profit_engine.intraday import binance, data, labels, metrics
from profit_engine.intraday.backtest import run_backtest
from profit_engine.intraday.holdout import HoldoutViolation, open_holdout
from profit_engine.intraday.overfitting import dsr_from_daily, pbo_cscv, permutation_test
from profit_engine.intraday.splits import purged_kfold, purged_tail_split, walk_forward_splits

# ---------------------------------------------------------------- (f) every scoring entry point


def _t1(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.index, index=df.index).shift(-3)


def _pbo_input(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0, 0.01, (len(df), 3)), index=df.index)


GUARDED = {
    "labels.forward_return": lambda df, res: labels.forward_return(df, 3),
    "labels.triple_barrier": lambda df, res: labels.triple_barrier(df, 3, vol_window=12),
    "splits.walk_forward_splits(index)": lambda df, res: walk_forward_splits(df.index, _t1(df)),
    "splits.purged_kfold": lambda df, res: purged_kfold(df.index, _t1(df), 3),
    "splits.purged_tail_split": lambda df, res: purged_tail_split(df.index, _t1(df)),
    "overfitting.permutation_test": lambda df, res: permutation_test(res, n_perm=5),
    "overfitting.dsr_from_daily": lambda df, res: dsr_from_daily(res.daily, 10, 0.01),
    "overfitting.pbo_cscv(DataFrame)": lambda df, res: pbo_cscv(_pbo_input(df)),
    "metrics.sharpe": lambda df, res: metrics.sharpe(res.daily),
    "metrics.sortino": lambda df, res: metrics.sortino(res.daily),
    "metrics.cagr": lambda df, res: metrics.cagr(res.daily),
    "metrics.max_drawdown": lambda df, res: metrics.max_drawdown(res.daily),
    "metrics.monthly_returns": lambda df, res: metrics.monthly_returns(res.daily),
    "metrics.yearly_returns": lambda df, res: metrics.yearly_returns(res.daily),
    "metrics.best_month_share": lambda df, res: metrics.best_month_share(res.daily),
    "metrics.trade_stats": lambda df, res: metrics.trade_stats(res.trades),
}


@pytest.mark.parametrize("name", sorted(GUARDED))
def test_f_every_scoring_entry_point_is_guarded(isolated, bar_factory, name):
    df = bar_factory(n=600, start="2026-06-30 23:00")
    call = GUARDED[name]
    with open_holdout(reason=f"guard coverage test for {name}"):
        res = run_backtest(df, 1.0)
        call(df, res)  # allowed inside the audited block
    with pytest.raises(HoldoutViolation):
        call(df, res)  # a result built inside, analysed outside, still raises


# ---------------------------------------------------------------- (e) inner CV purge after the fold


def _hourly(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")


def test_e_purged_kfold_purges_after_fold_by_validation_label_end():
    idx = _hourly("2021-01-01", "2021-03-01")
    t1 = pd.Series(idx, index=idx).shift(-49)  # label horizon 2 days > 1 day embargo
    embargo = pd.Timedelta("1D")
    checked = 0
    for train, test in purged_kfold(idx, t1, n_splits=4, embargo=embargo):
        if test[-1]:
            continue  # last fold: nothing after it
        after = train & (idx > idx[test][-1])
        assert after.any()
        assert idx[after].min() >= t1[test].max() + embargo
        checked += 1
    assert checked == 3


def test_e_purged_kfold_unknown_validation_label_end_purges_everything_after():
    idx = _hourly("2021-01-01", "2021-03-01")
    t1 = pd.Series(idx, index=idx).shift(-6)
    t1.iloc[10] = pd.NaT  # inside the first validation fold
    train, test = purged_kfold(idx, t1, n_splits=4)[0]
    assert test[10] and not train.any()


def test_e_t1_is_required():
    idx = _hourly("2019-12-01", "2020-02-01")
    end = pd.Timestamp("2020-02-01", tz="UTC")
    with pytest.raises(TypeError):
        walk_forward_splits(idx, eval_end=end)  # no t1 at all
    for call in (
        lambda: walk_forward_splits(idx, None, eval_end=end),
        lambda: purged_kfold(idx, None),
        lambda: purged_tail_split(idx, None),
    ):
        with pytest.raises(ValueError, match="allow_unpurged"):
            call()
    assert walk_forward_splits(idx, None, eval_end=end, allow_unpurged=True)


# ---------------------------------------------------------------- (d) seed's last bar is refreshed

MS = 300_000


def _raw_rows(open_ms: list[int], volume: str) -> list[list[str]]:
    return [[str(o), "1", "2", "0.5", "1.5", volume, str(o + MS - 1), "15", "7", "4", "6", "0"] for o in open_ms]


def test_d_seed_last_bar_is_refetched(isolated, tmp_path, monkeypatch):
    now = pd.Timestamp("2026-11-05 14:20", tz="UTC")
    t0 = pd.Timestamp("2026-11-05 13:00", tz="UTC").value // 10**6
    seed_opens = [t0 + i * MS for i in range(14)]  # 13:00 .. 14:05
    seed = _raw_rows(seed_opens[:-1], "10") + _raw_rows(seed_opens[-1:], "3")  # last one was mid-bar
    cols = [c for c in binance.RAW_COLUMNS if c != "ignore"]
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    with gzip.open(seed_dir / "binance_global_BTCUSDT_5m.csv.gz", "wt") as fh:
        pd.DataFrame([r[:-1] for r in seed], columns=cols).to_csv(fh, index=False)

    def fake_rest(symbol, interval, start_ms, session, end_ms=None, timeout=30):
        opens = [o for o in (t0 + i * MS for i in range(17)) if o >= start_ms]  # through 14:20
        return pd.DataFrame(_raw_rows(opens, "10"), columns=list(binance.RAW_COLUMNS))

    monkeypatch.setattr(binance, "fetch_rest", fake_rest)
    data.sync_cache("BTCUSDT", now=now, session=object(), seed_dir=seed_dir)
    df = pd.read_parquet(data.cache_path("BTCUSDT"))
    assert (df["volume"] == 10.0).all()
    assert df.index[-1] == pd.Timestamp("2026-11-05 14:15", tz="UTC")  # 14:20 still in progress
    assert df.loc[pd.Timestamp("2026-11-05 14:05", tz="UTC"), "volume"] == 10.0


# ---------------------------------------------------------------- simulator


def _flat_bars(opens: list[float], start: str = "2022-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(opens), freq="5min", tz="UTC")
    o = np.asarray(opens, dtype=float)
    return pd.DataFrame({"open": o, "high": o, "low": o, "close": o}, index=idx)


def test_gap_crossing_is_flagged(bar_factory):
    df = _flat_bars([100.0] * 10)
    df = df.drop(df.index[4:7])  # bar 3 is followed by a 20-minute jump
    target = pd.Series(0.0, index=df.index)
    target.iloc[2] = 1.0  # held over bar 3, across the hole
    res = run_backtest(df, target, cost_bps_per_side=0)
    assert res.bars["span_bars"].tolist() == [1, 1, 1, 4, 1, 1, 1]
    trade = res.trades.iloc[0]
    assert bool(trade["crosses_gap"]) and trade["holding_bars"] == 1
    assert trade["holding_time"] == pd.Timedelta("20min")
    assert res.params["n_gaps"] == 1 and res.params["bars_held_across_gap"] == 1
    clean = run_backtest(_flat_bars([100.0] * 10), 1.0)
    assert not clean.trades["crosses_gap"].any() and clean.params["n_gaps"] == 0


def test_fractional_position_drifts_like_a_held_position():
    df = _flat_bars([100, 100, 200, 200, 400, 400])
    res = run_backtest(df, 0.5, cost_bps_per_side=0)
    assert res.bars["equity"].iloc[-1] == pytest.approx(0.5 + 0.5 * 4)  # true 50% hold, not 2.25
    c = 5e-4
    res = run_backtest(df, 0.5, cost_bps_per_side=5)
    w_exit = 0.5 * 4 / (0.5 + 0.5 * 4)
    assert res.bars["equity"].iloc[-1] == pytest.approx((1 - c * 0.5) * 2.5 * (1 - c * w_exit), rel=1e-12)


def test_rebalance_is_charged_against_the_drifted_weight():
    df = _flat_bars([100, 100, 200, 200, 200, 200])
    target = pd.Series([0.5, 0.5, 0.6, 0.6, 0.6, 0.0], index=df.index)
    res = run_backtest(df, target, cost_bps_per_side=5)
    drifted = 0.5 * 2 / (0.5 + 0.5 * 2)
    assert res.bars["cost"].iloc[3] == pytest.approx(5e-4 * abs(0.6 - drifted), rel=1e-12)
    assert res.bars["cost"].iloc[2] == 0.0  # no fill while the target is unchanged


def test_misaligned_target_raises_instead_of_going_flat(bar_factory):
    df = bar_factory(n=100)
    good = pd.Series(1.0, index=df.index)
    for bad in (
        pd.Series(1.0, index=df.index.tz_localize(None)),
        pd.Series(1.0, index=df.index + pd.Timedelta("1s")),
        pd.Series(1.0, index=range(len(df))),
        np.ones(len(df)),
    ):
        with pytest.raises((ValueError, TypeError)):
            run_backtest(df, bad)
    wider = bar_factory(n=200)
    assert run_backtest(df, pd.Series(1.0, index=wider.index)).bars["position"].sum() > 0  # superset ok
    assert run_backtest(df, good.iloc[::2]).bars["position"].sum() > 0  # sparse on-grid ok
