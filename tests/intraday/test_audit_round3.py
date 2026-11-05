"""Audit round 3: inputs that used to pass silently now raise or FAIL.

* Cross-asset features refuse frames with different bar lengths (a coarser ``other`` bar's
  close would land on a finer primary bar that closes earlier: look-ahead).
* The promotion gate requires the EVALUATED base and stress runs (by returns hash) to be in the
  registry, and always reads the canonical registry.
* The simulator refuses non-finite targets, non-finite or non-positive opens (a reindexed frame)
  and a non-UTC index; ``min_hold_bars`` and the daily entry cap hold on the deferred-fill path.
* The inner purged splitters can be pinned to an outer fold and raise on rows that reach its
  test block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from profit_engine.intraday import config, features, trials
from profit_engine.intraday.backtest import run_backtest
from profit_engine.intraday.promotion import Candidate, evaluate
from profit_engine.intraday.resample import resample
from profit_engine.intraday.splits import purged_kfold, purged_tail_split, walk_forward_splits

# ---------------------------------------------------------------- features: mixed bar lengths


def test_cross_asset_refuses_a_coarser_other_frame(bar_factory):
    btc_5m = bar_factory(n=2000, seed=1)
    eth_1h = resample(bar_factory(n=2000, seed=2), "1h", incomplete="drop")
    for call in (
        lambda: features.make_feature_frame(btc_5m, eth_1h),
        lambda: features.cross_asset(btc_5m, eth_1h),
        lambda: features.align(btc_5m, eth_1h),
        lambda: features.make_feature_frame(resample(btc_5m, "1h", incomplete="drop"), bar_factory(n=2000, seed=2)),
    ):
        with pytest.raises(ValueError, match="bar length"):
            call()


def test_cross_asset_accepts_same_bar_length_with_holes(bar_factory):
    btc = resample(bar_factory(n=3000, seed=1), "1h", incomplete="drop")
    eth = resample(bar_factory(n=3000, seed=2), "1h", incomplete="drop").drop(btc.index[[3, 9]])
    out = features.make_feature_frame(btc, eth)
    assert len(out) == len(btc) - 2


# ---------------------------------------------------------------- promotion: the evaluated run is logged


def _oos_bars(seed: int) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", "2026-06-30", freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, len(idx))))
    return pd.DataFrame({"open": px, "high": px, "low": px, "close": px}, index=idx)


def _noise(seed: int) -> pd.Series:
    idx = pd.date_range("2020-01-01", "2026-06-30", freq="D", tz="UTC")
    return pd.Series(np.random.default_rng(seed).normal(0, 0.01, len(idx)), index=idx)


def _gate_candidate() -> Candidate:
    bars = {s: _oos_bars(i) for i, s in enumerate(config.SYMBOLS)}
    tgt = {s: (b["close"].pct_change() > 0).astype(float) for s, b in bars.items()}
    base = {s: run_backtest(bars[s], tgt[s], config.BASE_COST_BPS) for s in bars}
    stress = {s: run_backtest(bars[s], tgt[s], config.STRESS_COST_BPS) for s in bars}
    return Candidate("fam", {"k": 1}, "1d", base, stress, bars)


def _logged_rows(table: pd.DataFrame, which: str) -> pd.Series:
    rows = table[table["criterion"] == f"{which} run logged (returns hash)"]
    assert len(rows) == 2, f"no '{which} run logged' row per symbol"
    return rows["pass"]


def _log(c: Candidate, sym: str, cost: float, returns: pd.Series) -> None:
    trials.log_trial(c.family, c.params, c.timeframe, sym, cost, {}, returns, "h", 0)


def test_gate_fails_when_only_the_config_name_is_logged(isolated):
    c = _gate_candidate()
    for i, sym in enumerate(config.SYMBOLS):  # same config_id, other returns, both costs
        _log(c, sym, config.BASE_COST_BPS, _noise(10 + i))
        _log(c, sym, config.STRESS_COST_BPS, _noise(20 + i))
    table = evaluate(c, n_perm=10)
    assert not _logged_rows(table, "base").any()
    assert not _logged_rows(table, "stress").any()


def test_gate_passes_the_logged_row_only_for_the_evaluated_runs(isolated):
    c = _gate_candidate()
    for sym in config.SYMBOLS:
        _log(c, sym, config.BASE_COST_BPS, c.base[sym].daily)
    table = evaluate(c, n_perm=10)
    assert _logged_rows(table, "base").all()
    assert not _logged_rows(table, "stress").any()  # stress run not logged yet
    for sym in config.SYMBOLS:
        _log(c, sym, config.STRESS_COST_BPS, c.stress[sym].daily)
        assert trials.returns_matrix(sym).shape[1] == 1  # the candidate is a PBO column
    table = evaluate(c, n_perm=10)
    assert _logged_rows(table, "base").all() and _logged_rows(table, "stress").all()


def test_gate_always_reads_the_canonical_registry(isolated, tmp_path):
    c = _gate_candidate()
    hand_picked = tmp_path / "hand_picked"
    for sym in config.SYMBOLS:
        _log(c, sym, config.BASE_COST_BPS, c.base[sym].daily)
    with pytest.raises(TypeError, match="trials_dir"):
        evaluate(c, trials_dir=hand_picked, n_perm=10)


# ---------------------------------------------------------------- simulator inputs


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_backtest_refuses_an_infinite_target(bars, bad):
    tgt = pd.Series(1.0, index=bars.index)
    tgt.iloc[100] = bad
    with pytest.raises(ValueError, match="finite"):
        run_backtest(bars, tgt, 5)


def test_backtest_refuses_a_frame_reindexed_onto_a_full_grid(bars):
    holed = bars.drop(bars.index[500:560])
    assert run_backtest(holed, 1.0, 5).params["n_gaps"] == 1  # the honest frame flags the gap
    grid = holed.reindex(pd.date_range(holed.index[0], holed.index[-1], freq="5min", tz="UTC"))
    with pytest.raises(ValueError, match="open"):
        run_backtest(grid, 1.0, 5)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_backtest_refuses_non_positive_opens(bars, bad):
    d = bars.copy()
    d.iloc[50, d.columns.get_loc("open")] = bad
    with pytest.raises(ValueError, match="open"):
        run_backtest(d, 1.0, 5)


def test_backtest_requires_a_utc_index(bars):
    with pytest.raises(ValueError, match="UTC"):
        run_backtest(bars.tz_convert("Asia/Tokyo"), 1.0, 5)


def _halted(df: pd.DataFrame, j: int) -> pd.DataFrame:
    d = df.copy()
    d.iloc[j, d.columns.get_loc("volume")] = 0.0
    return d


def test_min_hold_survives_a_deferred_entry(bar_factory):
    d = _halted(bar_factory(n=30), 1)
    tgt = pd.Series(0.0, index=d.index)
    tgt.iloc[0:3] = 1.0  # entry due at bar 1 (halted), exit due at bar 4
    r = run_backtest(d, tgt, 0, min_hold_bars=3)
    assert r.trades["entry_time"].iloc[0] == d.index[2]  # deferred off the halted bar
    assert r.trades["holding_bars"].min() >= 3
    assert r.params["fills_deferred"] >= 1


def test_entry_cap_counts_the_day_the_entry_fills(bar_factory):
    d = _halted(bar_factory(n=30, start="2021-03-01 23:00"), 12)  # bar 12 = 00:00, halted
    tgt = pd.Series(0.0, index=d.index)
    tgt.iloc[11] = 1.0  # entry due at 00:00 (halted, never fills)
    tgt.iloc[13:20] = 1.0  # entry due at 00:10
    r = run_backtest(d, tgt, 0, max_round_trips_per_day=1)
    assert r.bars["position"].iloc[12] == 0.0
    assert r.bars["position"].iloc[14] == 1.0  # the cap was not used up by the unfilled entry
    assert len(r.trades) == 1


# ---------------------------------------------------------------- inner splitters pinned to an outer fold


def _outer() -> tuple[pd.DatetimeIndex, pd.Series, object]:
    idx = pd.date_range("2019-10-01", "2020-02-29 23:00", freq="h", tz="UTC")
    t1 = pd.Series(idx + pd.Timedelta(hours=4), index=idx)
    fold = walk_forward_splits(idx, t1, eval_end=pd.Timestamp("2020-03-01", tz="UTC"))[0]
    return idx, t1, fold


@pytest.mark.parametrize("splitter", [purged_kfold, purged_tail_split])
def test_inner_splitter_pinned_to_a_fold(splitter):
    idx, t1, fold = _outer()
    tr = idx[fold.train]
    splitter(tr, t1[tr], within=fold)  # the outer training rows: fine
    splitter(tr, t1[tr], within=fold.test_start)
    with pytest.raises(ValueError, match="test block"):
        splitter(idx, t1, within=fold)  # reaches into the OOS month
    late_t1 = t1[tr].copy()
    late_t1.iloc[-1] = fold.test_start  # a label that ends inside the test block
    with pytest.raises(ValueError, match="test block"):
        splitter(tr, late_t1, within=fold)
