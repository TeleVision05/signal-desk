"""Audit round 2: the registry only holds full-window walk-forward runs, scored by the harness.

* Nothing past EVAL_END reaches the registry or the gate, even inside ``open_holdout``.
* Every counted run contributes to the DSR variance and the best-baseline bar, from its saved
  returns rather than from a caller-supplied metrics dict.
* A run that does not cover the walk-forward window is rejected (it would shrink the PBO matrix).
* Runs whose code differs only in untracked files are still separate trials.
* The holdout guard fails closed on anything it cannot read as a time.
* The simulator refuses an unordered index and never fills on a zero-volume (halted) bar.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from profit_engine.intraday import config, labels, promotion, trials
from profit_engine.intraday.backtest import run_backtest
from profit_engine.intraday.holdout import HoldoutViolation, assert_pre_holdout, open_holdout
from profit_engine.intraday.metrics import sharpe


def _full(seed: int, end: str = "2026-06-30", mu: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", end, freq="D", tz="UTC")
    return pd.Series(rng.normal(mu, 0.01, len(idx)), index=idx)


def _log(family: str, params: dict, r: pd.Series, metrics: dict | None = None, **kw) -> str:
    cost = kw.pop("cost", 5.0)
    return trials.log_trial(family, params, "1h", "BTCUSDT", cost, metrics or {}, r, "h", kw.pop("seed", 0), **kw)


# ---------------------------------------------------------------- window: holdout never enters the registry


def test_log_trial_rejects_returns_past_eval_end_even_inside_open_holdout(isolated):
    leaky = _full(0, end="2026-10-31")
    reason = "unit test: holdout returns must not reach the registry"
    with open_holdout(reason=reason), pytest.raises(ValueError, match="EVAL_END"):
        _log("leaky", {"a": 1}, leaky, {"sharpe": 9.9}, kind="baseline")
    assert not (config.TRIALS_DIR / "leaky.jsonl").exists()
    assert not (config.TRIALS_DIR / "returns" / "leaky").exists()
    assert trials.best_baseline_sharpe("BTCUSDT") is None


def _bars_4h(end: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", end, freq="4h", tz="UTC", inclusive="left")
    px = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx))))
    return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0}, index=idx)


def test_promotion_window_is_bounded_whether_or_not_the_holdout_is_open(isolated):
    bars = _bars_4h("2026-10-01")
    with open_holdout(reason="unit test: promotion must not score holdout months"):
        base, stress = run_backtest(bars, 1.0, 5), run_backtest(bars, 1.0, 10)
        cand = promotion.Candidate("p", {}, "4h", {"BTCUSDT": base}, {"BTCUSDT": stress}, {"BTCUSDT": bars})
        with pytest.raises(HoldoutViolation, match="open_holdout"):
            promotion.evaluate(cand, n_perm=10)
        rows = promotion._consistency_rows("BTCUSDT", base, stress, bars)
    covers = [r for r in rows if r["criterion"].startswith("covers")]
    assert len(covers) == 1 and not covers[0]["pass"]
    ok = _bars_4h("2026-07-01")
    rows = promotion._consistency_rows("BTCUSDT", run_backtest(ok, 1.0, 5), run_backtest(ok, 1.0, 10), ok)
    assert [r["pass"] for r in rows if r["criterion"].startswith("covers")] == [True]


# ---------------------------------------------------------------- the harness scores every run itself


def test_registry_sharpes_come_from_the_saved_returns_not_the_caller(isolated):
    series = [_full(s) for s in range(6)]
    for i, r in enumerate(series[:5]):  # no sharpe_daily in the caller's metrics
        _log("famA", {"i": i}, r, {"sharpe": 1.0} if i % 2 else {})
    _log("rules", {"fast": 10}, series[5], {"sharpe": 9.9, "sharpe_daily": 9.9}, kind="baseline")
    expected = pd.Series([sharpe(r, annualize=False) for r in series]).var(ddof=1)
    assert trials.trial_sharpe_variance() == pytest.approx(expected, rel=1e-9)
    assert trials.best_baseline_sharpe("BTCUSDT") == pytest.approx(sharpe(series[5]), rel=1e-9)
    rec = json.loads((config.TRIALS_DIR / "rules.jsonl").read_text().splitlines()[-1])
    assert rec["metrics"]["sharpe"] == pytest.approx(sharpe(series[5]), rel=1e-9)
    assert rec["metrics"]["sharpe_daily"] == pytest.approx(sharpe(series[5], annualize=False), rel=1e-9)


def test_registry_variance_without_any_caller_metrics(isolated):
    for i in range(3):
        _log("bare", {"i": i}, _full(i), {})
    assert trials.trial_sharpe_variance() > 0


# ---------------------------------------------------------------- coverage: no run may shrink the PBO matrix


def test_log_trial_rejects_series_that_do_not_cover_the_window(isolated):
    for i in range(3):
        _log("full", {"i": i}, _full(i))
    before = trials.returns_matrix("BTCUSDT").shape
    with pytest.raises(ValueError, match="cover"):
        _log("debug", {"i": 0}, _full(9).iloc[:31])
    with pytest.raises(ValueError, match="cover"):
        _log("debug", {"i": 1}, _full(9).loc["2021-01-01":])
    with pytest.raises(ValueError, match="cover"):
        _log("debug", {"i": 2}, _full(9).loc[:"2025-12-31"])
    assert trials.returns_matrix("BTCUSDT").shape == before
    assert trials.n_trials() == 3


# ---------------------------------------------------------------- identity: untracked code changes are new trials


def test_runs_differing_only_in_their_returns_are_separate_trials(isolated):
    for variant in range(5):  # same family/params/seed/sha/data, different (untracked) code
        _log("famC", {"thr": 0.5}, _full(variant), seed=7)
    assert trials.n_trials() == 5
    assert trials.returns_matrix("BTCUSDT").shape[1] == 5
    _log("famC", {"thr": 0.5}, _full(0), seed=7)  # identical re-run: no new trial
    assert trials.n_trials() == 5
    for variant in range(5):  # stress-cost re-scoring of the same runs: no new trial
        _log("famC", {"thr": 0.5}, _full(variant) - 1e-4, seed=7, cost=10.0)
    assert trials.n_trials() == 5
    assert trials.returns_matrix("BTCUSDT").shape[1] == 5


def test_benchmark_runs_are_not_exempt_from_the_trial_count(isolated):
    for i in range(3):
        _log("famD", {"i": i}, _full(i), kind="benchmark")
    assert trials.n_trials() == 3
    assert trials.returns_matrix("BTCUSDT").shape[1] == 3


# ---------------------------------------------------------------- holdout guard fails closed


def test_holdout_guard_fails_closed_on_non_datetime_input(isolated):
    h = pd.date_range("2026-06-01", "2026-08-01", freq="1h", tz="UTC")
    df = pd.DataFrame({"open": np.linspace(1, 2, len(h)), "close": 1.0, "high": 2.0, "low": 0.5}, index=h)
    with pytest.raises(HoldoutViolation):
        labels.forward_return(df, 4)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        labels.forward_return(df.reset_index(drop=True), 4)
    post = pd.Series(np.random.default_rng(0).normal(0, 0.01, 60), index=pd.date_range("2026-07-01", periods=60))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        sharpe(post.reset_index(drop=True))
    for scalar in (5, 1.5, True):
        with pytest.raises(TypeError):
            assert_pre_holdout(scalar, "scalar")
    assert_pre_holdout(pd.Timestamp("2026-06-30", tz="UTC"), "timestamp")  # datetimes still fine
    assert_pre_holdout(pd.Series([], dtype=float), "empty")  # nothing to touch


# ---------------------------------------------------------------- simulator input order and halted bars


def test_run_backtest_rejects_an_unordered_or_duplicated_index(bar_factory):
    df = bar_factory(n=500)
    bad = {
        "shuffled": df.sample(frac=1, random_state=0),
        "reversed": df.iloc[::-1],
        "duplicated": pd.concat([df.iloc[:300], df.iloc[200:]]),
        "tz-naive": df.tz_localize(None),
    }
    for name, frame in bad.items():
        with pytest.raises(ValueError, match=r"(?i)index"):
            run_backtest(frame, 1.0, 0)
            pytest.fail(f"{name} index accepted")


def _halt(df: pd.DataFrame, at: list[int], jump: float) -> pd.DataFrame:
    """Zero-volume bars at ``at`` with the stale previous close as open; next bar gaps by ``jump``."""
    d = df.copy()
    stale = d["close"].iloc[at[0] - 1]
    for j in at:
        d.iloc[j, d.columns.get_loc("volume")] = 0.0
        d.iloc[j, d.columns.get_loc("open")] = stale
        d.iloc[j, d.columns.get_loc("close")] = stale
    d.iloc[at[-1] + 1, d.columns.get_loc("open")] = stale * (1 + jump)
    return d


def test_entry_decided_before_a_halt_fills_at_the_first_traded_open(bar_factory):
    d = _halt(bar_factory(n=30), [10, 11], 0.03)
    tgt = pd.Series(0.0, index=d.index)
    tgt.iloc[9:25] = 1.0  # decided at bar 9 close: the naive fill is bar 10, a halted bar
    r = run_backtest(d, tgt, 0)
    t = r.trades.iloc[0]
    assert t.entry_time == d.index[12] and t.entry_price == d["open"].iloc[12]
    assert (r.bars["position"].iloc[10:12] == 0).all()
    assert t.exit_time == d.index[26]  # untouched: flat decided at bar 25 close
    assert t.gross_ret == pytest.approx(d["open"].iloc[26] / d["open"].iloc[12] - 1)  # the +3% gap is not earned
    assert r.params["n_stale_bars"] == 2 and r.params["fills_deferred"] == 2


def test_exit_across_a_halt_is_deferred_and_flagged(bar_factory):
    d = _halt(bar_factory(n=30), [21], -0.03)
    tgt = pd.Series(0.0, index=d.index)
    tgt.iloc[5:20] = 1.0  # flat decided at bar 20 close: the naive exit is bar 21, a halted bar
    r = run_backtest(d, tgt, 0)
    t = r.trades.iloc[0]
    assert t.exit_time == d.index[22] and t.exit_price == d["open"].iloc[22]
    assert bool(t.crosses_stale)
    assert r.params["bars_held_across_stale"] == 1 and r.params["fills_deferred"] == 1
    clean = run_backtest(bar_factory(n=30), tgt, 0)
    assert not clean.trades["crosses_stale"].any() and clean.params["n_stale_bars"] == 0
