"""Audit round 1: the trial registry counts every run, and the promotion gate checks its inputs.

* Re-runs with a new seed, code version or data hash are separate trials (DSR N, variance, PBO).
* Nothing outside the walk-forward OOS window reaches the registry or the gate.
* The gate's stress run and benchmark bars must belong to the base backtest.
* The holdout audit record always lands in the canonical log.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from profit_engine.intraday import config, trials
from profit_engine.intraday.backtest import run_backtest
from profit_engine.intraday.holdout import open_holdout
from profit_engine.intraday.promotion import Candidate, evaluate, holdout_check

# ---------------------------------------------------------------- registry


def _daily(seed: int, start: str = "2020-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, "2026-06-30", freq="D", tz="UTC")
    return pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)


def _log(seed: int | None, data_hash: str = "h", returns: pd.Series | None = None) -> str:
    r = _daily(seed or 0) if returns is None else returns
    sd = float(r.mean() / r.std())
    m = {"sharpe": sd * 365**0.5, "sharpe_daily": sd}
    return trials.log_trial("lgbm", {"lr": 0.05}, "1h", "BTCUSDT", 5.0, m, r, data_hash, seed)


def test_registry_every_seed_is_its_own_trial(isolated):
    for seed in range(5):
        _log(seed)
    assert trials.n_trials() == 5
    assert len(list((config.TRIALS_DIR / "returns" / "lgbm").glob("*.parquet"))) == 5
    assert trials.returns_matrix("BTCUSDT").shape[1] == 5
    assert trials.trial_sharpe_variance() > 0


def test_registry_counts_code_and_data_changes_and_never_overwrites(isolated, monkeypatch):
    monkeypatch.setattr(trials, "git_sha", lambda: "sha_a")
    _log(0, "data_1")
    _log(0, "data_2")  # same params and seed, new data
    monkeypatch.setattr(trials, "git_sha", lambda: "sha_b")
    _log(0, "data_2")  # same params, seed and data, new code
    assert trials.n_trials() == 3
    _log(0, "data_2")  # identical run re-logged (same returns): same trial
    assert trials.n_trials() == 3
    # Same declared spec, different returns (e.g. an untracked script changed): a new trial.
    _log(0, "data_2", returns=_daily(99))
    assert trials.n_trials() == 4
    assert len(list((config.TRIALS_DIR / "returns" / "lgbm").glob("*.parquet"))) == 5
    mat = trials.returns_matrix("BTCUSDT")
    assert mat.shape[1] == 4
    assert any(np.allclose(mat[c].to_numpy(), _daily(99).to_numpy()) for c in mat.columns)


def test_registry_rejects_returns_before_the_oos_window(isolated):
    with pytest.raises(ValueError, match="EVAL_START"):
        _log(0, returns=_daily(0, start="2019-12-01"))
    assert trials.n_trials() == 0


# ---------------------------------------------------------------- promotion inputs


def _daily_bars(start: str, seed: int = 0) -> pd.DataFrame:
    """Daily bars: a crash before EVAL_START, a steady rise after it."""
    idx = pd.date_range(start, "2026-06-30", freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    drift = np.where(idx < config.EVAL_START, -0.01, 0.002)
    px = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.02, len(idx))))
    return pd.DataFrame({"open": px, "high": px, "low": px, "close": px}, index=idx)


def _rows(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = table[table["criterion"].str.startswith(prefix)]
    assert len(rows) == 2, f"no row '{prefix}' per symbol"
    return rows


def _candidate(base_bars: dict, bench_bars: dict, stress_target: float | None = None) -> Candidate:
    base = {s: run_backtest(b, 1.0, 5) for s, b in base_bars.items()}
    st = 1.0 if stress_target is None else stress_target
    stress = {s: run_backtest(b, st, 10) for s, b in base_bars.items()}
    return Candidate("probe", {}, "1d", base, stress, bench_bars)


def test_promotion_rejects_backtest_starting_before_eval_window(isolated):
    full = {s: _daily_bars("2019-06-01", i) for i, s in enumerate(config.SYMBOLS)}
    table = evaluate(_candidate(full, full), n_perm=10)
    assert not _rows(table, "covers 2020-01..2026-06")["pass"].any()
    oos = {s: b[b.index >= config.EVAL_START] for s, b in full.items()}
    table = evaluate(_candidate(oos, oos), n_perm=10)
    assert _rows(table, "covers 2020-01..2026-06")["pass"].all()


def test_promotion_benchmarks_run_on_the_backtest_bars(isolated):
    full = {s: _daily_bars("2019-06-01", i) for i, s in enumerate(config.SYMBOLS)}
    oos = {s: b[b.index >= config.EVAL_START] for s, b in full.items()}
    table = evaluate(_candidate(oos, full), n_perm=10)  # strategy IS buy-and-hold on oos
    assert not _rows(table, "benchmark bars match backtest")["pass"].any()
    bh = _rows(table, "Sharpe vs buy-and-hold")
    assert not bh["pass"].any()
    assert np.allclose(bh["value"].to_numpy(dtype=float), 0.0)
    assert not _rows(table, "Sharpe vs vol-matched B&H")["pass"].any()
    table = evaluate(_candidate(oos, oos), n_perm=10)
    assert _rows(table, "benchmark bars match backtest")["pass"].all()


def test_promotion_stress_run_must_hold_base_positions(isolated):
    oos = {s: _daily_bars("2020-01-01", i) for i, s in enumerate(config.SYMBOLS)}
    table = evaluate(_candidate(oos, oos, stress_target=0.0), n_perm=10)
    assert not _rows(table, "stress run holds base positions")["pass"].any()
    table = evaluate(_candidate(oos, oos), n_perm=10)
    assert _rows(table, "stress run holds base positions")["pass"].all()


# ---------------------------------------------------------------- holdout audit path


def test_holdout_audit_cannot_be_redirected(isolated, tmp_path):
    with pytest.raises(TypeError):
        open_holdout(reason="trying to divert the audit record", audit_log=tmp_path / "x.jsonl")
    with open_holdout(reason="unit test: canonical audit log only"), pytest.raises(TypeError):
        holdout_check({}, audit_log=tmp_path / "x.jsonl")
    assert not (tmp_path / "x.jsonl").exists()
    assert len(config.HOLDOUT_AUDIT_LOG.read_text().splitlines()) == 1
