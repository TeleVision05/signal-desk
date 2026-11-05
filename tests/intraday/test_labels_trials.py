"""Triple-barrier labels, the trial registry, and the promotion gate wiring."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from profit_engine.intraday import config, trials
from profit_engine.intraday.backtest import run_backtest
from profit_engine.intraday.holdout import HoldoutViolation
from profit_engine.intraday.labels import triple_barrier
from profit_engine.intraday.metrics import summarize
from profit_engine.intraday.promotion import Candidate, evaluate, verdict


def _path_bars(opens, highs, lows) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=len(opens), freq="5min", tz="UTC")
    o = np.asarray(opens, dtype=float)
    return pd.DataFrame({"open": o, "high": highs, "low": lows, "close": o}, index=idx)


def test_triple_barrier_first_touch_and_end_time():
    # sigma is forced via vol_window=2 on a path with a known 1-bar log-return std.
    o = [100, 101, 100, 101, 100, 100, 100, 100, 100, 100]
    hi = [x * 1.001 for x in o]
    lo = [x * 0.999 for x in o]
    hi[5] = 103.0  # profit-take touched in bar 5
    df = _path_bars(o, hi, lo)
    lab = triple_barrier(df, horizon=3, pt_mult=1.0, sl_mult=1.0, vol_window=2)
    sig = lab["sigma"].iloc[3]
    assert np.isfinite(sig)
    row = lab.iloc[3]  # entry at open[4] = 100, scans bars 4..6
    assert 100 * np.exp(sig) < 103.0 and 100 * np.exp(-sig) < 99.9  # only the bar-5 spike can touch
    assert row["label"] == 1 and row["barrier"] == "pt"
    assert row["t1"] == df.index[6]  # touched in bar 5 => known at bar 6's open
    assert row["ret"] == pytest.approx(sig)


def test_triple_barrier_stop_wins_ties_and_vertical_barrier():
    o = [100.0] * 12
    hi, lo = [100.0] * 12, [100.0] * 12
    hi[3], lo[3] = 150.0, 50.0  # both barriers in one bar => stop assumed first
    df = _path_bars(o, hi, lo)
    df["close"] = df["open"] * np.exp(np.tile([0.01, -0.01], 6))  # nonzero sigma
    lab = triple_barrier(df, horizon=2, vol_window=2)
    assert lab["label"].iloc[2] == -1  # first row with sigma; entry open[3], bars 3..4
    assert lab["t1"].iloc[2] == df.index[4]
    assert lab["label"].iloc[5] == 0 and lab["barrier"].iloc[5] == "vb"
    assert lab["t1"].iloc[5] == df.index[8]
    assert lab["label"].iloc[-3:].isna().all()


def _daily(seed: int, mu: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", "2026-06-30", freq="D", tz="UTC")
    return pd.Series(rng.normal(mu, 0.01, len(idx)), index=idx)


def test_registry_counts_distinct_configs_and_saves_returns(isolated):
    m = {"sharpe": 0.5, "sharpe_daily": 0.026}
    for thr in (0.5, 0.55, 0.6):
        for cost in (5.0, 10.0):
            trials.log_trial("lr", {"thr": thr}, "15m", "BTCUSDT", cost, m, _daily(int(thr * 100)), "h", 7)
    trials.log_trial("rules", {"fast": 10}, "1h", "BTCUSDT", 5.0, m, _daily(1), "h", 7, kind="baseline")
    trials.log_trial("bench", {}, "1h", "BTCUSDT", 5.0, m, _daily(2), "h", None, kind="benchmark")
    assert trials.n_trials() == 4  # 3 thresholds + 1 baseline; cost re-scoring and benchmarks excluded
    rec = json.loads((config.TRIALS_DIR / "lr.jsonl").read_text().splitlines()[0])
    assert {"git_sha", "data_hash", "seed", "params", "timeframe", "symbol", "metrics"} <= rec.keys()
    assert trials.returns_matrix("BTCUSDT").shape[1] == 4
    assert trials.best_baseline_sharpe("BTCUSDT") == 0.5
    with pytest.raises(HoldoutViolation):
        late = pd.Series([0.0], index=[pd.Timestamp("2026-07-02", tz="UTC")])
        trials.log_trial("lr", {"thr": 1}, "15m", "BTCUSDT", 5.0, m, late, "h", 7)


def test_promotion_fails_loudly_on_a_mediocre_candidate(isolated, bar_factory):
    bars, base, stress = {}, {}, {}
    for i, sym in enumerate(config.SYMBOLS):
        df = bar_factory(n=288 * 40, seed=20 + i, start="2020-01-01")
        target = (df["close"].pct_change(12) > 0).astype(float)
        bars[sym], base[sym], stress[sym] = df, run_backtest(df, target, 5), run_backtest(df, target, 10)
        trials.log_trial("mom", {"h": 12}, "5m", sym, 5.0, summarize(base[sym]), base[sym].daily, "h", 0)
    table = evaluate(Candidate("mom", {"h": 12}, "5m", base, stress, bars), n_perm=50)
    assert verdict(table) == "FAIL"
    assert set(table["symbol"]) == set(config.SYMBOLS)
    for name in ("net Sharpe @5bps", "covers 2020-01..2026-06", "Sharpe vs best rule baseline"):
        rows = table[table["criterion"] == name]
        assert len(rows) == 2 and not rows["pass"].any()
    assert table[table["criterion"] == "logged in trial registry"]["pass"].all()
