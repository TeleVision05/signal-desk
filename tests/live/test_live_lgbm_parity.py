"""LightGBM 4h: the live ``train`` command reproduces the research fold, and the live book then
decides exactly as the research target did.

Research fold: ``experiments/intraday/lgbm.py`` for config ``lgbm_4h_h30_pt2_sl1`` and the test
month 2026-06 (the last OOS month), on the cached bars. Live: ``signal-desk-live train lgbm
--month 2026-06`` reading the same cache, then ``LgbmStrategy.decide`` on each June 4h bar with
the last ``warmup_bars`` closed bars.
"""

from __future__ import annotations

import json
from functools import cache

import numpy as np
import pandas as pd
import pytest
from parity_helpers import bars, raw_5m, require_cache, research

from profit_engine.intraday import config, labels, splits
from profit_engine.live import cli
from profit_engine.live.strategies import lgbm_artifacts as art
from profit_engine.live.strategies import lgbm_core as core
from profit_engine.live.strategies.lgbm import LgbmStrategy
from profit_engine.live.strategy import decision_window, other_window

CONFIG = "lgbm_4h_h30_pt2_sl1"
MONTH = pd.Timestamp("2026-06-01", tz="UTC")
SYMBOLS = ("BTCUSDT", "ETHUSDT")


@cache
def research_fold(symbol: str) -> dict:
    """The research's own fold for MONTH: model, threshold, test-row probabilities and target."""
    lg = research("lgbm")
    cfg, other = lg.GRID_BY_NAME[CONFIG], lg._other(symbol)
    xy = lg.build_xy(bars(symbol, "4h"), bars(other, "4h"), cfg)
    hi = MONTH + pd.DateOffset(months=1)
    (fold,) = splits.walk_forward_splits(
        xy["df"].index, xy["t1"], test_months=1, eval_start=MONTH, eval_end=hi, min_train_rows=2000
    )
    model, thr, q = lg.fit_fold(xy, fold.train & xy["has_x"] & xy["has_y"], fold)
    test = np.flatnonzero(fold.test & xy["has_x"])
    target, info = lg.oos_positions(bars(symbol, "4h"), bars(other, "4h"), cfg, MONTH, hi)
    assert info["thr"] == [thr]  # oos_positions ran the very same fold
    return {"xy": xy, "thr": thr, "q": q, "n_train": int((fold.train & xy["has_x"] & xy["has_y"]).sum()),
            "x_test": xy["x"].iloc[test], "p": model.predict_proba(xy["x"].iloc[test])[:, 1],
            "target": target[(target.index >= MONTH) & (target.index < hi)]}  # fmt: skip


def test_ported_constants_and_labels_are_the_research_ones():
    lg = research("lgbm")
    cfg = lg.GRID_BY_NAME[CONFIG]
    assert core.CONFIGS[CONFIG] == {k: cfg[k] for k in ("tf", "h", "pt", "sl")}
    assert core.MODEL_PARAMS == lg.MODEL_PARAMS and core.SEED == lg.SEED and core.INNER_SPLITS == lg.INNER_SPLITS
    assert core.LABEL_MIN_RET == lg.LABEL_MIN_RET and np.array_equal(core.QUANTILES, lg.QUANTILES)
    assert core.feature_params("4h") == lg.feature_params("4h")
    df = bars("BTCUSDT", "4h").iloc[-3000:]
    want = labels.triple_barrier(df, 30, 2.0, 1.0, vol_window=42)
    got = core.triple_barrier(df, 30, 2.0, 1.0, vol_window=42)
    for col in ("label", "ret"):
        assert np.array_equal(got[col].to_numpy(), want[col].to_numpy(), equal_nan=True)
    assert got["t1"].equals(want["t1"])
    assert np.array_equal(core.forward_return_1(df), labels.forward_return(df, 1)["ret"].to_numpy(), equal_nan=True)


@pytest.fixture
def trained(tmp_path, monkeypatch):
    """Run the real training command for MONTH on the research cache (no download)."""
    require_cache()
    monkeypatch.setenv("SIGNAL_DESK_MODELS_DIR", str(tmp_path / "models"))
    out = {}
    for symbol in SYMBOLS:
        argv = ["train", "lgbm", "--symbol", symbol, "--month", "2026-06", "--data-dir", str(config.CACHE_DIR),
                "--no-sync"]  # fmt: skip
        assert cli.main(argv) == 0
        out[symbol] = LgbmStrategy(symbol, models_dir=tmp_path / "models")
    return out


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_train_command_reproduces_the_research_fold(trained, symbol):
    want, strategy = research_fold(symbol), trained[symbol]
    meta = strategy.loaded.meta
    assert strategy.problem == "" and meta["trades_month"] == "2026-06"
    assert meta["threshold"] == want["thr"] and meta["threshold_quantile"] == want["q"]  # bitwise
    assert meta["train_window"]["n_rows"] == want["n_train"]
    assert meta["train_window"]["end_exclusive"] == MONTH.isoformat()
    assert meta["feature_columns"] == list(want["xy"]["x"].columns)
    p = strategy.loaded.booster.predict(want["x_test"][meta["feature_columns"]])
    assert np.array_equal(p, want["p"])  # bitwise, through the saved text model
    assert set(meta["versions"]) == {"lightgbm", "sklearn", "numpy", "pandas"} and meta["git_sha"]
    assert set(meta["data_hash"]) == set(SYMBOLS) and meta["model_params"]["seed"] == 7
    on_disk = json.loads((strategy.loaded.folder / "metadata.json").read_text())
    assert on_disk["artifact_sha256"] == strategy.loaded.sha256


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_live_decisions_on_the_following_month_equal_the_research_target(trained, symbol):
    want, strategy = research_fold(symbol), trained[symbol]
    four_h, other = bars(symbol, "4h"), bars(strategy.other_symbol, "4h")
    probs = pd.Series(want["p"], index=want["x_test"].index)
    got = {}
    for ts in want["target"].index:
        window = decision_window(four_h.loc[:ts], strategy)
        d = strategy.decide(window, other_window(other.loc[:ts], window))
        got[ts] = d.target
        if ts in probs.index:
            assert d.details["p_up"] == probs[ts]  # features from a 3000-bar window, same probability
    got = pd.Series(got)
    assert len(got) == 180 and 0 < got.sum() < len(got)
    assert got.equals(want["target"].astype(float))


def test_training_uses_only_closed_bars_before_the_cutoff():
    """Bars at or after the cutoff (the month the model trades) cannot change the model."""
    require_cache()
    frames = {s: raw_5m(s) for s in SYMBOLS}
    poisoned = {s: f.copy() for s, f in frames.items()}
    for f in poisoned.values():
        f.loc[f.index >= MONTH, ["open", "high", "low", "close"]] *= 3.0
    a = art.train(frames, "ETHUSDT", MONTH, CONFIG)
    b = art.train(poisoned, "ETHUSDT", MONTH, CONFIG)
    assert a.threshold == b.threshold and a.booster.model_to_string() == b.booster.model_to_string()
