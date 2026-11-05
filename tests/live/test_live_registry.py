"""books.json, model artifacts, the demo label, and notification throttling."""

from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from live_helpers import Const, make_5m
from sklearn.linear_model import LogisticRegression

from profit_engine.intraday.features import make_feature_frame
from profit_engine.intraday.resample import resample
from profit_engine.live import paths
from profit_engine.live.models import ArtifactError
from profit_engine.live.notify import Notifier, summarize
from profit_engine.live.registry import ConfigError, load_config, parse_config
from profit_engine.live.strategies.lgbm import LgbmStrategy
from profit_engine.live.strategies.voltarget import VolTargetTrend
from profit_engine.live.strategy import DEMO_LABEL, InvalidTarget, SmaTrendDemo, normalize_decision


def const_factory(meta: dict) -> Const:
    """Used by a kind="python" artifact below (entry 'test_live_registry:const_factory')."""
    return Const(0.5, symbol=meta["symbol"], timeframe=meta["timeframe"])


def test_default_config_is_the_four_research_books_and_no_demo():
    cfg = load_config()  # no user books.json in the isolated app folder: the repo default
    assert cfg.source == paths.DEFAULT_BOOKS
    ids = {b.book_id: b for b in cfg.books}
    assert sorted(ids) == ["btc-lgbm-4h", "btc-trend-1h", "eth-lgbm-4h", "eth-trend-1h"]
    for b in cfg.books:
        assert b.starting_equity == 1000.0 and b.cost_bps_per_side == 5.0 and b.enabled
        assert not isinstance(b.strategy, SmaTrendDemo) and not b.strategy.meta.promoted
    trend, lgbm = ids["btc-trend-1h"].strategy, ids["eth-lgbm-4h"].strategy
    assert isinstance(trend, VolTargetTrend) and trend.config == "vt_1h_slow4_z_s30" and trend.timeframe == "1h"
    assert trend.meta.label == "not promoted — best research candidate (forward test)"
    assert isinstance(lgbm, LgbmStrategy) and lgbm.config == "lgbm_4h_h30_pt2_sl1" and lgbm.timeframe == "4h"
    assert lgbm.meta.label == "not promoted — comparison model (backtest trailed buy-and-hold)"
    assert lgbm.other_symbol == "BTCUSDT" and "train lgbm --symbol ETHUSDT" in lgbm.problem  # no model trained here
    assert DEMO_LABEL not in {b.strategy.meta.label for b in cfg.books}


def test_bad_configs_are_rejected():
    entry = {"book_id": "x", "strategy": "demo_sma_trend", "symbol": "BTCUSDT", "timeframe": "4h"}
    with pytest.raises(ConfigError, match="twice"):
        parse_config({"books": [entry, entry]})
    with pytest.raises(ConfigError, match="symbol"):
        parse_config({"books": [{**entry, "symbol": "DOGEUSDT"}]})
    with pytest.raises(ConfigError, match="unknown strategy"):
        parse_config({"books": [{**entry, "strategy": "nope"}]})
    with pytest.raises(ConfigError, match="rejected"):
        parse_config({"books": [{**entry, "params": {"fast": 30, "slow": 10}}]})


def test_targets_are_validated():
    assert normalize_decision(float("nan")).target == 0.0  # NaN is flat, as in the harness
    for bad in (1.5, -0.1, float("inf")):
        with pytest.raises(InvalidTarget):
            normalize_decision(bad)


def _write_model(folder, bars, promoted: bool) -> None:
    x = make_feature_frame(bars)[["ret_1", "ret_3"]].dropna()
    y = (bars["close"].shift(-1).loc[x.index] > bars["close"].loc[x.index]).astype(int)
    folder.mkdir(parents=True)
    joblib.dump(LogisticRegression(random_state=0).fit(x, y), folder / "model.joblib")
    meta = {
        "name": "logreg_demo", "version": "2026-11-05", "symbol": "BTCUSDT", "timeframe": "1h", "warmup_bars": 10,
        "kind": "proba_threshold", "feature_columns": ["ret_1", "ret_3"], "threshold": 0.5,
        "artifact": "model.joblib",
        "artifact_sha256": hashlib.sha256((folder / "model.joblib").read_bytes()).hexdigest(),
        "promoted": promoted, "promotion": {"verdict": "PASS" if promoted else "FAIL"},
    }  # fmt: skip
    (folder / "metadata.json").write_text(json.dumps(meta))


def test_model_artifact_book(tmp_path):
    bars = resample(make_5m(3000, seed=5), "1h", incomplete="drop")
    _write_model(tmp_path / "models" / "logreg_demo", bars, promoted=True)
    cfg = parse_config({"books": [{"book_id": "m", "model": "logreg_demo"}]}, models_dir=tmp_path / "models")
    (spec,) = cfg.books
    assert spec.strategy.meta.promoted and spec.strategy.meta.label == "promoted"
    assert len(spec.strategy.meta.artifact_hash) == 64
    decision = normalize_decision(spec.strategy.decide(bars.iloc[-10:]))
    assert decision.target in (0.0, 1.0) and "chance of a rise" in decision.explanation


def test_tampered_artifact_is_refused(tmp_path):
    bars = resample(make_5m(3000, seed=5), "1h", incomplete="drop")
    folder = tmp_path / "models" / "m1"
    _write_model(folder, bars, promoted=False)
    (folder / "model.joblib").write_bytes(b"not the model that was hashed")
    with pytest.raises(ArtifactError, match="does not match"):
        parse_config({"books": [{"book_id": "m", "model": "m1"}]}, models_dir=tmp_path / "models")


def test_python_artifact_and_unpromoted_label(tmp_path):
    folder = tmp_path / "models" / "py1"
    folder.mkdir(parents=True)
    meta = {"name": "py", "version": "3", "symbol": "ETHUSDT", "timeframe": "15m", "warmup_bars": 1,
            "kind": "python", "entry": "test_live_registry:const_factory", "promoted": True,
            "promotion": {"verdict": "FAIL"}}  # fmt: skip
    (folder / "metadata.json").write_text(json.dumps(meta))
    (spec,) = parse_config({"books": [{"book_id": "p", "model": str(folder)}]}).books
    assert spec.symbol == "ETHUSDT" and spec.timeframe == "15m"
    assert not spec.strategy.meta.promoted  # promoted:true without a PASS verdict is not promoted


def test_notifications_are_throttled_and_summarised(store):
    sent: list[str] = []
    n = Notifier(True, 300, sender=lambda _t, text: sent.append(text) or True)
    t0 = pd.Timestamp("2021-03-01 05:00+00:00")
    for i in range(4):
        n.queue(store, "b1", f"fill {i}", t0)
    assert n.flush(store, t0, {"b1": "BTC"}) == 1
    assert sent == [summarize("BTC", ["fill 0", "fill 1", "fill 2", "fill 3"])] and "+1 earlier" in sent[0]
    n.queue(store, "b1", "fill 4", t0 + pd.Timedelta("1min"))
    assert n.flush(store, t0 + pd.Timedelta("2min")) == 0  # within 5 minutes: held back
    assert n.flush(store, t0 + pd.Timedelta("5min")) == 1
    assert sent[-1] == "fill 4"
    Notifier(False).queue(store, "b1", "ignored", t0)
    assert store.pending_notifications() == []


def test_demo_strategy_is_finite_window():
    s = SmaTrendDemo("BTCUSDT", "4h", fast=2, slow=4)
    bars = pd.DataFrame({"close": np.array([1.0, 1, 1, 1, 2, 3])})
    assert s.decide(bars).target == 1.0
    assert s.decide(bars.iloc[-4:]).target == s.decide(bars).target
