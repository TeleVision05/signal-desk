"""LightGBM artifacts: latest wins, wrong library versions or an altered file are refused, a
failed or flat retrain keeps the old model, and a book without a model keeps its position."""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from lgbm_helpers import CONFIG, dated_4h, fake_trained, frames_5m
from live_helpers import FakeSource, book, config, make_5m, rows

from profit_engine.live import paths, training
from profit_engine.live.runner import tick
from profit_engine.live.store import Store
from profit_engine.live.strategies import lgbm_artifacts as art
from profit_engine.live.strategies import lgbm_core as core
from profit_engine.live.strategies.lgbm import LgbmStrategy
from profit_engine.live.strategy import StrategyUnavailable


def test_newest_artifact_is_loaded_and_decides_on_closed_bars(tmp_path):
    art.publish(fake_trained("BTCUSDT", "2026-09"), tmp_path, pd.Timestamp("2026-09-01 07:00", tz="UTC"))
    newest = art.publish(fake_trained("BTCUSDT", "2026-10"), tmp_path, pd.Timestamp("2026-10-01 07:00", tz="UTC"))
    s = LgbmStrategy("BTCUSDT", models_dir=tmp_path)
    assert s.problem == "" and s.loaded.folder == newest and s.meta.version == newest.name
    b = dated_4h("2026-10-31 20:00")
    _, x = core.feature_frame(b["BTCUSDT"], b["ETHUSDT"], "4h")
    d = s.decide(b["BTCUSDT"], b["ETHUSDT"])
    p = float(s.loaded.booster.predict(x.iloc[[-1]])[0])
    assert d.details["p_up"] == p and d.target == float(p > 0.5) and d.details["model"] == newest.name


def test_other_library_versions_are_refused(tmp_path):
    folder = art.publish(fake_trained("ETHUSDT", "2026-10"), tmp_path)
    meta = json.loads((folder / "metadata.json").read_text())
    meta["versions"]["lightgbm"] = "3.0.0"
    (folder / "metadata.json").write_text(json.dumps(meta))
    s = LgbmStrategy("ETHUSDT", models_dir=tmp_path)
    assert s.loaded is None and f"lightgbm 3.0.0 (installed {lgb.__version__})" in s.problem_detail
    assert "different library versions" in s.problem and "signal-desk-live retrain" in s.problem
    with pytest.raises(StrategyUnavailable):
        s.decide(dated_4h("2026-10-31 20:00")["ETHUSDT"], dated_4h("2026-10-31 20:00")["BTCUSDT"])


def test_an_altered_model_file_is_refused(tmp_path):
    folder = art.publish(fake_trained("ETHUSDT", "2026-10"), tmp_path)
    with (folder / "model.txt").open("a") as fh:
        fh.write("\n")
    s = LgbmStrategy("ETHUSDT", models_dir=tmp_path)
    assert "SHA256" in s.problem_detail and "changed or was damaged" in s.problem


def test_a_book_without_a_model_keeps_its_position_and_says_why(store, quiet):
    bars5 = make_5m(12 * 24 * 3, seed=4)
    strategy = LgbmStrategy("BTCUSDT", models_dir=paths.models_dir())
    cfg = config(book(strategy, book_id="btc-lgbm-4h"))
    src = FakeSource({"BTCUSDT": bars5, "ETHUSDT": bars5}, bars5.index[-1] + pd.Timedelta("30s"))
    result = tick(cfg, store, src, quiet)
    (b,) = result["books"]
    assert b["decision"] == "model_unavailable" and b["marked"] and not result["ok"]
    assert rows(store, "SELECT * FROM decisions") == [] and rows(store, "SELECT * FROM fills") == []
    (ev,) = rows(store, "SELECT kind, message FROM events WHERE kind='model_unavailable'")
    assert "No trained BTC LightGBM model yet" in ev["message"] and "train lgbm --symbol BTCUSDT" in ev["message"]


@pytest.mark.parametrize(
    ("failure", "status"), [(art.NoThreshold("flat"), "kept_old"), (RuntimeError("boom"), "failed")]
)
def test_retrain_keeps_the_old_model_when_training_fails_or_finds_no_threshold(monkeypatch, failure, status):
    models = paths.models_dir()
    old = {s: art.publish(fake_trained(s, "2026-10"), models) for s in art.OTHER}
    monkeypatch.setattr(training, "load_5m", lambda *a, **k: frames_5m("2026-11-05"))

    def boom(*_a, **_k):
        raise failure

    monkeypatch.setattr(training.art, "train", boom)
    results = training.retrain(now=pd.Timestamp("2026-11-05 07:00", tz="UTC"))
    assert [r["status"] for r in results] == [status, status]
    for s, folder in old.items():
        assert art.versions(models, CONFIG, s) == [folder] and LgbmStrategy(s).loaded.folder == folder
    kinds = {r["kind"] for r in _events()}
    assert kinds == {f"model_{status}"}


def test_retrain_refuses_history_that_stops_before_the_month(monkeypatch):
    monkeypatch.setattr(training, "load_5m", lambda *a, **k: frames_5m("2026-10-31 12:00"))
    results = training.retrain(now=pd.Timestamp("2026-11-05 07:00", tz="UTC"))
    assert {r["status"] for r in results} == {"failed"} and art.versions(paths.models_dir(), CONFIG, "BTCUSDT") == []


def test_retrain_trains_once_per_month(monkeypatch):
    calls = []
    monkeypatch.setattr(training, "load_5m", lambda *a, **k: frames_5m("2026-11-05"))
    monkeypatch.setattr(training.art, "train", lambda f, s, cut, c: calls.append(s) or fake_trained(s, f"{cut:%Y-%m}"))
    now = pd.Timestamp("2026-11-05 07:00", tz="UTC")
    assert [r["status"] for r in training.retrain(now=now)] == ["trained", "trained"]
    assert [r["status"] for r in training.retrain(now=now + pd.Timedelta("1D"))] == ["up_to_date", "up_to_date"]
    assert sorted(calls) == ["BTCUSDT", "ETHUSDT"]
    assert LgbmStrategy("ETHUSDT").meta.version.startswith("2026-11-")
    assert {r["book_id"] for r in _events()} == {"btc-lgbm-4h", "eth-lgbm-4h"}


def _events() -> list[dict]:
    s = Store(paths.db_path())
    try:
        return rows(s, "SELECT book_id, kind, message FROM events")
    finally:
        s.close()


def test_training_that_finds_no_threshold_publishes_nothing(monkeypatch, tmp_path):
    """Research trades flat in such a month; live keeps the previous model instead (never a new flat one)."""
    frames = {s: make_5m(12 * 24 * 400, seed=i, vol=0.003, start="2017-09-01") for i, s in enumerate(art.OTHER)}
    cutoff = pd.Timestamp("2018-10-01", tz="UTC")
    seen = {}

    def flat_fold(xy, train, block_start):
        seen["rows"], seen["cutoff"] = int(train.sum()), block_start
        return core.make_model().fit(xy["x"].iloc[:600].fillna(0), np.arange(600) % 2), np.inf, float("nan")

    monkeypatch.setattr(core, "fit_fold", flat_fold)
    with pytest.raises(art.NoThreshold):
        training.train_and_publish(frames, "BTCUSDT", cutoff, tmp_path)
    assert seen["cutoff"] == cutoff and seen["rows"] > 1000
    assert art.versions(tmp_path, CONFIG, "BTCUSDT") == [] and not any(tmp_path.rglob("model.txt"))
