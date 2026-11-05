"""LightGBM books at a month boundary, after an unusable model, and on a bar the other coin lacks.

The research (``experiments/intraday/lgbm.py: oos_positions``) decides every 4h bar OPENING in
month M with fold M's model and threshold, flat for the month when fold M found no threshold,
and it never sees a bar that only one coin has (``features.align``), so it holds its position
across such a bar. These tests pin the live book to that.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest
from lgbm_helpers import CONFIG, dated_4h, fake_trained, frames_5m
from live_helpers import FakeSource, book, config, make_5m

from profit_engine.live import macos, paths, queries, training
from profit_engine.live.runner import tick
from profit_engine.live.store import Store
from profit_engine.live.strategies import lgbm_artifacts as art
from profit_engine.live.strategies.lgbm import LgbmStrategy
from profit_engine.live.strategy import StrategyUnavailable, call_decide

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def _publish(models, symbol: str, month: str, at: str):
    return art.publish(fake_trained(symbol, month), models, T(at))


# ----------------------------------------------------------------------- finding 1: stale model


def test_a_bar_of_a_new_month_is_never_decided_with_last_months_model(tmp_path):
    _publish(tmp_path, "BTCUSDT", "2026-10", "2026-10-01 07:00")
    s = LgbmStrategy("BTCUSDT", models_dir=tmp_path)
    b = dated_4h("2026-11-01 00:00")  # the first November bar; only the October model exists
    with pytest.raises(StrategyUnavailable) as err:
        s.decide(b["BTCUSDT"], b["ETHUSDT"])
    text = str(err.value)
    assert "November 2026" in text and "signal-desk-live retrain" in text and "keeps its position" in text


def test_each_bar_is_decided_by_the_model_for_the_month_it_opens_in(tmp_path):
    oct_ = _publish(tmp_path, "BTCUSDT", "2026-10", "2026-10-01 07:00")
    nov = _publish(tmp_path, "BTCUSDT", "2026-11", "2026-11-01 00:10")
    s = LgbmStrategy("BTCUSDT", models_dir=tmp_path)
    assert s.loaded.folder == nov  # the newest is what the dashboard shows...
    late = dated_4h("2026-10-31 20:00")  # ...but an October bar decided after the retrain uses October's
    assert s.decide(late["BTCUSDT"], late["ETHUSDT"]).details["model"] == oct_.name
    first = dated_4h("2026-11-01 00:00")
    d = s.decide(first["BTCUSDT"], first["ETHUSDT"])
    assert d.details["model"] == nov.name and s.meta.version == nov.name and "2026-11" in d.explanation


def test_a_month_without_a_profitable_threshold_keeps_the_previous_model(tmp_path, monkeypatch):
    """Documented live behaviour (Known limitations): the research stays in cash that month."""
    oct_ = _publish(tmp_path, "BTCUSDT", "2026-10", "2026-10-01 07:00")

    def no_threshold(*_a, **_k):
        raise art.NoThreshold("flat")

    monkeypatch.setattr(training.art, "train", no_threshold)
    with pytest.raises(art.NoThreshold):
        training.train_and_publish(frames_5m("2026-11-01 01:00"), "BTCUSDT", T("2026-11-01"), tmp_path)
    b = dated_4h("2026-11-01 00:00")
    d = LgbmStrategy("BTCUSDT", models_dir=tmp_path).decide(b["BTCUSDT"], b["ETHUSDT"])
    assert d.details["model"] == oct_.name and "found no setting that made money" in d.explanation


def _first_fire_after(start_utc: pd.Timestamp, tz: str, entries: list[dict]) -> pd.Timestamp:
    """When launchd (which reads StartCalendarInterval on the Mac's LOCAL clock) next runs the job."""
    t = start_utc.tz_convert(tz).floor("min")
    while not any(all(getattr(t, k.lower()) == v for k, v in e.items()) for e in entries):
        t += pd.Timedelta("1min")
    return t.tz_convert("UTC")


@pytest.mark.parametrize("tz", ["America/Los_Angeles", "Asia/Tokyo", "Asia/Kolkata", "Pacific/Kiritimati", "UTC"])
def test_retrain_runs_before_the_first_4h_bar_of_a_new_utc_month_closes(tz):
    plist = macos.retrain_plist(macos.InstallPlan(python=paths.REPO_ROOT, repo=paths.REPO_ROOT,
                                                  home=paths.REPO_ROOT, logs=paths.REPO_ROOT))  # fmt: skip
    for month_start in ("2026-12-01", "2027-04-01"):  # PST and PDT
        fired = _first_fire_after(T(month_start), tz, plist["StartCalendarInterval"])
        assert T(month_start) < fired < T(month_start) + pd.Timedelta("4h")  # the first 4h bar closes at 04:00
        assert art.month_start(fired) == T(month_start)  # so retrain's cutoff is the new month


def test_retrain_that_finds_no_threshold_is_not_repeated_every_hour(monkeypatch):
    calls = []

    def no_threshold(_f, symbol, *_a):
        calls.append(symbol)
        raise art.NoThreshold("flat")

    monkeypatch.setattr(training, "load_5m", lambda *a, **k: frames_5m("2026-11-05"))
    monkeypatch.setattr(training.art, "train", no_threshold)
    now = T("2026-11-01 00:10")
    assert {r["status"] for r in training.retrain(now=now)} == {"kept_old"}
    assert {r["status"] for r in training.retrain(now=now + pd.Timedelta("1h"))} == {"up_to_date"}
    assert sorted(calls) == ["BTCUSDT", "ETHUSDT"]


# ----------------------------------------------------------------------- finding 2: unusable model


def _break(folder, how: str) -> None:
    if how == "versions":
        meta = json.loads((folder / "metadata.json").read_text())
        meta["versions"]["lightgbm"] = "3.0.0"
        (folder / "metadata.json").write_text(json.dumps(meta))
    else:
        with (folder / "model.txt").open("a") as fh:
            fh.write("\n")


@pytest.mark.parametrize("how", ["versions", "altered"])
def test_retrain_replaces_this_months_model_when_it_cannot_be_used(monkeypatch, how):
    models = paths.models_dir()
    for s in art.OTHER:
        _break(_publish(models, s, "2026-11", "2026-11-01 00:10"), how)
    assert LgbmStrategy("BTCUSDT").problem
    monkeypatch.setattr(training, "load_5m", lambda *a, **k: frames_5m("2026-11-05"))
    monkeypatch.setattr(training.art, "train", lambda f, s, cut, c: fake_trained(s, f"{cut:%Y-%m}"))
    results = training.retrain(now=T("2026-11-05 07:00"))
    assert [r["status"] for r in results] == ["trained", "trained"]
    for s in art.OTHER:
        fixed = LgbmStrategy(s)
        assert fixed.problem == "" and fixed.loaded.meta["trades_month"] == "2026-11"


# ----------------------------------------------------------------------- minor: no matching bar


def test_a_bar_the_other_coin_lacks_holds_the_position_as_the_research_did(tmp_path):
    _publish(tmp_path, "BTCUSDT", "2026-10", "2026-10-01 07:00")
    s = LgbmStrategy("BTCUSDT", models_dir=tmp_path)
    b = dated_4h("2026-10-31 20:00")
    eth = b["ETHUSDT"].iloc[:-1]  # no ETH bar for the decision bar
    assert getattr(s, "stateful", False)
    for prev in (0.0, 1.0):
        d = call_decide(s, b["BTCUSDT"], eth, prev)
        assert d.target == prev and "keeps its position" in d.explanation
        assert "as the backtest did" in d.explanation


# ----------------------------------------------------------------------- minor: dashboard + plain text


def _unusable_book_overview(tmp_path, quiet):
    folder = _publish(paths.models_dir(), "BTCUSDT", "2021-03", "2021-03-01 00:10")
    _break(folder, "altered")
    strategy = LgbmStrategy("BTCUSDT")
    db = tmp_path / "live.sqlite"
    store = Store(db)
    bars5 = make_5m(12 * 24 * 3, seed=4)
    src = FakeSource({"BTCUSDT": bars5, "ETHUSDT": bars5}, bars5.index[-1] + pd.Timedelta("30s"))
    tick(config(book(strategy, book_id="btc-lgbm-4h")), store, src, quiet)
    store.close()
    return strategy, queries.overview(db)


def test_an_unusable_model_is_explained_in_plain_words_on_the_book_card(tmp_path, quiet):
    strategy, ov = _unusable_book_overview(tmp_path, quiet)
    for text in (strategy.problem, ov["books"][0]["problem"]):
        assert "BTC LightGBM model can't be used" in text and "signal-desk-live retrain" in text
        assert "SHA256" not in text and "/" not in text.replace("signal-desk-live", "")
        assert not re.search(r"\d+\.\d+\.\d+", text)  # no library version strings
    html = (paths.REPO_ROOT / "src/profit_engine/live/static/index.html").read_text()
    assert "b.problem" in html and "vno model" not in html


def test_the_problem_clears_from_the_card_once_the_book_decides_again(tmp_path, quiet):
    _, ov = _unusable_book_overview(tmp_path, quiet)
    assert ov["books"][0]["problem"]
    store = Store(tmp_path / "live.sqlite")
    try:
        store.conn.execute(
            "INSERT INTO decisions (book_id, bar_open_ts, decided_at, target, prev_target, explanation, "
            "strategy_version) VALUES ('btc-lgbm-4h', '2099-01-01T00:00:00Z', '2099-01-01T04:00:01Z', 0, 0, 'x', 'v')"
        )
    finally:
        store.close()
    assert queries.overview(tmp_path / "live.sqlite")["books"][0]["problem"] is None


def test_problem_text_still_names_the_cause_and_logs_the_detail(tmp_path):
    folder = _publish(tmp_path, "ETHUSDT", "2026-10", "2026-10-01 07:00")
    _break(folder, "versions")
    s = LgbmStrategy("ETHUSDT", models_dir=tmp_path)
    assert "different library versions" in s.problem
    assert "lightgbm 3.0.0" in s.problem_detail and CONFIG in str(folder)
