"""Plumbing for the research books: schema v2, the strategy's own state across a pause, the
dashboard's groups, labels and buy-and-hold reference, install's books.json copy, and the
monthly retrain job."""

from __future__ import annotations

import json
import plistlib
import re
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from live_helpers import Const, FakeSource, book, config, make_5m, rows

from profit_engine.intraday.binance import RAW_COLUMNS
from profit_engine.live import cli, macos, market, paths, queries
from profit_engine.live.runner import tick
from profit_engine.live.store import SCHEMA_V1, BarCommit, Store
from profit_engine.live.strategies.voltarget import _explain
from profit_engine.live.strategy import Decision, StrategyMeta

SNAPSHOTS = Path(__file__).parent / "snapshots"


def test_a_version_1_database_is_upgraded_in_place(tmp_path):
    db = tmp_path / "old.sqlite"
    with sqlite3.connect(db) as conn:
        for statement in SCHEMA_V1:
            conn.execute(statement)
        conn.execute(
            "INSERT INTO books (book_id, title, symbol, timeframe, strategy, strategy_version, strategy_source, "
            "params_json, starting_equity, cash, created_at, updated_at) "
            "VALUES ('b', 'B', 'BTCUSDT', '1h', 'demo', '1', 'builtin', '{}', 1000, 1000, 't', 't')"
        )
        conn.execute("PRAGMA user_version=1")
    store = Store(db)
    assert store.user_version() == 2
    (b,) = rows(store, "SELECT book_id, cash, status_label, family FROM books")
    assert b == {"book_id": "b", "cash": 1000.0, "status_label": "", "family": ""}
    store.close()


def test_a_stateful_strategy_continues_from_its_own_target_after_a_pause(store):
    store.upsert_book(book(Const(0.6)), pd.Timestamp("2026-01-01", tz="UTC"))
    assert store.strategy_target("b1") == 0.0  # a new book starts flat
    t0 = pd.Timestamp("2026-01-01 01:00", tz="UTC")
    store.conn.execute("UPDATE books SET paused_until='2026-01-02T00:00:00Z' WHERE book_id='b1'")
    store.commit_bar("b1", BarCommit(t0, t0 + pd.Timedelta("1h"), Decision(0.6, "x"), 100.0, 100.0, 5.0,
                                     t0 + pd.Timedelta("1h"), "t1"))  # fmt: skip
    (d,) = rows(store, "SELECT target FROM decisions")
    assert d["target"] == 0.0  # the pause kept the book in cash...
    assert store.strategy_target("b1") == 0.6  # ...but the strategy's own state is what it asked for


def test_dashboard_groups_books_by_strategy_with_labels_and_buy_and_hold(tmp_path, quiet):
    db = tmp_path / "live.sqlite"
    store = Store(db)
    bars = make_5m(400, seed=2)
    a, b = Const(1.0), Const(0.0)
    a.meta = StrategyMeta("1", status="not promoted — best research candidate (forward test)", family="Trend")
    b.meta = StrategyMeta("1", status="not promoted — comparison model", family="LightGBM 4h")
    cfg = config(book(a, book_id="a"), book(b, book_id="z"))
    src = FakeSource({"BTCUSDT": bars}, "2021-03-01 05:00:30+00:00")
    tick(cfg, store, src, quiet)
    src.now = pd.Timestamp("2021-03-01 09:00:30+00:00")
    tick(cfg, store, src, quiet)
    store.close()
    ov = queries.overview(db)
    assert [(x["book_id"], x["family"], x["label"]) for x in ov["books"]] == [
        ("a", "Trend", "not promoted — best research candidate (forward test)"),
        ("z", "LightGBM 4h", "not promoted — comparison model"),
    ]
    first = bars.at[pd.Timestamp("2021-03-01 04:55", tz="UTC"), "close"]  # newest closed 5m bar at each tick
    last = bars.at[pd.Timestamp("2021-03-01 08:55", tz="UTC"), "close"]
    for x in ov["books"]:
        assert x["bh_start_price"] == first and x["bh_equity"] == 10_000.0 * last / first
        assert x["bh_return_pct"] == pytest.approx((last / first - 1) * 100, rel=1e-12)
    assert ov["books"][1]["equity"] == 10_000.0  # the cash book is not the buy-and-hold number
    html = (paths.REPO_ROOT / "src/profit_engine/live/static/index.html").read_text()
    assert "Buy and hold" in html and "b.family" in html and "b.label" in html


def test_install_copies_the_default_books_once_and_backs_up_before_replacing(tmp_path):
    said: list[str] = []
    user = paths.user_books_path()
    cli.install_books(use_default=False, dry_run=True, out=said.append)
    assert not user.exists() and "Would copy" in said[-1]
    cli.install_books(use_default=False, dry_run=False, out=said.append)
    assert user.read_text() == paths.DEFAULT_BOOKS.read_text() and paths.books_path() == user
    edited = json.loads(user.read_text())
    edited["books"] = edited["books"][:1]
    user.write_text(json.dumps(edited))
    cli.install_books(use_default=False, dry_run=False, out=said.append)
    assert json.loads(user.read_text()) == edited and "Kept your books settings" in said[-1]
    cli.install_books(use_default=True, dry_run=False, out=said.append)
    (backup,) = user.parent.glob("books.json.bak-*")
    assert json.loads(backup.read_text()) == edited and user.read_text() == paths.DEFAULT_BOOKS.read_text()


def _plan(home: Path) -> macos.InstallPlan:
    return macos.InstallPlan(python=Path("/opt/signal-desk/.venv/bin/python"), repo=Path("/opt/signal-desk"),
                             home=home, logs=home / "Library" / "Logs" / "SignalDesk")  # fmt: skip


def test_monthly_retrain_job_is_scheduled_not_kept_alive(tmp_path):
    p = macos.retrain_plist(_plan(Path("/Users/demo")))
    assert macos.render(p).decode() == (SNAPSHOTS / "com.signaldesk.retrain.plist").read_text()
    assert p["ProgramArguments"][-1] == "retrain" and "KeepAlive" not in p and p["RunAtLoad"] is False
    assert p["StartCalendarInterval"] == [{"Minute": 10}]  # every hour: see test_live_lgbm_month.py
    ran, said = [], []
    plan = _plan(tmp_path)
    assert macos.install(plan, run=lambda c: ran.append(c) or 0, out=said.append, uid=501) == 0
    assert (plan.agents_dir / "com.signaldesk.retrain.plist").exists()
    assert any("Scheduled com.signaldesk.retrain" in s for s in said)
    macos.uninstall(plan, run=lambda c: ran.append(c) or 0, out=said.append, uid=501)
    assert ["launchctl", "bootout", "gui/501/com.signaldesk.retrain"] in ran
    assert not (plan.agents_dir / "com.signaldesk.retrain.plist").exists()
    assert plistlib.loads(macos.render(p))["Label"] == "com.signaldesk.retrain"


def test_bar_source_keeps_the_deep_history_and_fetches_only_what_it_lacks(monkeypatch):
    idx = pd.date_range("2026-01-01", "2026-03-01", freq="5min", tz="UTC", inclusive="left")
    calls: list[tuple[int, int | None]] = []

    def fetch(_symbol, _interval, start_ms, _session, end_ms=None, timeout=30):
        calls.append((start_ms, end_ms))
        ms = idx.as_unit("ms").asi8
        keep = (ms >= start_ms) & ((ms < end_ms) if end_ms is not None else True)
        out = [[str(o), "1", "1", "1", "1", "1", str(o + 299_999), "1", "1", "1", "1", "0"] for o in ms[keep]]
        return pd.DataFrame(out, columns=list(RAW_COLUMNS)).astype(str)

    monkeypatch.setattr(market.binance, "fetch_rest", fetch)
    src = market.BinanceBarSource(retries=0)
    end = idx[-1]
    ms = lambda t: int(t.value // 1_000_000)  # noqa: E731
    assert len(src.klines_5m("BTCUSDT", end - pd.Timedelta("10D"))) == 10 * 288 + 1
    deep = src.klines_5m("BTCUSDT", end - pd.Timedelta("40D"))  # older than the cache: fetch only the older part
    assert deep.index[0] == end - pd.Timedelta("40D") and len(deep) == 40 * 288 + 1
    assert calls[1] == (ms(end - pd.Timedelta("40D")), ms(end - pd.Timedelta("10D")))
    src.klines_5m("BTCUSDT", end - pd.Timedelta("30min"))  # a mark: must not trim the 40 days away
    calls.clear()
    assert len(src.klines_5m("BTCUSDT", end - pd.Timedelta("40D"))) == 40 * 288 + 1
    assert all(start >= ms(end - pd.Timedelta("15min")) for start, _ in calls)  # only the newest bars again


def test_trend_explanations_are_plain_and_match_the_decision():
    flat = {"raw": 0.01, "score": 0.0, "vol": 0.46, "scale": 0.6}
    assert _explain("BTC", flat, 0.0, 0.0, 0.2).endswith("holding no BTC, so the book stays in cash.")
    assert _explain("BTC", flat, 0.4, 0.0, 0.2).endswith("so the book sells to cash.")
    held = {"raw": 0.45, "score": 0.9, "vol": 0.5, "scale": 0.5}
    assert "keeps its position" in _explain("ETH", held, 0.4, 0.4, 0.2)
    assert _explain("ETH", held, 0.0, 0.45, 0.2).endswith("moves from 0% to 45% in ETH.")


def test_a_second_backup_in_the_same_second_never_overwrites_the_first(monkeypatch):
    said: list[str] = []
    user = paths.user_books_path()
    cli.install_books(use_default=False, dry_run=False, out=said.append)
    first = json.loads(user.read_text())
    first["starting_equity"] = 5000
    user.write_text(json.dumps(first))
    fixed = pd.Timestamp("2026-11-05 20:38:16", tz="UTC")
    monkeypatch.setattr(cli, "_utc_now", lambda: fixed)
    cli.install_books(use_default=True, dry_run=False, out=said.append)
    user.write_text(user.read_text() + " ")  # edited again, then replaced again within the same second
    cli.install_books(use_default=True, dry_run=False, out=said.append)
    backups = sorted(user.parent.glob("books.json.bak-*"))
    assert len(backups) == 2 and json.loads(backups[0].read_text()) == first


def test_dry_run_does_not_claim_it_checks_the_scheduled_retrain_job(tmp_path):
    said: list[str] = []
    macos.install(_plan(tmp_path), dry_run=True, out=said.append, uid=501)
    assert not any("Would check it is running" in s and macos.RETRAIN_LABEL in s for s in said)
    assert any("Would check it is running" in s and macos.RUNNER_LABEL in s for s in said)
    docs = (paths.REPO_ROOT / "docs/live_runner.md").read_text()
    for name in ("retrain.jsonl", "train.jsonl", "install.jsonl", "retrain.{out,err}.log"):
        assert name in docs


def test_every_test_file_a_live_docstring_cites_exists():
    src = paths.REPO_ROOT / "src/profit_engine/live"
    cited = {m for f in src.rglob("*.py") for m in re.findall(r"tests/live/\w+\.py", f.read_text())}
    assert cited and all((paths.REPO_ROOT / c).exists() for c in cited), sorted(cited)


def test_a_book_removed_from_the_settings_is_shown_as_turned_off_after_the_others(tmp_path, quiet):
    """Someone who ran the earlier default (the demo books) and then the four research books must
    not see the old demo book as an active card next to them."""
    db = tmp_path / "live.sqlite"
    store = Store(db)
    demo, trend = Const(1.0), Const(0.5)
    demo.meta = StrategyMeta("1", family="demo_sma_trend")
    trend.meta = StrategyMeta("1", status="not promoted — best research candidate (forward test)", family="Trend")
    src = FakeSource({"BTCUSDT": make_5m(400, seed=3)}, "2021-03-01 05:00:30+00:00")
    tick(config(book(demo, book_id="btc-demo-4h")), store, src, quiet)
    tick(config(book(trend, book_id="btc-trend-1h")), store, src, quiet)
    ov = queries.overview(db)
    assert [(b["book_id"], b["enabled"]) for b in ov["books"]] == [("btc-trend-1h", True), ("btc-demo-4h", False)]
    tick(config(book(demo, book_id="btc-demo-4h"), book(trend, book_id="btc-trend-1h")), store, src, quiet)
    assert {b["book_id"]: b["enabled"] for b in queries.overview(db)["books"]} == {
        "btc-trend-1h": True, "btc-demo-4h": True,
    }  # fmt: skip
    store.close()
