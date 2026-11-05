"""A bar is traded at most once: duplicate ticks, concurrent ticks, and crashes mid-commit."""

from __future__ import annotations

import threading

import pandas as pd
import pytest
from live_helpers import Const, FakeSource, book, config, make_5m, rows

from profit_engine.live.runner import tick
from profit_engine.live.store import BarCommit, Store
from profit_engine.live.strategy import Decision

NOW = "2021-03-01 05:00:30+00:00"  # 30s after the 04:00 1h bar closed


@pytest.fixture
def frames():
    return {"BTCUSDT": make_5m(400)}


def test_same_bar_ticked_twice_trades_once(store, quiet, frames):
    cfg = config(book(Const(1.0)))
    src = FakeSource(frames, NOW)
    first = tick(cfg, store, src, quiet)
    assert first["books"][0]["decision"] == "traded"
    second = tick(cfg, store, src, quiet)
    assert second["books"][0]["decision"] == "up_to_date"
    src.now = pd.Timestamp("2021-03-01 05:40:00+00:00")  # later in the same bar: still nothing new
    third = tick(cfg, store, src, quiet)
    assert third["books"][0]["decision"] == "up_to_date"
    assert len(rows(store, "SELECT * FROM fills")) == 1
    assert len(rows(store, "SELECT * FROM decisions")) == 1


def test_commit_bar_is_idempotent_at_the_store(store, frames):
    spec = book(Const(1.0))
    now = pd.Timestamp(NOW)
    store.upsert_book(spec, now)
    bar = pd.Timestamp("2021-03-01 04:00+00:00")
    bc = BarCommit(bar, bar + pd.Timedelta("1h"), Decision(1.0, "x"), 100.0, 100.0, 5.0, now, "t1")
    assert store.commit_bar("b1", bc).fill is not None
    again = store.commit_bar("b1", bc)
    assert not again.claimed and again.fill is None
    older = BarCommit(bar - pd.Timedelta("1h"), bar, Decision(0.0, "x"), 100.0, 100.0, 5.0, now, "t1")
    assert not store.commit_bar("b1", older).claimed  # never go back in time either
    assert len(rows(store, "SELECT * FROM fills")) == 1


def test_two_concurrent_ticks_make_one_fill(tmp_path, quiet, frames):
    strategy = Const(1.0)
    strategy.barrier = threading.Barrier(2)  # both threads are inside decide() before either commits
    cfg = config(book(strategy))
    db = tmp_path / "race.sqlite"
    Store(db).close()
    results, errors = [], []

    def worker() -> None:
        s = Store(db)
        try:
            results.append(tick(cfg, s, FakeSource(frames, NOW), quiet))
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            s.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors
    statuses = sorted(r["books"][0]["decision"] for r in results)
    assert statuses == ["already_done", "traded"]
    check = Store(db)
    assert len(rows(check, "SELECT * FROM fills")) == 1
    assert len(rows(check, "SELECT * FROM decisions")) == 1
    check.close()


def test_crash_mid_transaction_persists_nothing_and_bar_can_be_retried(store, quiet, frames, monkeypatch):
    cfg = config(book(Const(1.0)))
    src = FakeSource(frames, NOW)

    def boom(*_args, **_kw):
        raise RuntimeError("disk vanished between claim and fill")

    monkeypatch.setattr(Store, "_insert_fill", boom)
    result = tick(cfg, store, src, quiet)
    assert result["books"][0]["decision"] == "error"
    assert rows(store, "SELECT * FROM decisions") == []  # the claim rolled back with the fill
    assert rows(store, "SELECT * FROM fills") == []
    b = store.book("b1")
    assert b["last_bar_ts"] is None and b["units"] == 0 and b["cash"] == 10_000
    err = rows(store, "SELECT kind, message, data_json FROM events WHERE kind='error'")
    assert len(err) == 1 and "disk vanished" not in err[0]["message"]  # plain words, no raw error

    monkeypatch.undo()
    retry = tick(cfg, store, src, quiet)
    assert retry["books"][0]["decision"] == "traded"
    assert len(rows(store, "SELECT * FROM fills")) == 1
