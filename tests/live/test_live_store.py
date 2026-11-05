"""Store basics: local file only, WAL, idempotent migrations; service lock and backoff."""

from __future__ import annotations

import random
import sqlite3

import pytest

from profit_engine.live import paths
from profit_engine.live.service import InstanceLock, exchange_trouble, next_delay
from profit_engine.live.store import SCHEMA_VERSION, Store


def test_db_path_ignores_database_url(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@supabase.example/db")
    assert paths.db_path() == tmp_path / "app" / "live.sqlite"
    monkeypatch.setenv("SIGNAL_DESK_LIVE_DB", str(tmp_path / "x.sqlite"))
    assert paths.db_path() == tmp_path / "x.sqlite"
    monkeypatch.setenv("SIGNAL_DESK_LIVE_DB", "postgresql://user:pw@host/db")
    with pytest.raises(ValueError, match="file path"):
        paths.db_path()


def test_store_refuses_urls():
    with pytest.raises(ValueError, match="local SQLite"):
        Store("postgresql://user:pw@host/db")


def test_wal_busy_timeout_and_schema(store):
    assert store.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30_000
    assert store.user_version() == SCHEMA_VERSION
    tables = {r[0] for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"books", "decisions", "fills", "equity_marks", "events", "heartbeat", "notifications"} <= tables


def test_migrate_is_idempotent(tmp_path):
    db = tmp_path / "m.sqlite"
    a = Store(db)
    a.conn.execute("INSERT INTO heartbeat (component, ts, status) VALUES ('x', 't', 'ok')")
    assert a.migrate() == SCHEMA_VERSION
    b = Store(db)  # a second process opening the same file
    assert b.migrate() == SCHEMA_VERSION
    assert b.conn.execute("SELECT COUNT(*) FROM heartbeat").fetchone()[0] == 1
    a.close()
    b.close()


def test_decisions_unique_per_book_and_bar(store):
    store.conn.execute(
        "INSERT INTO books (book_id, title, symbol, timeframe, strategy, strategy_version, strategy_source, "
        "params_json, starting_equity, cash, created_at, updated_at) VALUES "
        "('b', 't', 'BTCUSDT', '1h', 's', '1', 'builtin', '{}', 1, 1, 'n', 'n')"
    )
    sql = (
        "INSERT INTO decisions (book_id, bar_open_ts, decided_at, target, prev_target, explanation, strategy_version) "
        "VALUES ('b', '2021-01-01T00:00:00Z', 'n', 1, 0, 'x', '1')"
    )
    store.conn.execute(sql)
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute(sql)


def test_instance_lock_is_exclusive(tmp_path):
    first, second = InstanceLock(tmp_path / "r.lock"), InstanceLock(tmp_path / "r.lock")
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()


def test_backoff_grows_and_caps():
    rng = random.Random(1)
    assert 27 <= next_delay(30, 0, 0.1, rng) <= 33
    assert 54 <= next_delay(30, 1, 0.1, rng) <= 66
    assert next_delay(30, 50, 0.1, rng) <= 660
    assert exchange_trouble({"status": "exchange_unreachable"})
    assert exchange_trouble({"status": "degraded", "books": [{"decision": "data_error"}]})
    assert not exchange_trouble({"status": "degraded", "books": [{"decision": "strategy_error"}]})
