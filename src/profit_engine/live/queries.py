"""Read-only views of the live store for the dashboard and ``signal-desk-live status``.

Every connection here sets ``PRAGMA query_only``: these functions cannot write.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from profit_engine.live.strategy import DEMO_LABEL, PROMOTED_LABEL


def connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=5, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _rows(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def _age_s(ts: str | None, now: pd.Timestamp) -> float | None:
    return None if not ts else round((now - pd.Timestamp(ts)).total_seconds(), 1)


def buy_and_hold(conn: sqlite3.Connection, book: dict, price: float | None) -> dict:
    """What the starting cash would be worth had it bought the coin at the book's first marked
    price and held it (before fees): the reference the dashboard compares each book with."""
    first = conn.execute(
        "SELECT ts, price FROM equity_marks WHERE book_id=? ORDER BY ts LIMIT 1", (book["book_id"],)
    ).fetchone()
    if first is None or not price or not first["price"]:
        return {"bh_start_ts": None, "bh_start_price": None, "bh_equity": None, "bh_return_pct": None}
    value = book["starting_equity"] * price / first["price"]
    pct = (value / book["starting_equity"] - 1) * 100 if book["starting_equity"] else 0.0
    return {"bh_start_ts": first["ts"], "bh_start_price": first["price"], "bh_equity": value, "bh_return_pct": pct}


def book_summary(conn: sqlite3.Connection, book: dict) -> dict:
    bid = book["book_id"]
    mark = conn.execute("SELECT * FROM equity_marks WHERE book_id=? ORDER BY ts DESC LIMIT 1", (bid,)).fetchone()
    decision = conn.execute(
        "SELECT bar_open_ts, decided_at, target, prev_target, explanation, fill_bar_ts FROM decisions "
        "WHERE book_id=? ORDER BY bar_open_ts DESC LIMIT 1",
        (bid,),
    ).fetchone()
    price = mark["price"] if mark else None
    equity = book["cash"] + book["units"] * price if price else book["cash"]
    weight = book["units"] * price / equity if price and equity else 0.0
    return {
        "book_id": bid,
        "title": book["title"],
        "symbol": book["symbol"],
        "coin": book["symbol"].removesuffix("USDT"),
        "timeframe": book["timeframe"],
        "strategy": book["strategy"],
        "strategy_version": book["strategy_version"],
        "strategy_source": book["strategy_source"],
        "params": json.loads(book["params_json"] or "{}"),
        "artifact_hash": book["artifact_hash"],
        "promoted": bool(book["promoted"]),
        "label": PROMOTED_LABEL if book["promoted"] else (book.get("status_label") or DEMO_LABEL),
        "family": book.get("family") or book["strategy"],
        "enabled": bool(book["enabled"]),
        "starting_equity": book["starting_equity"],
        "equity": equity,
        "return_pct": (equity / book["starting_equity"] - 1) * 100 if book["starting_equity"] else 0.0,
        "units": book["units"],
        "cash": book["cash"],
        "weight": weight,
        "target": book["target"],
        "last_price": price,
        "last_mark_ts": book["last_mark_ts"],
        "last_bar_ts": book["last_bar_ts"],
        "paused_until": book["paused_until"],
        "last_decision": dict(decision) if decision else None,
        **buy_and_hold(conn, book, price),
    }


def heartbeat(conn: sqlite3.Connection, now: pd.Timestamp | None = None) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    out = {}
    for row in _rows(conn, "SELECT * FROM heartbeat"):
        out[row["component"]] = {
            "ts": row["ts"],
            "status": row["status"],
            "age_s": _age_s(row["ts"], now),
            "detail": json.loads(row["detail_json"] or "{}"),
        }
    return out


def overview(path: Path) -> dict:
    with closing(connect_ro(path)) as conn:
        # Grouped by strategy for the dashboard (DESC puts "Trend (vol-targeted)" before "LightGBM 4h").
        books = _rows(conn, "SELECT * FROM books ORDER BY family DESC, book_id")
        return {
            "paper": True,
            "server_time": pd.Timestamp.now(tz="UTC").isoformat(),
            "books": [book_summary(conn, b) for b in books],
            "heartbeat": heartbeat(conn),
        }


def book_exists(path: Path, book_id: str) -> bool:
    with closing(connect_ro(path)) as conn:
        return conn.execute("SELECT 1 FROM books WHERE book_id=?", (book_id,)).fetchone() is not None


def equity_curve(path: Path, book_id: str, limit: int = 3000) -> list[dict]:
    sql = "SELECT ts, equity, price FROM equity_marks WHERE book_id=? ORDER BY ts DESC LIMIT ?"
    with closing(connect_ro(path)) as conn:
        return list(reversed(_rows(conn, sql, (book_id, limit))))


def fills(path: Path, book_id: str, limit: int = 50) -> list[dict]:
    sql = (
        "SELECT fill_bar_ts, decision_bar_ts, reason, side, price, units_delta, units_after, target, cost, "
        "equity_before, equity_after, latency_s FROM fills WHERE book_id=? ORDER BY fill_bar_ts DESC, id DESC LIMIT ?"
    )
    with closing(connect_ro(path)) as conn:
        return _rows(conn, sql, (book_id, limit))


def decisions(path: Path, book_id: str, limit: int = 50) -> list[dict]:
    sql = (
        "SELECT bar_open_ts, decided_at, target, prev_target, explanation, close_price, fill_bar_ts, skipped_bars "
        "FROM decisions WHERE book_id=? ORDER BY bar_open_ts DESC LIMIT ?"
    )
    with closing(connect_ro(path)) as conn:
        return _rows(conn, sql, (book_id, limit))


def events(path: Path, book_id: str | None = None, limit: int = 100) -> list[dict]:
    """Plain-language events (technical detail stays in the logs)."""
    sql = "SELECT ts, book_id, kind, message FROM events"
    args: tuple = ()
    if book_id:
        sql, args = sql + " WHERE book_id IN (?, '')", (book_id,)
    with closing(connect_ro(path)) as conn:
        return _rows(conn, sql + " ORDER BY ts DESC, id DESC LIMIT ?", (*args, limit))
