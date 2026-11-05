"""Local SQLite store for the live paper books (never Postgres, never DATABASE_URL).

Exactly-once per bar: :meth:`Store.commit_bar` claims ``(book_id, bar_open_ts)`` with
``INSERT ... ON CONFLICT DO NOTHING`` inside ONE ``BEGIN IMMEDIATE`` transaction that also
writes the fill and the new book position. A duplicate tick, a second process, or a crash
and restart can therefore never trade the same bar twice: either the whole bar (claim,
fill, position, events) is committed, or none of it is and the bar can be retried.

WAL mode lets the dashboard read while the runner writes; ``busy_timeout`` makes a second
writer wait instead of failing. Schema changes go through :data:`MIGRATIONS`, keyed by
``PRAGMA user_version``; :meth:`Store.migrate` is idempotent and safe under concurrency.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from profit_engine.live.accounting import FillPlan, equity, plan_fill
from profit_engine.live.strategy import Decision, paused_decision

BUSY_TIMEOUT_MS = 30_000

SCHEMA_V1 = (
    """CREATE TABLE IF NOT EXISTS books (
        book_id TEXT PRIMARY KEY, title TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
        strategy TEXT NOT NULL, strategy_version TEXT NOT NULL, strategy_source TEXT NOT NULL,
        params_json TEXT NOT NULL, artifact_hash TEXT NOT NULL DEFAULT '', promoted INTEGER NOT NULL DEFAULT 0,
        enabled INTEGER NOT NULL DEFAULT 1, starting_equity REAL NOT NULL, cash REAL NOT NULL,
        units REAL NOT NULL DEFAULT 0, target REAL NOT NULL DEFAULT 0, last_bar_ts TEXT, paused_until TEXT,
        day TEXT, day_start_equity REAL, last_mark_ts TEXT, last_mark_price REAL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY, book_id TEXT NOT NULL REFERENCES books(book_id), bar_open_ts TEXT NOT NULL,
        decided_at TEXT NOT NULL, target REAL NOT NULL, prev_target REAL NOT NULL, explanation TEXT NOT NULL,
        details_json TEXT NOT NULL DEFAULT '{}', strategy_version TEXT NOT NULL, close_price REAL,
        fill_bar_ts TEXT, skipped_bars INTEGER NOT NULL DEFAULT 0, UNIQUE(book_id, bar_open_ts))""",
    """CREATE TABLE IF NOT EXISTS fills (
        id INTEGER PRIMARY KEY, book_id TEXT NOT NULL REFERENCES books(book_id), fill_key TEXT NOT NULL,
        reason TEXT NOT NULL, decision_bar_ts TEXT, fill_bar_ts TEXT NOT NULL, side TEXT NOT NULL,
        price REAL NOT NULL, units_delta REAL NOT NULL, units_after REAL NOT NULL, target REAL NOT NULL,
        weight_before REAL NOT NULL, cost REAL NOT NULL, equity_before REAL NOT NULL, equity_after REAL NOT NULL,
        latency_s REAL, created_at TEXT NOT NULL, UNIQUE(book_id, fill_key))""",
    """CREATE TABLE IF NOT EXISTS equity_marks (
        book_id TEXT NOT NULL REFERENCES books(book_id), ts TEXT NOT NULL, price REAL NOT NULL,
        equity REAL NOT NULL, units REAL NOT NULL, cash REAL NOT NULL, PRIMARY KEY(book_id, ts))""",
    """CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY, book_id TEXT NOT NULL DEFAULT '', ts TEXT NOT NULL, kind TEXT NOT NULL,
        message TEXT NOT NULL, data_json TEXT NOT NULL DEFAULT '{}', dedupe_key TEXT NOT NULL,
        UNIQUE(book_id, kind, dedupe_key))""",
    "CREATE INDEX IF NOT EXISTS events_by_ts ON events(ts)",
    """CREATE TABLE IF NOT EXISTS heartbeat (
        component TEXT PRIMARY KEY, ts TEXT NOT NULL, pid INTEGER, status TEXT NOT NULL,
        detail_json TEXT NOT NULL DEFAULT '{}')""",
    """CREATE TABLE IF NOT EXISTS notifications (
        id INTEGER PRIMARY KEY, book_id TEXT NOT NULL, created_at TEXT NOT NULL, text TEXT NOT NULL, sent_at TEXT)""",
)
# v2: the dashboard's status label and strategy group for each book.
SCHEMA_V2 = (
    "ALTER TABLE books ADD COLUMN status_label TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE books ADD COLUMN family TEXT NOT NULL DEFAULT ''",
)
MIGRATIONS: dict[int, tuple[str, ...]] = {1: SCHEMA_V1, 2: SCHEMA_V2}
SCHEMA_VERSION = max(MIGRATIONS)
TABLES = frozenset({"books", "decisions", "fills", "equity_marks", "events", "heartbeat", "notifications"})


class StoreError(RuntimeError):
    """The database file is not one this version can use (a newer schema, or not ours)."""


def iso(ts: object) -> str:
    """UTC ISO-8601 with a Z suffix (lexical order == time order)."""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(value: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(value).tz_convert("UTC") if value else None


@dataclass(frozen=True)
class EventSpec:
    kind: str
    message: str
    data: dict = field(default_factory=dict)
    dedupe_key: str | None = None


@dataclass(frozen=True)
class BarCommit:
    """Everything the runner learned about one decision bar, committed atomically."""

    bar_ts: pd.Timestamp
    fill_ts: pd.Timestamp
    decision: Decision
    price: float
    close_price: float
    cost_bps: float
    now: pd.Timestamp
    strategy_version: str
    skipped_bars: int = 0
    events: tuple[EventSpec, ...] = ()


@dataclass(frozen=True)
class CommitResult:
    claimed: bool
    fill: FillPlan | None = None


class _AlreadyDone(Exception):
    """Internal: roll back a claim that turned out to be stale."""


class BookConflict(ValueError):
    """books.json now gives an existing book a different coin or timeframe than its history."""

    def __init__(self, book_id: str, stored: tuple[str, str], wanted: tuple[str, str]) -> None:
        super().__init__(f"book {book_id!r} was {stored}, settings now say {wanted}")
        self.book_id, self.stored, self.wanted = book_id, stored, wanted


class Store:
    def __init__(self, path: Path | str) -> None:
        if "://" in str(path):  # before Path(), which would fold "//" into "/"
            raise ValueError("the live store is local SQLite only; got a URL")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """One write transaction, taken with the write lock up front (BEGIN IMMEDIATE)."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    def user_version(self) -> int:
        return int(self.conn.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self) -> int:
        """Apply pending migrations in order; re-checks the version under the write lock.

        Raises :class:`StoreError` for a schema newer than this code knows (a downgrade) or a
        file that claims a version but lacks the tables: writing to either would fail later
        with nothing on the dashboard to say why."""
        version = self.user_version()
        if version > SCHEMA_VERSION:
            raise StoreError(
                f"The live database {self.path} was written by a newer version of Signal Desk Live "
                f"(format {version}; this copy understands up to {SCHEMA_VERSION}). Update this checkout, or set "
                "SIGNAL_DESK_LIVE_DB to a different file."
            )
        if version >= 1:
            names = {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not names >= TABLES:
                raise StoreError(
                    f"{self.path} is not a Signal Desk Live database (it is marked with a version but lacks the "
                    "expected tables). Set SIGNAL_DESK_LIVE_DB to a different file."
                )
        if version == SCHEMA_VERSION:
            return version
        with self.tx() as c:
            current = int(c.execute("PRAGMA user_version").fetchone()[0])
            for version in sorted(v for v in MIGRATIONS if v > current):
                for statement in MIGRATIONS[version]:
                    c.execute(statement)
                c.execute(f"PRAGMA user_version={int(version)}")
        return self.user_version()

    # -- books ---------------------------------------------------------------------------

    def upsert_book(self, spec, now: pd.Timestamp) -> None:
        """Create the book on first sight (cash = starting equity); refresh its metadata after.

        Raises :class:`BookConflict` (and changes nothing) if the book already exists with a
        different symbol or timeframe: its units, marks and last bar belong to the old market.
        """
        s, meta = spec.strategy, spec.strategy.meta
        row = {
            "book_id": spec.book_id, "title": spec.title, "symbol": s.symbol, "timeframe": s.timeframe,
            "strategy": s.name, "strategy_version": meta.version, "strategy_source": meta.source,
            "params_json": json.dumps(meta.params, sort_keys=True, default=str), "artifact_hash": meta.artifact_hash,
            "promoted": int(meta.promoted), "enabled": int(spec.enabled), "starting_equity": spec.starting_equity,
            "cash": spec.starting_equity, "created_at": iso(now), "updated_at": iso(now),
            "status_label": getattr(meta, "status", ""), "family": getattr(meta, "family", ""),
        }  # fmt: skip
        cols = ", ".join(row)
        marks = ", ".join(f":{k}" for k in row)
        refresh = ("title", "strategy", "strategy_version", "strategy_source", "params_json", "artifact_hash",
                   "promoted", "enabled", "status_label", "family")  # fmt: skip
        update = ", ".join(f"{k}=excluded.{k}" for k in refresh)
        with self.tx() as c:
            have = c.execute("SELECT symbol, timeframe FROM books WHERE book_id=?", (spec.book_id,)).fetchone()
            if have is not None and tuple(have) != (s.symbol, s.timeframe):
                raise BookConflict(spec.book_id, tuple(have), (s.symbol, s.timeframe))
            c.execute(f"INSERT INTO books ({cols}) VALUES ({marks}) ON CONFLICT(book_id) DO UPDATE SET {update}", row)

    def book(self, book_id: str, conn: sqlite3.Connection | None = None) -> dict | None:
        row = (conn or self.conn).execute("SELECT * FROM books WHERE book_id=?", (book_id,)).fetchone()
        return dict(row) if row else None

    def strategy_target(self, book_id: str) -> float:
        """The strategy's own last target: what a stateful strategy continues from. A decision
        overridden by the daily-loss pause recorded the strategy's wish as ``strategy_target``."""
        row = self.conn.execute(
            "SELECT target, details_json FROM decisions WHERE book_id=? ORDER BY bar_open_ts DESC LIMIT 1", (book_id,)
        ).fetchone()
        if row is None:
            return 0.0
        details = json.loads(row["details_json"] or "{}")
        return float(details.get("strategy_target", row["target"]))

    # -- the exactly-once bar commit -------------------------------------------------------

    def commit_bar(self, book_id: str, bc: BarCommit) -> CommitResult:
        """Claim the bar, fill, move the position and log its events - all or nothing."""
        try:
            with self.tx() as c:
                book = self.book(book_id, c)
                if book is None:
                    raise KeyError(book_id)
                last = parse_ts(book["last_bar_ts"])
                if last is not None and bc.bar_ts <= last:
                    raise _AlreadyDone
                decision = bc.decision
                paused = parse_ts(book["paused_until"])  # read under the write lock: never a stale pause
                if paused is not None and bc.now < paused:
                    decision = paused_decision(paused, decision)
                claimed = c.execute(
                    "INSERT INTO decisions (book_id, bar_open_ts, decided_at, target, prev_target, explanation, "
                    "details_json, strategy_version, close_price, fill_bar_ts, skipped_bars) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(book_id, bar_open_ts) DO NOTHING",
                    (book_id, iso(bc.bar_ts), iso(bc.now), decision.target, book["target"],
                     decision.explanation, json.dumps(decision.details, default=str), bc.strategy_version,
                     bc.close_price, iso(bc.fill_ts), bc.skipped_bars),
                ).rowcount  # fmt: skip
                if not claimed:
                    raise _AlreadyDone
                plan = plan_fill(book["cash"], book["units"], book["target"], decision.target, bc.price, bc.cost_bps)
                if plan is not None:
                    latency = (bc.now - bc.fill_ts).total_seconds()
                    key = f"bar:{iso(bc.bar_ts)}"
                    self._insert_fill(c, book_id, key, "signal", plan, (bc.bar_ts, bc.fill_ts, latency, bc.now))
                    self._set_position(c, book_id, plan, bc.now)
                c.execute(
                    "UPDATE books SET last_bar_ts=?, updated_at=? WHERE book_id=?",
                    (iso(bc.bar_ts), iso(bc.now), book_id),
                )
                for ev in bc.events:
                    self._insert_event(c, book_id, ev, bc.now)
        except _AlreadyDone:
            return CommitResult(claimed=False)
        return CommitResult(claimed=True, fill=plan)

    def _insert_fill(self, c: sqlite3.Connection, book_id: str, key: str, reason: str, plan: FillPlan, when) -> None:
        decision_ts, fill_ts, latency, now = when
        c.execute(
            "INSERT INTO fills (book_id, fill_key, reason, decision_bar_ts, fill_bar_ts, side, price, units_delta, "
            "units_after, target, weight_before, cost, equity_before, equity_after, latency_s, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (book_id, key, reason, iso(decision_ts) if decision_ts is not None else None, iso(fill_ts), plan.side,
             plan.price, plan.units_delta, plan.units_after, plan.target, plan.weight_before, plan.cost,
             plan.equity_before, plan.equity_after, latency, iso(now)),
        )  # fmt: skip

    @staticmethod
    def _set_position(c: sqlite3.Connection, book_id: str, plan: FillPlan, now: pd.Timestamp) -> None:
        c.execute(
            "UPDATE books SET cash=?, units=?, target=?, updated_at=? WHERE book_id=?",
            (plan.cash_after, plan.units_after, plan.target, iso(now), book_id),
        )

    # -- marks, kill switch, pause -----------------------------------------------------------

    def record_mark(
        self, book_id: str, ts: pd.Timestamp, price: float, now: pd.Timestamp, midnight_price: float | None = None
    ) -> dict:
        """Mark the book at ``price`` (a bar close at ``ts``). On the first mark of a UTC day the
        day's starting equity is set to the equity at 00:00 UTC (see :meth:`_midnight_equity`),
        not to this mark, so losses made while the Mac slept still count against the daily limit.
        Returns equity, day-start equity and the book row."""
        with self.tx() as c:
            book = self.book(book_id, c)
            eq = equity(book["cash"], book["units"], price)
            day = pd.Timestamp(ts).strftime("%Y-%m-%d")
            if book["day"] != day:
                start = eq if book["day"] is None else self._midnight_equity(c, book, ts, midnight_price)
                c.execute("UPDATE books SET day=?, day_start_equity=? WHERE book_id=?", (day, start, book_id))
                book["day_start_equity"] = start
            c.execute(
                "INSERT INTO equity_marks (book_id, ts, price, equity, units, cash) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(book_id, ts) DO NOTHING",
                (book_id, iso(ts), price, eq, book["units"], book["cash"]),
            )
            c.execute("UPDATE books SET last_mark_ts=?, last_mark_price=? WHERE book_id=?", (iso(ts), price, book_id))
        return {"equity": eq, "day_start_equity": book["day_start_equity"], "book": book}

    @staticmethod
    def _midnight_equity(c: sqlite3.Connection, book: dict, ts: pd.Timestamp, price: float | None) -> float:
        """Equity at 00:00 UTC of ``ts``'s day: the position held then, at the 00:00 price.

        The position is the book's current one unless a fill landed at or after midnight, in
        which case it is the position just before the first such fill. Without a 00:00 price
        (a data gap), the last mark before midnight stands in for it."""
        midnight = iso(pd.Timestamp(ts).floor("D"))
        first = c.execute(
            "SELECT price, units_delta, units_after, equity_before FROM fills WHERE book_id=? AND fill_bar_ts>=? "
            "ORDER BY fill_bar_ts, id LIMIT 1",
            (book["book_id"], midnight),
        ).fetchone()
        units, cash = book["units"], book["cash"]
        if first is not None:
            units = first["units_after"] - first["units_delta"]
            cash = first["equity_before"] - units * first["price"]
        if price is not None:
            return equity(cash, units, price)
        last = c.execute(
            "SELECT equity FROM equity_marks WHERE book_id=? AND ts<=? ORDER BY ts DESC LIMIT 1",
            (book["book_id"], midnight),
        ).fetchone()
        return float(last["equity"]) if last is not None else equity(cash, units, book["last_mark_price"] or 0.0)

    def trigger_kill_switch(self, book_id: str, kill: dict) -> CommitResult:
        """Go flat at ``kill['price']`` and pause until ``kill['paused_until']``, at most once
        per pause: the UPDATE that sets the pause is the claim."""
        now = kill["now"]
        with self.tx() as c:
            claimed = c.execute(
                "UPDATE books SET paused_until=? WHERE book_id=? AND (paused_until IS NULL OR paused_until<=?)",
                (iso(kill["paused_until"]), book_id, iso(now)),
            ).rowcount
            if not claimed:
                return CommitResult(claimed=False)
            book = self.book(book_id, c)
            plan = plan_fill(book["cash"], book["units"], book["target"], 0.0, kill["price"], kill["cost_bps"])
            if plan is not None:
                latency = (now - kill["fill_ts"]).total_seconds()
                key = f"kill:{iso(kill['fill_ts'])}"
                self._insert_fill(c, book_id, key, "kill_switch", plan, (None, kill["fill_ts"], latency, now))
                self._set_position(c, book_id, plan, now)
            self._insert_event(c, book_id, kill["event"], now)
        return CommitResult(claimed=True, fill=plan)

    def resume_if_due(self, book_id: str, now: pd.Timestamp, event: EventSpec) -> bool:
        with self.tx() as c:
            done = c.execute(
                "UPDATE books SET paused_until=NULL WHERE book_id=? AND paused_until IS NOT NULL AND paused_until<=?",
                (book_id, iso(now)),
            ).rowcount
            if done:
                self._insert_event(c, book_id, event, now)
        return bool(done)

    # -- events, heartbeat, notifications ------------------------------------------------------

    def _insert_event(self, c: sqlite3.Connection, book_id: str | None, ev: EventSpec, now: pd.Timestamp) -> bool:
        key = ev.dedupe_key or f"{iso(now)}:{uuid.uuid4().hex[:10]}"
        return bool(
            c.execute(
                "INSERT INTO events (book_id, ts, kind, message, data_json, dedupe_key) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(book_id, kind, dedupe_key) DO NOTHING",
                (book_id or "", iso(now), ev.kind, ev.message, json.dumps(ev.data, default=str), key),
            ).rowcount
        )

    def add_event(self, book_id: str | None, ev: EventSpec, now: pd.Timestamp) -> bool:
        """Log an event; with a ``dedupe_key`` the same (book, kind, key) is logged once."""
        with self.tx() as c:
            return self._insert_event(c, book_id, ev, now)

    def beat(self, component: str, status: str, now: pd.Timestamp, detail: dict | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO heartbeat (component, ts, pid, status, detail_json) VALUES (?,?,?,?,?) "
                "ON CONFLICT(component) DO UPDATE SET ts=excluded.ts, pid=excluded.pid, status=excluded.status, "
                "detail_json=excluded.detail_json",
                (component, iso(now), os.getpid(), status, json.dumps(detail or {}, default=str)),
            )

    def note_next_check(self, component: str, seconds: float) -> None:
        """Record when the component plans its next check, so the dashboard can tell a runner
        that is backing off on schedule from one that stopped."""
        with self.tx() as c:
            c.execute(
                "UPDATE heartbeat SET detail_json=json_set(detail_json, '$.next_check_s', ?) WHERE component=?",
                (round(float(seconds), 1), component),
            )

    def queue_notification(self, book_id: str, text: str, now: pd.Timestamp) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO notifications (book_id, created_at, text) VALUES (?,?,?)", (book_id, iso(now), text))

    def pending_notifications(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM notifications WHERE sent_at IS NULL ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def last_notification_sent(self, book_id: str) -> pd.Timestamp | None:
        row = self.conn.execute("SELECT MAX(sent_at) FROM notifications WHERE book_id=?", (book_id,)).fetchone()
        return parse_ts(row[0])

    def mark_notifications_sent(self, ids: list[int], now: pd.Timestamp) -> None:
        with self.tx() as c:
            c.executemany("UPDATE notifications SET sent_at=? WHERE id=?", [(iso(now), i) for i in ids])
