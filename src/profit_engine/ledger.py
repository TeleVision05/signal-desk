"""Append-only paper ledger (fake money). Postgres via DATABASE_URL, else SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import DEFAULT_SQLITE, database_url

_PG_SCHEMA_READY = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperLedger:
    def __init__(self, db_path: Path | str | None = None):
        self._url = database_url()
        self.db_path = Path(db_path) if db_path else DEFAULT_SQLITE
        if not self._url:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    @property
    def backend(self) -> str:
        return "postgres" if self._url else "sqlite"

    def _pg_conn(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self._url, row_factory=dict_row)

    def _sqlite_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        global _PG_SCHEMA_READY
        if self._url:
            if _PG_SCHEMA_READY:
                return
            # Schema is applied via Supabase migrations; just probe connectivity.
            with self._pg_conn() as conn:
                conn.execute("SELECT 1")
            _PG_SCHEMA_READY = True
            return
        with self._sqlite_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY,
                  strategy TEXT NOT NULL,
                  config_json TEXT NOT NULL,
                  started_at TEXT NOT NULL,
                  initial_cash REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS fills (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT NOT NULL,
                  ts TEXT NOT NULL,
                  side TEXT NOT NULL,
                  asset TEXT NOT NULL,
                  shares REAL NOT NULL,
                  price REAL NOT NULL,
                  cost REAL NOT NULL,
                  source TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS equity (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT NOT NULL,
                  ts TEXT NOT NULL,
                  cash REAL NOT NULL,
                  asset TEXT,
                  shares REAL NOT NULL,
                  mark_price REAL,
                  equity REAL NOT NULL,
                  source TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS positions (
                  run_id TEXT PRIMARY KEY,
                  cash REAL NOT NULL,
                  asset TEXT,
                  shares REAL NOT NULL
                );
                """
            )

    def start_run(
        self,
        run_id: str,
        strategy: str,
        config: dict,
        initial_cash: float = 100_000.0,
    ) -> None:
        started = _now()
        cfg = json.dumps(config, sort_keys=True)
        if self._url:
            with self._pg_conn() as conn:
                with conn.transaction():
                    conn.execute(
                        """
                        INSERT INTO paper_runs(run_id, strategy, config_json, started_at, initial_cash)
                        VALUES (%s, %s, %s::jsonb, %s::timestamptz, %s)
                        ON CONFLICT (run_id) DO UPDATE SET
                          strategy = EXCLUDED.strategy,
                          config_json = EXCLUDED.config_json,
                          started_at = EXCLUDED.started_at,
                          initial_cash = EXCLUDED.initial_cash
                        """,
                        (run_id, strategy, cfg, started, initial_cash),
                    )
                    conn.execute(
                        """
                        INSERT INTO paper_positions(run_id, cash, asset, shares)
                        VALUES (%s, %s, NULL, 0.0)
                        ON CONFLICT (run_id) DO UPDATE SET cash = EXCLUDED.cash, asset = NULL, shares = 0.0
                        """,
                        (run_id, initial_cash),
                    )
                    conn.execute(
                        """
                        INSERT INTO paper_equity(run_id, ts, cash, asset, shares, mark_price, equity, source)
                        VALUES (%s, %s::timestamptz, %s, NULL, 0.0, NULL, %s, 'init')
                        """,
                        (run_id, started, initial_cash, initial_cash),
                    )
            return
        with self._sqlite_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs(run_id, strategy, config_json, started_at, initial_cash) VALUES (?,?,?,?,?)",
                (run_id, strategy, cfg, started, initial_cash),
            )
            conn.execute(
                "INSERT OR REPLACE INTO positions(run_id, cash, asset, shares) VALUES (?,?,?,?)",
                (run_id, initial_cash, None, 0.0),
            )
            conn.execute(
                "INSERT INTO equity(run_id, ts, cash, asset, shares, mark_price, equity, source) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, started, initial_cash, None, 0.0, None, initial_cash, "init"),
            )

    def get_position(self, run_id: str) -> dict:
        if self._url:
            with self._pg_conn() as conn:
                row = conn.execute(
                    "SELECT run_id, cash, asset, shares FROM paper_positions WHERE run_id = %s",
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(run_id)
                return dict(row)
        with self._sqlite_conn() as conn:
            row = conn.execute("SELECT * FROM positions WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            return dict(row)

    def last_buy_price(self, run_id: str) -> float | None:
        if self._url:
            with self._pg_conn() as conn:
                row = conn.execute(
                    """
                    SELECT price FROM paper_fills
                    WHERE run_id = %s AND side = 'BUY'
                    ORDER BY id DESC LIMIT 1
                    """,
                    (run_id,),
                ).fetchone()
                return float(row["price"]) if row else None
        with self._sqlite_conn() as conn:
            row = conn.execute(
                "SELECT price FROM fills WHERE run_id=? AND side='BUY' ORDER BY id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            return float(row["price"]) if row else None

    def apply_fill(
        self,
        run_id: str,
        *,
        side: str,
        asset: str,
        shares: float,
        price: float,
        cost: float,
        source: str,
        ts: str | None = None,
    ) -> None:
        ts = ts or _now()
        pos = self.get_position(run_id)
        cash = float(pos["cash"])
        cur_asset = pos["asset"]
        cur_shares = float(pos["shares"])

        if side == "SELL":
            cash += shares * price - cost
            cur_shares = 0.0
            cur_asset = None
        elif side == "BUY":
            cash -= shares * price + cost
            cur_asset = asset
            cur_shares = shares
        else:
            raise ValueError(side)

        if self._url:
            with self._pg_conn() as conn:
                with conn.transaction():
                    conn.execute(
                        """
                        INSERT INTO paper_fills(run_id, ts, side, asset, shares, price, cost, source)
                        VALUES (%s, %s::timestamptz, %s, %s, %s, %s, %s, %s)
                        """,
                        (run_id, ts, side, asset, shares, price, cost, source),
                    )
                    conn.execute(
                        "UPDATE paper_positions SET cash=%s, asset=%s, shares=%s WHERE run_id=%s",
                        (cash, cur_asset, cur_shares, run_id),
                    )
            return
        with self._sqlite_conn() as conn:
            conn.execute(
                "INSERT INTO fills(run_id, ts, side, asset, shares, price, cost, source) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, ts, side, asset, shares, price, cost, source),
            )
            conn.execute(
                "UPDATE positions SET cash=?, asset=?, shares=? WHERE run_id=?",
                (cash, cur_asset, cur_shares, run_id),
            )

    def mark(
        self,
        run_id: str,
        *,
        mark_price: float | None,
        source: str,
        ts: str | None = None,
    ) -> float:
        ts = ts or _now()
        pos = self.get_position(run_id)
        cash = float(pos["cash"])
        shares = float(pos["shares"])
        asset = pos["asset"]
        equity = cash + (shares * mark_price if mark_price is not None and shares else 0.0)
        if self._url:
            with self._pg_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO paper_equity(run_id, ts, cash, asset, shares, mark_price, equity, source)
                    VALUES (%s, %s::timestamptz, %s, %s, %s, %s, %s, %s)
                    """,
                    (run_id, ts, cash, asset, shares, mark_price, equity, source),
                )
                conn.commit()
            return equity
        with self._sqlite_conn() as conn:
            conn.execute(
                "INSERT INTO equity(run_id, ts, cash, asset, shares, mark_price, equity, source) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, ts, cash, asset, shares, mark_price, equity, source),
            )
        return equity

    def snapshot(self, run_id: str) -> dict[str, Any]:
        """Fills / equity / position for the dashboard."""
        if self._url:
            with self._pg_conn() as conn:
                run = conn.execute(
                    "SELECT run_id, strategy, config_json::text AS config_json, started_at::text AS started_at, initial_cash FROM paper_runs WHERE run_id=%s",
                    (run_id,),
                ).fetchone()
                pos = conn.execute(
                    "SELECT run_id, cash, asset, shares FROM paper_positions WHERE run_id=%s",
                    (run_id,),
                ).fetchone()
                fills = conn.execute(
                    """
                    SELECT id, run_id, ts::text AS ts, side, asset, shares, price, cost, source
                    FROM paper_fills WHERE run_id=%s ORDER BY id ASC
                    """,
                    (run_id,),
                ).fetchall()
                equity = conn.execute(
                    """
                    SELECT id, run_id, ts::text AS ts, cash, asset, shares, mark_price, equity, source
                    FROM paper_equity WHERE run_id=%s ORDER BY id ASC
                    """,
                    (run_id,),
                ).fetchall()
            return {
                "run": dict(run) if run else None,
                "position": dict(pos) if pos else None,
                "fills": [dict(r) for r in fills],
                "equity": [dict(r) for r in equity],
            }
        if not self.db_path.exists():
            return {"fills": [], "equity": [], "position": None, "run": None}
        with self._sqlite_conn() as conn:
            run = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            pos = conn.execute("SELECT * FROM positions WHERE run_id=?", (run_id,)).fetchone()
            fills = conn.execute(
                "SELECT * FROM fills WHERE run_id=? ORDER BY id ASC", (run_id,)
            ).fetchall()
            equity = conn.execute(
                "SELECT * FROM equity WHERE run_id=? ORDER BY id ASC", (run_id,)
            ).fetchall()
        return {
            "run": dict(run) if run else None,
            "position": dict(pos) if pos else None,
            "fills": [dict(r) for r in fills],
            "equity": [dict(r) for r in equity],
        }
