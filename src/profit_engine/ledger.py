"""Append-only SQLite paper ledger (fake money, auditable)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "paper_ledger.sqlite"


class PaperLedger:
    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._conn() as conn:
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
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs(run_id, strategy, config_json, started_at, initial_cash) VALUES (?,?,?,?,?)",
                (
                    run_id,
                    strategy,
                    json.dumps(config, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                    initial_cash,
                ),
            )
            conn.execute(
                "INSERT OR REPLACE INTO positions(run_id, cash, asset, shares) VALUES (?,?,?,?)",
                (run_id, initial_cash, None, 0.0),
            )
            conn.execute(
                "INSERT INTO equity(run_id, ts, cash, asset, shares, mark_price, equity, source) VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    initial_cash,
                    None,
                    0.0,
                    None,
                    initial_cash,
                    "init",
                ),
            )

    def get_position(self, run_id: str) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM positions WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            return dict(row)

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
        ts = ts or datetime.now(timezone.utc).isoformat()
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

        with self._conn() as conn:
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
        ts = ts or datetime.now(timezone.utc).isoformat()
        pos = self.get_position(run_id)
        cash = float(pos["cash"])
        shares = float(pos["shares"])
        asset = pos["asset"]
        equity = cash + (shares * mark_price if mark_price is not None and shares else 0.0)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO equity(run_id, ts, cash, asset, shares, mark_price, equity, source) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, ts, cash, asset, shares, mark_price, equity, source),
            )
        return equity