"""Database backend selection: Postgres (DATABASE_URL) or local SQLite fallback."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE = ROOT / "data" / "paper_ledger.sqlite"


def database_url() -> str | None:
    url = (os.environ.get("DATABASE_URL") or "").strip()
    return url or None


def using_postgres() -> bool:
    return bool(database_url())
