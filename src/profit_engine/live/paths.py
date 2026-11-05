"""Where Signal Desk Live keeps its files.

Everything is LOCAL. The live store is a SQLite file and nothing here ever reads
``DATABASE_URL`` (that variable points the Render app at Supabase Postgres; the local
runner must never write there, even when it is set in the shell).

| what | default | override |
|---|---|---|
| app folder | ``~/Library/Application Support/SignalDesk`` | ``SIGNAL_DESK_HOME`` |
| live database | ``<app folder>/live.sqlite`` | ``SIGNAL_DESK_LIVE_DB`` |
| books config | ``<app folder>/books.json``, else ``config/live_books.json`` | ``SIGNAL_DESK_BOOKS`` |
| model artifacts | ``<app folder>/models`` | ``SIGNAL_DESK_MODELS_DIR`` |
| 5m history for training | ``<app folder>/market_data`` | ``SIGNAL_DESK_DATA_DIR`` |
| logs | ``~/Library/Logs/SignalDesk`` | ``SIGNAL_DESK_LOGS`` |
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BOOKS = REPO_ROOT / "config" / "live_books.json"
APP_NAME = "SignalDesk"


def _env_path(name: str) -> Path | None:
    value = (os.environ.get(name) or "").strip()
    return Path(value).expanduser() if value else None


def home() -> Path:
    return Path(os.environ.get("HOME") or Path.home())


def app_dir() -> Path:
    return _env_path("SIGNAL_DESK_HOME") or home() / "Library" / "Application Support" / APP_NAME


def db_path() -> Path:
    """The local SQLite file. A URL (``postgres://...``) is refused: the live store is file-only."""
    if "://" in os.environ.get("SIGNAL_DESK_LIVE_DB", ""):  # check the raw string: Path() folds "//"
        raise ValueError("SIGNAL_DESK_LIVE_DB must be a file path; the live store is local SQLite only")
    return _env_path("SIGNAL_DESK_LIVE_DB") or app_dir() / "live.sqlite"


def books_path() -> Path:
    """User config if present, else the repo default (the four research books)."""
    explicit = _env_path("SIGNAL_DESK_BOOKS")
    if explicit:
        return explicit
    user = app_dir() / "books.json"
    return user if user.exists() else DEFAULT_BOOKS


def user_books_path() -> Path:
    """Where ``install`` puts an editable copy of the default books."""
    return app_dir() / "books.json"


def models_dir() -> Path:
    return _env_path("SIGNAL_DESK_MODELS_DIR") or app_dir() / "models"


def market_data_dir() -> Path:
    """Closed 5m bars the LightGBM training reads (kept apart from the research cache)."""
    return _env_path("SIGNAL_DESK_DATA_DIR") or app_dir() / "market_data"


def logs_dir() -> Path:
    return _env_path("SIGNAL_DESK_LOGS") or home() / "Library" / "Logs" / APP_NAME


def lock_path() -> Path:
    return app_dir() / "runner.lock"
