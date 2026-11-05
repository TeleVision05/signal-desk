"""Fixtures for the live runner tests (helpers live in live_helpers.py)."""

from __future__ import annotations

import pytest

from profit_engine.live.notify import Notifier
from profit_engine.live.store import Store


@pytest.fixture
def store(tmp_path) -> Store:
    s = Store(tmp_path / "live.sqlite")
    yield s
    s.close()


@pytest.fixture
def quiet() -> Notifier:
    return Notifier(enabled=False)


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """No test may touch the real app folder, logs, or a configured Postgres."""
    monkeypatch.setenv("SIGNAL_DESK_HOME", str(tmp_path / "app"))
    monkeypatch.setenv("SIGNAL_DESK_LOGS", str(tmp_path / "logs"))
    monkeypatch.delenv("SIGNAL_DESK_LIVE_DB", raising=False)
    monkeypatch.delenv("SIGNAL_DESK_BOOKS", raising=False)
    monkeypatch.setenv("SIGNAL_DESK_NOTIFY", "0")
