"""The long-running loop behind ``signal-desk-live run`` (what launchd keeps alive).

* single instance: an exclusive ``flock`` on ``<app folder>/runner.lock`` (released by the
  OS if the process dies, so a crash never leaves a stale lock);
* every ``poll_seconds`` (default 30) with +/- ``jitter`` so the requests do not line up
  with every other client on the minute, backing off exponentially (max 10 minutes)
  while Binance is unreachable;
* SIGTERM / SIGINT finish the current tick and exit cleanly (launchd sends SIGTERM);
* ``books.json`` is re-read every loop, so edits apply without a restart (a broken edit
  keeps the last good config and logs an event);
* structured JSON logs in ``~/Library/Logs/SignalDesk/runner.jsonl`` (rotated).

A sleeping Mac runs nothing. On wake the next tick sees the newest closed bar, acts on it
alone and records how many bars were missed (``catch_up`` event).
"""

from __future__ import annotations

import fcntl
import json
import logging
import logging.handlers
import os
import random
import signal
import threading
from pathlib import Path

import pandas as pd

from profit_engine.live import paths
from profit_engine.live.market import BarSource, BinanceBarSource
from profit_engine.live.notify import Notifier
from profit_engine.live.registry import ConfigError, LiveConfig, load_config
from profit_engine.live.runner import tick
from profit_engine.live.store import EventSpec, Store

log = logging.getLogger("signal_desk.service")
MAX_BACKOFF_S = 600.0
_STD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": pd.Timestamp(record.created, unit="s", tz="UTC").isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        out.update({k: v for k, v in record.__dict__.items() if k not in _STD_ATTRS})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging(name: str = "runner", to_stderr: bool = True) -> Path:
    folder = paths.logs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.jsonl"
    root = logging.getLogger("signal_desk")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    if to_stderr:
        err = logging.StreamHandler()
        err.setFormatter(JsonFormatter())
        root.addHandler(err)
    return path


class InstanceLock:
    """Exclusive, non-blocking ``flock``; ``acquire()`` is False if another runner holds it."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = self.path.open("a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.close()
            return False
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


def next_delay(poll_s: float, failures: int, jitter: float, rng: random.Random) -> float:
    """Seconds until the next tick: doubled per consecutive failure (capped), with jitter."""
    base = min(poll_s * (2 ** min(failures, 10)), MAX_BACKOFF_S) if failures else poll_s
    return max(1.0, base * rng.uniform(1 - jitter, 1 + jitter))


def exchange_trouble(result: dict) -> bool:
    """Back off only when Binance itself is the problem (not for one book's strategy error)."""
    if result.get("status") == "exchange_unreachable":
        return True
    return any(b.get("decision") == "data_error" for b in result.get("books", []))


def _reload(store: Store, previous: LiveConfig | None) -> LiveConfig | None:
    try:
        return load_config()
    except (ConfigError, ValueError, OSError) as exc:
        log.error("books config could not be loaded", extra={"error": str(exc)})
        now = pd.Timestamp.now(tz="UTC")
        msg = "The books settings file could not be read, so the runner kept the last good settings."
        store.add_event(None, EventSpec("config_error", msg, {}, now.floor("h").isoformat()), now)
        return previous


def run_forever(stop: threading.Event | None = None, source: BarSource | None = None) -> int:
    """Loop until SIGTERM/SIGINT (or ``stop`` is set). Returns a process exit code."""
    lock = InstanceLock(paths.lock_path())
    if not lock.acquire():
        log.warning("another Signal Desk Live runner is already running; exiting")
        return 0
    stop = stop or threading.Event()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop.set())
    store, source = Store(paths.db_path()), source or BinanceBarSource()
    rng, failures, config = random.Random(), 0, None
    log.info("runner started", extra={"db": str(store.path), "pid": os.getpid()})
    try:
        while not stop.is_set():
            config = _reload(store, config)
            if config is not None:
                settings = config.settings
                notifier = Notifier(settings.notifications, settings.notify_min_interval_s)
                try:
                    result = tick(config, store, source, notifier)
                    failures = failures + 1 if exchange_trouble(result) else 0
                    log.info("tick", extra={"result": result})
                except Exception:  # never let one bad tick kill the loop
                    log.exception("tick failed")
                    failures += 1
            poll, jitter = (config.settings.poll_seconds, config.settings.jitter) if config else (30.0, 0.1)
            stop.wait(next_delay(poll, failures, jitter, rng))
    finally:
        store.beat("runner", "stopped", pd.Timestamp.now(tz="UTC"), {"reason": "shutdown"})
        store.close()
        lock.release()
        log.info("runner stopped")
    return 0
