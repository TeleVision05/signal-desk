"""macOS notifications for paper fills and the daily-loss stop, throttled per book.

Messages go into the ``notifications`` table first (so a throttled message survives a
restart), and :meth:`Notifier.flush` sends at most ONE notification per book per
``min_interval`` seconds, summarising everything that piled up since the last one.
Delivery uses ``osascript -e 'display notification ...'``; on other systems, or with
notifications turned off in books.json (or ``SIGNAL_DESK_NOTIFY=0``), nothing is sent.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from collections.abc import Callable

import pandas as pd

from profit_engine.live.store import Store

TITLE = "Signal Desk Live (paper)"
MAX_LINES = 3
log = logging.getLogger("signal_desk.notify")

Sender = Callable[[str, str], bool]


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def osascript_sender(title: str, text: str) -> bool:
    """Show a macOS notification. Returns False (never raises) if it could not be shown."""
    if sys.platform != "darwin" or not shutil.which("osascript"):
        return False
    script = f"display notification {_quote(text)} with title {_quote(title)}"
    try:
        done = subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        log.warning("notification could not be shown", exc_info=True)
        return False
    return done.returncode == 0


def summarize(book_title: str, texts: list[str]) -> str:
    if len(texts) == 1:
        return texts[0]
    shown = "; ".join(texts[-MAX_LINES:])
    earlier = len(texts) - MAX_LINES
    more = f" (+{earlier} earlier)" if earlier > 0 else ""
    return f"{len(texts)} updates for {book_title}: {shown}{more}"


class Notifier:
    def __init__(self, enabled: bool = True, min_interval_s: float = 300.0, sender: Sender | None = None) -> None:
        self.enabled = enabled
        self.min_interval = pd.Timedelta(seconds=min_interval_s)
        self.sender = sender or osascript_sender

    def queue(self, store: Store, book_id: str, text: str, now: pd.Timestamp) -> None:
        if self.enabled:
            store.queue_notification(book_id, text, now)

    def flush(self, store: Store, now: pd.Timestamp, titles: dict[str, str] | None = None) -> int:
        """Send what is due; returns how many notifications were shown."""
        if not self.enabled:
            return 0
        by_book: dict[str, list[dict]] = {}
        for row in store.pending_notifications():
            by_book.setdefault(row["book_id"], []).append(row)
        shown = 0
        for book_id, rows in by_book.items():
            last = store.last_notification_sent(book_id)
            if last is not None and now - last < self.min_interval:
                continue
            title = (titles or {}).get(book_id, book_id)
            if self.sender(TITLE, summarize(title, [r["text"] for r in rows])):
                shown += 1
            # Mark as handled even if the OS refused: retrying every tick would only spam later.
            store.mark_notifications_sent([r["id"] for r in rows], now)
        return shown
