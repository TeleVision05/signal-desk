"""Where the runner gets 5m bars: Binance's public data API (or a fake in tests).

A :class:`BarSource` returns 5m klines INCLUDING the bar still in progress, plus the
exchange's clock. The runner (not the source) decides what is closed, against the
exchange's time, so a fast or slow Mac clock cannot make an unfinished bar look closed.

The Binance source reuses the harness plumbing by importing it
(``intraday.binance.fetch_rest`` and ``intraday.data.normalize_raw``) and keeps an in-memory
cache so a 30-second loop re-downloads only the newest bars. The cache keeps the widest span
any book has asked for (at least :data:`CACHE_SPAN`): the trend and LightGBM books need 150
and 500 days of 5m history, which is fetched once per process and then only topped up. A
request older than the cache fetches just the missing older bars.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol

import pandas as pd
import requests

from profit_engine.intraday import binance
from profit_engine.intraday.config import KLINE_COLUMNS
from profit_engine.intraday.data import normalize_raw

FIVE_MIN = pd.Timedelta("5min")
SERVER_TIME_URL = "https://data-api.binance.vision/api/v3/time"
CACHE_SPAN = pd.Timedelta(days=21)
# Each refresh re-fetches this many bars before the newest cached one: a bar that had just
# closed at the previous fetch (inside the settle margin) may since have been finalized.
REFETCH_BARS = 2
# What normalize_raw is told "now" is. It drops bars with close_time >= now; the runner wants
# every bar kept (the in-progress bar's OPEN is the fill price) and decides what is closed itself,
# on Binance's clock. A fixed far-future time, never the Mac's clock, which may be days off.
KEEP_ALL = pd.Timestamp("2200-01-01", tz="UTC")


class DataUnavailable(RuntimeError):
    """The exchange could not be reached or answered with something unusable."""


class BarSource(Protocol):
    def server_time(self) -> pd.Timestamp: ...

    def klines_5m(self, symbol: str, start: pd.Timestamp) -> pd.DataFrame:
        """5m bars opening at or after ``start`` (index = UTC open time, KLINE_COLUMNS), the
        in-progress bar included."""
        ...


def split_closed(frame: pd.DataFrame, now: pd.Timestamp, settle: pd.Timedelta) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(closed bars, in-progress bars). A 5m bar is closed once ``open + 5m + settle <= now``
    on the exchange's clock; anything later may still change and is never decided on."""
    closed = (frame.index + FIVE_MIN + settle) <= now
    return frame[closed], frame[~closed]


class BinanceBarSource:
    def __init__(
        self, session: requests.Session | None = None, retries: int = 2, timeout: float = 10.0,
        stop: threading.Event | None = None,
    ) -> None:  # fmt: skip
        """``stop`` (the runner's shutdown event) ends retries at once, so a stalled network
        cannot hold a SIGTERM past launchd's exit timeout."""
        self.session = session or requests.Session()
        self.retries, self.timeout, self.stop = retries, timeout, stop
        self._cache: dict[str, pd.DataFrame] = {}
        self._span: dict[str, pd.Timedelta] = {}  # widest lookback asked for, per symbol

    def _stopping(self) -> bool:
        return self.stop is not None and self.stop.is_set()

    def _with_retries(self, fn):
        delay = 1.0
        for attempt in range(self.retries + 1):
            if self._stopping():
                raise DataUnavailable("stopping: the runner is shutting down")
            try:
                return fn()
            except (requests.RequestException, ValueError, KeyError) as exc:
                if attempt == self.retries or self._stopping():
                    raise DataUnavailable(f"{type(exc).__name__}: {exc}") from exc
                if self.stop is not None:
                    self.stop.wait(delay)
                else:
                    time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def server_time(self) -> pd.Timestamp:
        def call() -> pd.Timestamp:
            resp = self.session.get(SERVER_TIME_URL, timeout=self.timeout)
            resp.raise_for_status()
            return pd.Timestamp(int(resp.json()["serverTime"]), unit="ms", tz="UTC")

        return self._with_retries(call)

    def _fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp | None = None) -> pd.DataFrame:
        def call() -> pd.DataFrame:
            start_ms = int(start.value // 1_000_000)
            end_ms = None if end is None else int(end.value // 1_000_000)
            raw = binance.fetch_rest(symbol, "5m", start_ms, self.session, end_ms=end_ms, timeout=self.timeout)
            if raw.empty:
                return pd.DataFrame(columns=list(KLINE_COLUMNS), index=pd.DatetimeIndex([], tz="UTC"))
            frame, _ = normalize_raw(raw, now=KEEP_ALL)  # split_closed() decides what is closed later
            return frame

        return self._with_retries(call)

    def klines_5m(self, symbol: str, start: pd.Timestamp) -> pd.DataFrame:
        cached = self._cache.get(symbol)
        if cached is None or cached.empty:
            frame = self._fetch(symbol, start)
        else:
            # Older bars than the cache holds: fetch only those. Then re-fetch from just before the
            # newest cached bar: it may have been in progress last time, and the bar before it may
            # have closed only moments before that fetch.
            older = [self._fetch(symbol, start, end=cached.index[0])] if cached.index[0] > start else []
            fresh = self._fetch(symbol, cached.index[max(len(cached) - 1 - REFETCH_BARS, 0)])
            frame = pd.concat([*older, cached, fresh])
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        if not frame.empty:
            span = max(self._span.get(symbol, CACHE_SPAN), frame.index[-1] - start)
            self._span[symbol] = span
            self._cache[symbol] = frame[frame.index >= frame.index[-1] - span]
        return frame[frame.index >= start]
