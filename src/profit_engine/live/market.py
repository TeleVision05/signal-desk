"""Where the runner gets 5m bars: Binance's public data API (or a fake in tests).

A :class:`BarSource` returns 5m klines INCLUDING the bar still in progress, plus the
exchange's clock. The runner (not the source) decides what is closed, against the
exchange's time, so a fast or slow Mac clock cannot make an unfinished bar look closed.

The Binance source reuses the harness plumbing by importing it
(``intraday.binance.fetch_rest`` and ``intraday.data.normalize_raw``) and keeps a small
in-memory cache so a 30-second loop re-downloads only the newest bars.
"""

from __future__ import annotations

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
    def __init__(self, session: requests.Session | None = None, retries: int = 2, timeout: float = 10.0) -> None:
        self.session = session or requests.Session()
        self.retries, self.timeout = retries, timeout
        self._cache: dict[str, pd.DataFrame] = {}

    def _with_retries(self, fn):
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                return fn()
            except (requests.RequestException, ValueError, KeyError) as exc:
                if attempt == self.retries:
                    raise DataUnavailable(f"{type(exc).__name__}: {exc}") from exc
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def server_time(self) -> pd.Timestamp:
        def call() -> pd.Timestamp:
            resp = self.session.get(SERVER_TIME_URL, timeout=self.timeout)
            resp.raise_for_status()
            return pd.Timestamp(int(resp.json()["serverTime"]), unit="ms", tz="UTC")

        return self._with_retries(call)

    def _fetch(self, symbol: str, start: pd.Timestamp) -> pd.DataFrame:
        def call() -> pd.DataFrame:
            start_ms = int(start.value // 1_000_000)
            raw = binance.fetch_rest(symbol, "5m", start_ms, self.session, timeout=self.timeout)
            if raw.empty:
                return pd.DataFrame(columns=list(KLINE_COLUMNS), index=pd.DatetimeIndex([], tz="UTC"))
            # normalize_raw drops bars with close_time >= now. Pass a far-future "now" so the
            # in-progress bar is kept: its OPEN is the fill price. split_closed() decides later.
            frame, _ = normalize_raw(raw, now=pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1))
            return frame

        return self._with_retries(call)

    def klines_5m(self, symbol: str, start: pd.Timestamp) -> pd.DataFrame:
        cached = self._cache.get(symbol)
        if cached is None or cached.empty or cached.index[0] > start:
            frame = self._fetch(symbol, start)
        else:
            # Re-fetch from the newest cached bar: it may have been in progress last time.
            fresh = self._fetch(symbol, cached.index[-1])
            frame = pd.concat([cached, fresh])
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        if not frame.empty:
            self._cache[symbol] = frame[frame.index >= frame.index[-1] - CACHE_SPAN]
        return frame[frame.index >= start]
