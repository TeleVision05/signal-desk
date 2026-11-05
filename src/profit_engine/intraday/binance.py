"""Raw Binance GLOBAL spot kline download (data.binance.vision + data-api REST).

Functions here return *raw* frames: integer ``open_time`` / ``close_time`` epochs exactly
as published (milliseconds before 2025-01-01, microseconds after in the archives) and
string/float columns. :mod:`profit_engine.intraday.data` normalises them.
"""

from __future__ import annotations

import hashlib
import io
import zipfile

import pandas as pd
import requests

ARCHIVE = "https://data.binance.vision/data/spot"
REST = "https://data-api.binance.vision/api/v3/klines"
RAW_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
)
REST_LIMIT = 1000


class ChecksumError(RuntimeError):
    """The archive's SHA256 does not match its published .CHECKSUM file."""


class ArchiveMissing(RuntimeError):
    """The archive does not exist (HTTP 404), e.g. month not yet published."""


def monthly_url(symbol: str, interval: str, year: int, month: int) -> str:
    name = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    return f"{ARCHIVE}/monthly/klines/{symbol}/{interval}/{name}"


def verify_checksum(payload: bytes, checksum_text: str, name: str) -> None:
    expected = checksum_text.split(maxsplit=1)[0].strip().lower()
    actual = hashlib.sha256(payload).hexdigest()
    if expected != actual:
        raise ChecksumError(f"{name}: sha256 {actual} != published {expected}")


def parse_zip(payload: bytes) -> pd.DataFrame:
    """Parse a kline CSV zip (with or without a header row) into the raw 12 columns."""
    with zipfile.ZipFile(io.BytesIO(payload)) as zf, zf.open(zf.namelist()[0]) as fh:
        raw = pd.read_csv(fh, header=None, dtype=str)
    if not raw.iloc[0, 0].strip().isdigit():  # header row present in some dumps
        raw = raw.iloc[1:]
    raw = raw.iloc[:, : len(RAW_COLUMNS)]
    raw.columns = list(RAW_COLUMNS)
    return raw.reset_index(drop=True)


def fetch_archive(url: str, session: requests.Session, timeout: float = 60) -> pd.DataFrame:
    """Download one archive zip, verify it against ``<url>.CHECKSUM``, and parse it."""
    resp = session.get(url, timeout=timeout)
    if resp.status_code == 404:
        raise ArchiveMissing(url)
    resp.raise_for_status()
    ck = session.get(url + ".CHECKSUM", timeout=timeout)
    ck.raise_for_status()
    verify_checksum(resp.content, ck.text, url.rsplit("/", 1)[-1])
    return parse_zip(resp.content)


def fetch_rest(
    symbol: str,
    interval: str,
    start_ms: int,
    session: requests.Session,
    end_ms: int | None = None,
    timeout: float = 30,
) -> pd.DataFrame:
    """Page through ``/api/v3/klines`` from ``start_ms`` (inclusive) to ``end_ms`` (exclusive)."""
    rows: list[list] = []
    cursor = int(start_ms)
    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": cursor, "limit": REST_LIMIT}
        resp = session.get(REST, params=params, timeout=timeout)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 1
        if len(batch) < REST_LIMIT or (end_ms is not None and cursor >= end_ms):
            break
    raw = pd.DataFrame(rows, columns=list(RAW_COLUMNS)).astype(str)
    if end_ms is not None and len(raw):
        raw = raw[raw["open_time"].astype("int64") < end_ms]
    return raw.reset_index(drop=True)


def month_range(first: tuple[int, int], last: tuple[int, int]) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) from ``first`` to ``last``."""
    out = []
    y, m = first
    while (y, m) <= last:
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out
