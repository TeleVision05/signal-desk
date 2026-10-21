"""Fetch and cache real market data. Never invent prices."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
BINANCE_KLINES_HOSTS = (
    "https://api.binance.us/api/v3/klines",  # US-reachable first
    "https://api.binance.com/api/v3/klines",
)


def _ensure_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def fetch_ohlcv(
    ticker: str,
    start: str = "2007-01-01",
    end: str | None = None,
    *,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance via yfinance (real vendor data)."""
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache = _ensure_dir() / f"{ticker}_{start}_{end}_adj{int(auto_adjust)}.csv"
    meta_path = cache.with_suffix(".meta.json")

    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")
        return df

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError(f"No data returned for {ticker} ({start}→{end})")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df = df.sort_index().dropna(subset=["Close"])
    df.to_csv(cache)
    meta_path.write_text(
        json.dumps(
            {
                "ticker": ticker,
                "start": start,
                "end": end,
                "auto_adjust": auto_adjust,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "rows": int(len(df)),
                "source": "yfinance/yahoo",
            },
            indent=2,
        )
    )
    return df


def _yahoo_crypto_symbol(symbol: str) -> str:
    """Map Binance-style symbols to Yahoo tickers for paper-price fallback."""
    s = symbol.upper().replace("/", "")
    mapping = {
        "BTCUSDT": "BTC-USD",
        "BTCUSD": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "ETHUSD": "ETH-USD",
    }
    return mapping.get(s, s)


def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    *,
    interval: str = "1d",
    start: str = "2018-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Public crypto daily OHLCV for the paper book (no live trading).

    Tries Binance.US / Binance klines; on geo-blocks (e.g. HTTP 451) falls back to
    Yahoo BTC-USD / ETH-USD. Prices are for paper fills only.
    """
    if interval != "1d":
        raise ValueError("Only daily (1d) bars are supported for the paper crypto book")
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache = _ensure_dir() / f"crypto_{symbol}_{interval}_{start}_{end}.csv"
    meta_path = cache.with_suffix(".meta.json")
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")
        return df

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows: list[list] = []
    source = "binance/klines"
    session = requests.Session()
    session.headers.update({"User-Agent": "SignalDesk/0.6-paper"})
    last_err: Exception | None = None

    for host in BINANCE_KLINES_HOSTS:
        rows = []
        cursor = start_ms
        try:
            while cursor < end_ms:
                resp = session.get(
                    host,
                    params={
                        "symbol": symbol.upper(),
                        "interval": interval,
                        "startTime": cursor,
                        "endTime": end_ms,
                        "limit": 1000,
                    },
                    timeout=45,
                )
                if resp.status_code in (418, 429):
                    time.sleep(2)
                    continue
                if resp.status_code == 451:
                    raise RuntimeError(f"{host} returned 451 (geo-blocked)")
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                rows.extend(batch)
                last_open = int(batch[-1][0])
                nxt = last_open + 1
                if nxt <= cursor:
                    break
                cursor = nxt
                if len(batch) < 1000:
                    break
                time.sleep(0.15)
            if rows:
                source = host
                break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            rows = []
            continue

    if rows:
        df = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )
        for col in ("Open", "High", "Low", "Close", "Volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
        df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        df = df[~df.index.duplicated(keep="last")].sort_index().dropna(subset=["Close"])
    else:
        # US-friendly paper fallback: Yahoo crypto (weekend bars included)
        yahoo = _yahoo_crypto_symbol(symbol)
        print(f"[crypto] Binance klines unavailable ({last_err}); Yahoo {yahoo}", flush=True)
        df = fetch_ohlcv(yahoo, start=start, end=end)
        source = f"yfinance/{yahoo}"

    if df.empty:
        raise RuntimeError(f"No crypto OHLCV for {symbol} ({start}→{end})")

    df.to_csv(cache)
    meta_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "interval": interval,
                "start": start,
                "end": end,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "rows": int(len(df)),
                "source": source,
                "paper_only": True,
            },
            indent=2,
        )
    )
    return df


def fetch_panel(
    tickers: list[str],
    start: str = "2007-01-01",
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    return {t: fetch_ohlcv(t, start=start, end=end) for t in tickers}


def month_end_closes(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Last available adjusted close per completed calendar month for each ticker."""
    series = {}
    for t, df in panel.items():
        s = df["Close"].resample("ME").last().dropna()
        series[t] = s
    out = pd.DataFrame(series).dropna(how="any")
    # Drop the current incomplete month (label is month-end but last bar is mid-month).
    if not out.empty:
        last_label = out.index[-1]
        # Keep only if we have a bar within 3 calendar days of the month-end label.
        ref_ticker = next(iter(panel))
        last_bar = panel[ref_ticker].index[panel[ref_ticker].index <= last_label].max()
        if (last_label - last_bar).days > 3:
            out = out.iloc[:-1]
    return out


def next_open_prices(panel: dict[str, pd.DataFrame], signal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For each month-end signal date, take the next session's Open."""
    rows = []
    for dt in signal_dates:
        row = {"signal_date": dt}
        ok = True
        for t, df in panel.items():
            future = df.loc[df.index > dt]
            if future.empty:
                ok = False
                break
            row[t] = float(future.iloc[0]["Open"])
            row["fill_date"] = future.index[0]
        if ok:
            rows.append(row)
    return pd.DataFrame(rows).set_index("signal_date")


def fetch_cpi_yoy(start: str = "2007-01-01") -> pd.Series:
    """YoY inflation from FRED CPIAUCSL (real BLS data; multiple download fallbacks)."""
    import io
    import re

    import requests

    cache = _ensure_dir() / "CPIAUCSL.csv"
    if not cache.exists() or cache.stat().st_size < 100:
        cpi = None
        # 1) Official graph CSV
        try:
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
            resp = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            cpi = pd.read_csv(io.StringIO(resp.text))
        except Exception:
            cpi = None
        # 2) FRED data table page
        if cpi is None:
            try:
                resp = requests.get(
                    "https://fred.stlouisfed.org/data/CPIAUCSL",
                    timeout=45,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                resp.raise_for_status()
                rows = re.findall(r"(\d{4}-\d{2}-\d{2})\s+([0-9]+(?:\.[0-9]+)?)", resp.text)
                if not rows:
                    rows = re.findall(
                        r"\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([0-9.]+)\s*\|", resp.text
                    )
                cpi = pd.DataFrame(rows, columns=["observation_date", "CPIAUCSL"])
            except Exception as exc:
                raise RuntimeError(
                    "Could not download CPIAUCSL from FRED; place CSV at data/cache/CPIAUCSL.csv"
                ) from exc
        if "observation_date" not in cpi.columns and "DATE" in cpi.columns:
            cpi = cpi.rename(columns={"DATE": "observation_date"})
        cpi.to_csv(cache, index=False)

    cpi = pd.read_csv(cache)
    date_col = "observation_date" if "observation_date" in cpi.columns else "DATE"
    cpi[date_col] = pd.to_datetime(cpi[date_col])
    cpi = cpi.set_index(date_col).sort_index()
    series = pd.to_numeric(cpi["CPIAUCSL"], errors="coerce").dropna()
    series = series.loc[series.index >= start]
    yoy = series.pct_change(12)
    return yoy.dropna()