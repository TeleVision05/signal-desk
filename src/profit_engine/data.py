"""Fetch and cache real market data. Never invent prices."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


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