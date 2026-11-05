"""Helpers for the live runner tests: a replayable fake bar source (no network) and tiny strategies."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd

from profit_engine.live.market import FIVE_MIN, DataUnavailable
from profit_engine.live.registry import BookSpec, LiveConfig, Settings
from profit_engine.live.store import Store
from profit_engine.live.strategy import Decision, StrategyMeta

POISON = 1e9  # what an in-progress bar's close/high look like here: any use of it is obvious


def make_5m(n: int = 2000, seed: int = 0, start: str = "2021-03-01", vol: float = 0.002) -> pd.DataFrame:
    """Random-walk 5m bars on the grid. Each open is NOT the previous close (a small jump), so
    a test can tell "filled at the next open" from "filled at the decision bar's close"."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC", name="open_time")
    close = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(rng.normal(0, vol / 4, n))
    wick = np.abs(rng.normal(0, vol, n))
    volume = rng.gamma(2.0, 5.0, n)
    taker = volume * rng.uniform(0.2, 0.8, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * np.exp(wick),
            "low": np.minimum(open_, close) * np.exp(-wick),
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": rng.integers(1, 500, n),
            "taker_buy_base": taker,
            "taker_buy_quote": taker * close,
        },
        index=idx,
    )


def path_bars(prices: list[float] | np.ndarray, start: str = "2021-03-01") -> pd.DataFrame:
    """5m bars whose closes follow ``prices`` exactly (open = previous close)."""
    close = np.asarray(prices, dtype="float64")
    idx = pd.date_range(start, periods=len(close), freq="5min", tz="UTC", name="open_time")
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = np.full(len(close), 10.0)
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close), "close": close,
         "volume": vol, "quote_volume": vol * close, "trades": np.full(len(close), 5),
         "taker_buy_base": vol / 2, "taker_buy_quote": vol / 2 * close},
        index=idx,
    )  # fmt: skip


class FakeSource:
    """Replays a fixed 5m history as Binance would serve it at ``now``: every bar that has
    STARTED, the in-progress one with a poisoned close/high (its open is real and final)."""

    def __init__(self, frames: dict[str, pd.DataFrame], now: str | pd.Timestamp) -> None:
        self.frames, self.now, self.down = frames, pd.Timestamp(now), False
        self.calls = 0

    def server_time(self) -> pd.Timestamp:
        if self.down:
            raise DataUnavailable("ConnectionError: offline")
        return self.now

    def klines_5m(self, symbol: str, start: pd.Timestamp) -> pd.DataFrame:
        if self.down:
            raise DataUnavailable("ConnectionError: offline")
        self.calls += 1
        df = self.frames[symbol]
        df = df[(df.index >= start) & (df.index <= self.now)].copy()
        if len(df) and df.index[-1] + FIVE_MIN > self.now:
            df.loc[df.index[-1], ["close", "high"]] = POISON
        return df


class Const:
    """Always the same target; optional barrier inside decide() to force a race."""

    def __init__(self, target: float = 1.0, symbol: str = "BTCUSDT", timeframe: str = "1h", warmup: int = 1):
        self.name, self.symbol, self.timeframe = "const", symbol, timeframe
        self.warmup_bars, self.other_symbol, self.target = warmup, None, target
        self.meta = StrategyMeta(version="t1", params={"target": target})
        self.barrier: threading.Barrier | None = None
        self.windows: list[pd.DataFrame] = []

    def decide(self, bars: pd.DataFrame, other: pd.DataFrame | None = None) -> Decision:
        self.windows.append(bars)
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        return Decision(self.target, f"constant {self.target}")


def book(strategy, book_id: str = "b1", **kw) -> BookSpec:
    return BookSpec(
        book_id=book_id,
        title=kw.pop("title", "Test book"),
        strategy=strategy,
        enabled=kw.pop("enabled", True),
        starting_equity=kw.pop("starting_equity", 10_000.0),
        cost_bps_per_side=kw.pop("cost_bps_per_side", 5.0),
        max_daily_loss=kw.pop("max_daily_loss", 0.05),
    )


def config(*books: BookSpec, **settings) -> LiveConfig:
    return LiveConfig(settings=Settings(**{"notifications": False, **settings}), books=books, source=Path("test"))


def rows(store: Store, sql: str, *args) -> list[dict]:
    return [dict(r) for r in store.conn.execute(sql, args).fetchall()]
