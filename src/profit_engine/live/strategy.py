"""Strategy contract for the live runner, plus one DEMO strategy.

A strategy sees only CLOSED bars of its own timeframe (index = bar OPEN time, UTC) and
returns the target for the LAST bar it was given: a float in ``[0, 1]`` (long/flat spot),
or a :class:`Decision` carrying that float and a plain-language explanation. ``NaN`` means
flat, as in the harness. The runner fills the target at the NEXT bar's open, exactly like
``intraday.backtest.run_backtest``.

``decide`` receives the last ``warmup_bars`` closed bars (fewer means the book waits), so a
strategy must be a function of that window only. :func:`target_series` replays a strategy
bar by bar with the same window, which is how a backtest of the live code path is built.

A strategy whose next target depends on its own previous one (a hysteresis band) sets
``stateful = True``; :func:`call_decide` then passes ``prev_target``: the strategy's own last
target as the runner persisted it (before any daily-loss pause), or 0.0 for a new book.
A strategy that cannot decide at all (say its model file is missing) raises
:class:`StrategyUnavailable` with a plain-language message; the book keeps its position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

DEMO_LABEL = "demo — not promoted"
PROMOTED_LABEL = "promoted"


@dataclass(frozen=True)
class StrategyMeta:
    version: str
    params: dict = field(default_factory=dict)
    artifact_hash: str = ""
    promoted: bool = False
    source: str = "builtin"
    status: str = ""  # label shown when not promoted (defaults to DEMO_LABEL)
    family: str = ""  # dashboard group, e.g. "Trend (vol-targeted)"

    @property
    def label(self) -> str:
        return PROMOTED_LABEL if self.promoted else (self.status or DEMO_LABEL)


@dataclass(frozen=True)
class Decision:
    target: float
    explanation: str
    details: dict = field(default_factory=dict)


@runtime_checkable
class Strategy(Protocol):
    name: str
    symbol: str
    timeframe: str
    warmup_bars: int
    other_symbol: str | None
    meta: StrategyMeta

    def decide(self, bars: pd.DataFrame, other: pd.DataFrame | None = None) -> float | Decision: ...


class InvalidTarget(ValueError):
    """A strategy returned something that is not a long/flat target in [0, 1]."""


class StrategyUnavailable(RuntimeError):
    """The strategy cannot decide at all (e.g. no usable model); the message is for the user."""


def call_decide(
    strategy: Strategy, bars: pd.DataFrame, other: pd.DataFrame | None, prev_target: float = 0.0
) -> float | Decision:
    """``strategy.decide``, passing ``prev_target`` only to a ``stateful`` strategy."""
    if getattr(strategy, "stateful", False):
        return strategy.decide(bars, other, prev_target=float(prev_target))
    return strategy.decide(bars, other)


def normalize_decision(raw: float | Decision) -> Decision:
    """Coerce a strategy's output to a validated :class:`Decision` (NaN -> flat, as the harness)."""
    decision = raw if isinstance(raw, Decision) else Decision(float(raw), "")
    try:
        target = float(decision.target)
    except (TypeError, ValueError) as exc:
        raise InvalidTarget(f"target is not a number: {decision.target!r}") from exc
    if math.isnan(target):
        target = 0.0
    if not math.isfinite(target) or target < 0 or target > 1:
        raise InvalidTarget(f"long/flat target must be in [0, 1], got {target!r}")
    text = decision.explanation or f"The strategy asked for {target:.0%} of the book in the coin."
    return Decision(target, text, dict(decision.details))


def paused_decision(paused_until: pd.Timestamp, raw: Decision) -> Decision:
    """What a book paused by its daily loss limit does instead of ``raw``: stay in cash."""
    text = (
        "The book is paused after hitting its daily loss limit, so it stays in cash until "
        f"{paused_until:%Y-%m-%d %H:%M} UTC."
    )
    return Decision(0.0, text, {"strategy_target": raw.target, "strategy_explanation": raw.explanation})


def decision_window(bars: pd.DataFrame, strategy: Strategy) -> pd.DataFrame | None:
    """The bars ``decide`` gets: the last ``warmup_bars`` closed bars, or None if too few."""
    n = max(int(strategy.warmup_bars), 1)
    return bars.iloc[-n:] if len(bars) >= n else None


def other_window(other: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
    """The second symbol's closed bars over the same span as ``window`` (never past its end)."""
    return other.loc[window.index[0] : window.index[-1]]


def target_series(
    strategy: Strategy, bars: pd.DataFrame, other: pd.DataFrame | None = None, prev_target: float = 0.0
) -> pd.Series:
    """Per-bar targets exactly as the live runner would decide them (NaN while warming up).

    A stateful strategy gets its own previous output back each bar, starting from
    ``prev_target`` (0.0: a new book is flat), as the runner feeds it from the store."""
    out = pd.Series(np.nan, index=bars.index, dtype="float64")
    prev = float(prev_target)
    for i in range(len(bars)):
        window = decision_window(bars.iloc[: i + 1], strategy)
        if window is None:
            continue
        oth = other_window(other, window) if other is not None else None
        prev = normalize_decision(call_decide(strategy, window, oth, prev)).target
        out.iloc[i] = prev
    return out


def _coin(symbol: str) -> str:
    return symbol.removesuffix("USDT")


def _fmt(x: float) -> str:
    return f"{x:,.2f}"


class SmaTrendDemo:
    """DEMO ONLY - not promoted. Long when the fast simple moving average of closes is above
    the slow one, else flat. Finite windows, so a live window and a full-history replay agree
    exactly. It exists to exercise the runner, not to make money."""

    name = "demo_sma_trend"

    def __init__(self, symbol: str, timeframe: str = "4h", fast: int = 10, slow: int = 30) -> None:
        if not 1 <= fast < slow:
            raise ValueError("demo_sma_trend needs 1 <= fast < slow")
        self.symbol, self.timeframe, self.fast, self.slow = symbol, timeframe, int(fast), int(slow)
        self.warmup_bars = self.slow
        self.other_symbol = None
        self.meta = StrategyMeta(version="1", params={"fast": self.fast, "slow": self.slow})

    def decide(self, bars: pd.DataFrame, other: pd.DataFrame | None = None) -> Decision:
        close = bars["close"].to_numpy(dtype="float64")
        fast = float(close[-self.fast :].mean())
        slow = float(close[-self.slow :].mean())
        coin = _coin(self.symbol)
        up = fast > slow
        verdict = f"so the book holds {coin}" if up else "so the book stays in cash"
        text = (
            f"The {self.fast}-bar average price ({_fmt(fast)}) is {'above' if up else 'not above'} "
            f"the {self.slow}-bar average ({_fmt(slow)}), {verdict}."
        )
        return Decision(1.0 if up else 0.0, text, {"sma_fast": fast, "sma_slow": slow})
