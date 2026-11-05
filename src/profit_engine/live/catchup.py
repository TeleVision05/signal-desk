"""Decision bars the runner did not act on, and the strategy state carried across them.

The runner trades only the newest closed bar. Bars before it that were never decided (the Mac
was asleep or offline, the feed was stale, a fill bar was halted, the book was warming up) are
not traded: replaying fills would book prices nobody could get any more. They are counted in a
``catch_up`` event (:func:`skipped`, :func:`catch_up_events`).

A path-dependent strategy's STATE is another matter. The trend book's band moves its target only
when the new value is far enough from the current one, so its next target depends on every bar
before it. Feeding it the last persisted target after a gap would run the band from a stale
starting point, and the book would follow a different path from the research until the band
resynced (measured 2026-11-05: 16% of the awake decisions differed with a nightly 23:00-06:00
sleep). :func:`advance` therefore runs the strategy over those bars first, on the window the
runner would have given it at each (the last ``warmup_bars`` closed bars), feeding back its own
output, exactly as :func:`strategy.target_series` does. Nothing is traded or recorded for them;
only the state the newest bar is decided from changes.

A strategy opts in with ``stateful = True`` and ``state_bars`` > 0, the most bars it replays. A
new book replays ``state_bars`` bars before its first decision starting flat, so it starts where
a research path that began years earlier most likely is (the research band starts flat in 2017).
The runner fetches the extra history this needs (:func:`history_needed`).
"""

from __future__ import annotations

import pandas as pd

from profit_engine.live.store import EventSpec, iso
from profit_engine.live.strategy import call_decide, normalize_decision, other_window


def state_bars(strategy) -> int:
    """How many undecided bars the strategy's state is replayed over (0: none)."""
    if not getattr(strategy, "stateful", False):
        return 0
    return max(int(getattr(strategy, "state_bars", 0) or 0), 0)


def history_needed(strategy, last: pd.Timestamp | None, due: pd.Timestamp, tf: pd.Timedelta) -> int:
    """Closed bars to fetch: the decision window, plus the bars whose state must be replayed when
    the newest bar to decide is ``due`` and the last one decided was ``last``."""
    n, cap = max(int(strategy.warmup_bars), 1), state_bars(strategy)
    if not cap:
        return n
    behind = cap if last is None else int((due - last) / tf) - 1
    return n + max(0, min(cap, behind))


def advance(
    strategy, bars: pd.DataFrame, last: pd.Timestamp | None, prev: float, other: pd.DataFrame | None = None
) -> tuple[float, int]:
    """(state to decide ``bars``' newest bar from, number of bars replayed to get it).

    Replays the bars after ``last`` (all of them up to ``state_bars``, or the ``state_bars``
    before the newest for a new book) that have a full window, starting from ``prev``."""
    cap = state_bars(strategy)
    if not cap or bars.empty:
        return float(prev), 0
    n, pos = max(int(strategy.warmup_bars), 1), len(bars) - 1
    start = max(pos - cap, n - 1)
    if last is not None:
        start = max(start, int(bars.index.searchsorted(last, side="right")))
    for i in range(start, pos):
        window = bars.iloc[i + 1 - n : i + 1]
        oth = other_window(other, window) if other is not None else None
        prev = normalize_decision(call_decide(strategy, window, oth, prev)).target
    return float(prev), max(pos - start, 0)


def skipped(tf: pd.Timedelta, frame: pd.DataFrame, bars: pd.DataFrame, last, latest: pd.Timestamp) -> int:
    """Decision bars between ``last`` and ``latest`` that existed but were not decided. Bars that
    were dropped for missing 5m data (a logged gap) never existed as decision bars."""
    if last is None:
        return 0
    slots = pd.date_range(last + tf, latest - tf, freq=tf)
    dropped = int(((slots >= frame.index[0]) & ~slots.isin(bars.index)).sum())
    return len(slots) - dropped


def catch_up_events(timeframe: str, n_skipped: int, replayed: int, latest: pd.Timestamp) -> tuple[EventSpec, ...]:
    if not n_skipped:
        return ()
    text = (
        f"{n_skipped} {timeframe} decision bar(s) since the last decision were not acted on: the Mac was "
        "asleep or offline, or those bars could not be traded (see the earlier events). The book acted on the "
        "latest closed bar only; missed bars are not traded."
    )
    data = {"skipped_bars": n_skipped}
    if replayed:
        text += (
            " The strategy's band was moved through them without trading, so it carries on from where the "
            "backtest would be."
        )
        data["replayed_bars"] = replayed
    return (EventSpec("catch_up", text, data, iso(latest)),)
