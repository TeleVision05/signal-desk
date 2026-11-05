"""Resample 5m bars to 15m / 1h / 4h with a hard completeness check.

Buckets are ``label='left', closed='left'`` (a 1h bar stamped 13:00 covers the 5m bars
opening 13:00..13:55) and anchored to the Unix epoch, so 4h bars start at 00/04/08/... UTC
regardless of where the input begins or ends.

A bucket is complete only if it holds every 5m sub-bar. Incomplete buckets are never used
silently; ``incomplete`` decides what happens:

* ``"raise"`` (default): raise :class:`IncompleteBarError` (an ``AssertionError``).
* ``"drop"``: remove them (the bar index then has holes; positions held across a hole earn
  the open-to-open move across it, which is correct P&L).
* ``"flag"``: keep them with ``complete=False`` so the caller can decide.

Fully empty buckets (no sub-bars at all) are not bars and are always removed in ``drop``
and ``flag`` mode.
"""

from __future__ import annotations

import pandas as pd

from profit_engine.intraday import config

AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "quote_volume": "sum",
    "trades": "sum",
    "taker_buy_base": "sum",
    "taker_buy_quote": "sum",
}
MODES = ("raise", "drop", "flag")


class IncompleteBarError(AssertionError):
    """A resampled bar is missing one or more 5m sub-bars."""


def to_rule(timeframe: str) -> str:
    """Harness timeframe ("5m", "15m", "1h", "4h") -> pandas rule ("5min", ...). pandas 3 reads "m" as month."""
    return timeframe[:-1] + "min" if timeframe.endswith("m") and not timeframe.endswith("min") else timeframe


def sub_bars_per_bar(rule: str) -> int:
    n = pd.Timedelta(to_rule(rule)) / config.BAR
    if n < 1 or n != int(n):
        raise ValueError(f"rule {rule!r} is not a whole multiple of {config.BAR}")
    return int(n)


def resample(df: pd.DataFrame, rule: str, incomplete: str = "raise") -> pd.DataFrame:
    """Aggregate 5m bars into ``rule`` bars ("15m", "1h", "4h" or a pandas rule).

    Adds ``n_sub`` (sub-bars present) and, in flag mode, ``complete``.
    """
    if incomplete not in MODES:
        raise ValueError(f"incomplete must be one of {MODES}")
    expected = sub_bars_per_bar(rule)
    agg = {c: AGG[c] for c in df.columns if c in AGG}
    grouped = df.resample(to_rule(rule), label="left", closed="left", origin="epoch")
    out = grouped.agg(agg)
    out["n_sub"] = grouped["open"].count().astype("int64")
    if len(out) and out["n_sub"].max() > expected:
        raise IncompleteBarError(f"{rule}: a bucket has more than {expected} sub-bars; input not on 5m grid")
    bad = out["n_sub"] < expected
    if incomplete == "raise" and bad.any():
        first = out.index[bad.to_numpy().argmax()]
        raise IncompleteBarError(
            f"{rule}: {int(bad.sum())} of {len(out)} bars miss 5m sub-bars "
            f"(first at {first}, has {out.loc[first, 'n_sub']}/{expected}). "
            "Pass incomplete='drop' or 'flag' explicitly."
        )
    out = out[out["n_sub"] > 0]
    if incomplete == "drop":
        return out[out["n_sub"] == expected]
    if incomplete == "flag":
        out["complete"] = out["n_sub"] == expected
    return out
