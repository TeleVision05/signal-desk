"""One paper-trading step per completed daily bar, following ``run_long_only_daily``.

The backtest decides at bar t's close and fills at bar t+1's open; a stop fires on a bar
using the highest high since entry. The paper books used to re-run the model on every sync
and trade on each answer, filling at the decision bar's own open (a price from before the
signal existed). The model's answer for one bar moved between syncs (e.g. the VIX download
failing), so the TQQQ book bought and sold the same Oct 30 2026 bar five times.

Rules here:
- The first decision recorded for a bar is final. A later sync for the same bar only reports.
- A decision is carried out at the open of the next bar, as in the backtest.
- Stops are checked on every bar since the last sync, with the peak taken since the buy.
- Bars between syncs get no decision of their own; the newest bar is decided.
- A step with missing inputs records nothing, so the next sync decides the bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .daily_backtest import StopConfig, _stop_hit
from .decisions import MAX_ROWS, append_decision, explain_decision, load_decisions
from .ledger import PaperLedger

STATE_VERSION = 2


@dataclass(frozen=True)
class BookSpec:
    run_id: str
    book: str
    strategy: str
    asset: str
    cost_bps: float
    stop: StopConfig
    mark_source: str
    threshold: float
    config: dict[str, Any]


def last_decided(run_id: str) -> dict[str, Any] | None:
    """Newest decision written by this module for the run (older rows have no state_version)."""
    for row in load_decisions(limit=MAX_ROWS, run_id=run_id):
        if row.get("state_version") == STATE_VERSION:
            return row
    return None


def _open_position(ledger: PaperLedger, spec: BookSpec, initial_cash: float) -> dict:
    try:
        return ledger.get_position(spec.run_id)
    except KeyError:
        ledger.start_run(spec.run_id, spec.strategy, spec.config, initial_cash=initial_cash)
        return ledger.get_position(spec.run_id)


def _fill(ledger: PaperLedger, spec: BookSpec, side: str, price: float, source: str, ts: pd.Timestamp) -> None:
    half = spec.cost_bps / 10_000 / 2
    pos = ledger.get_position(spec.run_id)
    if side == "BUY":
        cash = float(pos["cash"])
        cost = cash * half
        shares = (cash - cost) / price
    else:
        shares = float(pos["shares"])
        cost = shares * price * half
    ledger.apply_fill(
        spec.run_id,
        side=side,
        asset=spec.asset,
        shares=shares,
        price=price,
        cost=cost,
        source=source,
        ts=ts.isoformat(),
    )


def _run_bars(ledger: PaperLedger, spec: BookSpec, ohlc: pd.DataFrame, new: pd.DatetimeIndex, prev_signal) -> str:
    """Fill the previous decision at the first new bar's open, then check stops on every new bar."""
    action = "hold"
    for i, ts in enumerate(new):
        bar = ohlc.loc[ts]
        shares = float(ledger.get_position(spec.run_id)["shares"])
        if i == 0 and prev_signal is not None:
            if prev_signal == 1 and shares == 0:
                _fill(ledger, spec, "BUY", float(bar["Open"]), "ml_signal", ts)
                action = "signal_entry"
            elif prev_signal == 0 and shares > 0:
                _fill(ledger, spec, "SELL", float(bar["Open"]), "ml_signal", ts)
                action = "signal_exit"
        shares = float(ledger.get_position(spec.run_id)["shares"])
        if shares <= 0:
            continue
        buy = ledger.last_buy(spec.run_id)
        if buy is None:
            continue
        entry_px, entry_ts = buy
        held = ohlc.loc[(ohlc.index >= _naive_like(entry_ts, ohlc.index)) & (ohlc.index <= ts)]
        peak = max(entry_px, float(held["High"].max())) if len(held) else entry_px
        hit, exit_px = _stop_hit(bar, entry_px, peak, spec.stop)
        if hit and exit_px is not None:
            _fill(ledger, spec, "SELL", float(exit_px), "stop", ts)
            action = "stop_exit"
    return action


def _naive_like(ts: str, index: pd.DatetimeIndex) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if index.tz is None and out.tzinfo is not None:
        out = out.tz_convert("UTC").tz_localize(None)
    return out.normalize()


def step(
    spec: BookSpec,
    ohlc: pd.DataFrame,
    want: int | None,
    proba: float | None,
    *,
    ledger: PaperLedger,
    initial_cash: float = 1000.0,
    problem: str | None = None,
) -> dict[str, Any]:
    """Advance one book to the newest completed bar of ``ohlc``, deciding that bar once.

    ``want``/``proba`` are the model's answer for ``ohlc.index[-1]``. Pass ``problem`` (and
    want=None) when an input was missing: nothing is traded or recorded as a decision.
    """
    t = ohlc.index[-1]
    bar_date = str(t.date())
    now = datetime.now(timezone.utc).isoformat()
    prev = last_decided(spec.run_id)
    if prev is not None and str(prev["bar_date"]) >= bar_date:
        return {**prev, "action": "already_decided", "synced_at": now, "traded": False}

    _open_position(ledger, spec, initial_cash)
    if problem is not None or want is None or proba is None or pd.isna(proba):
        why = problem or "the model had no score for this bar"
        return {"run_id": spec.run_id, "book": spec.book, "bar_date": bar_date, "action": "skipped",
                "problem": why, "synced_at": now, "traded": False}  # fmt: skip

    pos_before = dict(ledger.get_position(spec.run_id))
    prev_bar = pd.Timestamp(prev["bar_date"]) if prev is not None else None
    new = ohlc.index[ohlc.index > _naive_like(str(prev_bar), ohlc.index)] if prev_bar is not None else ohlc.index[-1:]
    action = _run_bars(ledger, spec, ohlc, new, prev.get("signal") if prev is not None else None)

    pos = ledger.get_position(spec.run_id)
    mark = float(ohlc["Close"].iloc[-1])
    held = float(pos["shares"]) > 0
    equity = ledger.mark(spec.run_id, mark_price=mark if held else 0.0, source=spec.mark_source, ts=t.isoformat())
    info = {
        "run_id": spec.run_id,
        "book": spec.book,
        "strategy": spec.strategy,
        "state_version": STATE_VERSION,
        "signal": int(want),
        "proba": float(proba),
        "threshold": spec.threshold,
        "gap_to_threshold": float(proba) - spec.threshold,
        "comparison": "pass" if proba >= spec.threshold else "fail",
        "bar_date": bar_date,
        "action": action,
        "equity": equity,
        "cash": pos["cash"],
        "asset": pos["asset"] or spec.asset,
        "shares": pos["shares"],
        "mark": mark,
        "open": float(ohlc["Open"].iloc[-1]),
        "pos_before": {k: pos_before.get(k) for k in ("cash", "shares", "asset")},
        "synced_at": now,
        "traded": action != "hold",
    }
    info["explanation"] = explain_decision(info)
    append_decision({k: v for k, v in info.items() if k not in ("pos_before", "open", "traded")})
    return info
