"""One pass over the paper books: :func:`tick`.

Per enabled book:

1. If no decision bar can have closed since the last one processed (judged on the
   EXCHANGE's clock), skip straight to marking. Otherwise fetch recent 5m bars, drop the
   in-progress ones (:func:`market.split_closed`), resample with the harness ``resample``
   (``incomplete="drop"``) and take the latest CLOSED decision bar.
2. Guards: stale feed (latest closed bar more than ``stale_after_bars`` timeframes old) and
   price sanity skip with an event; the fill price must exist (the next bar's open).
3. Decide on the latest closed bar ONLY. Bars missed while the Mac slept are counted in a
   ``catch_up`` event, never replayed. Fill at the NEXT bar's open (its open is fixed once
   that bar starts), cost ``cost_bps_per_side`` on the traded fraction. All of it is one
   :meth:`Store.commit_bar` transaction, so the bar is traded at most once.
4. Mark equity at the latest closed 5m close. If equity is down ``max_daily_loss`` from the
   first mark of the UTC day, go flat at the next 5m open and pause until next UTC midnight.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import pandas as pd

from profit_engine.intraday.data import find_gaps
from profit_engine.intraday.resample import resample
from profit_engine.live.market import FIVE_MIN, BarSource, DataUnavailable, split_closed
from profit_engine.live.notify import Notifier
from profit_engine.live.registry import BookSpec, LiveConfig, Settings
from profit_engine.live.store import BarCommit, EventSpec, Store, iso, parse_ts
from profit_engine.live.strategy import Decision, InvalidTarget, decision_window, normalize_decision, other_window

log = logging.getLogger("signal_desk.runner")
MARK_LOOKBACK = pd.Timedelta("30min")
MARK_JUMP_WINDOW = pd.Timedelta("1h")
SKEW_WARN_S = 60.0


@dataclass
class TickContext:
    store: Store
    source: BarSource
    settings: Settings
    notifier: Notifier
    now: pd.Timestamp  # exchange time
    frames: dict[str, tuple[pd.Timestamp, pd.DataFrame]] = field(default_factory=dict)

    @property
    def settle(self) -> pd.Timedelta:
        return pd.Timedelta(seconds=self.settings.settle_seconds)

    def event(self, book_id: str | None, kind: str, message: str, key: str | None = None, **data) -> None:
        self.store.add_event(book_id, EventSpec(kind, message, data, key), self.now)


def coin(symbol: str) -> str:
    return symbol.removesuffix("USDT")


def latest_due_bar(now: pd.Timestamp, tf: pd.Timedelta, settle: pd.Timedelta) -> pd.Timestamp:
    """Open time of the newest decision bar that can have closed by ``now``."""
    return (now - settle).floor(tf) - tf


def bars_5m(ctx: TickContext, symbol: str, since: pd.Timestamp) -> pd.DataFrame:
    """5m frame (in-progress bar included), fetched at most once per symbol and window per tick."""
    have = ctx.frames.get(symbol)
    if have is not None and have[0] <= since:
        return have[1][have[1].index >= since]
    frame = ctx.source.klines_5m(symbol, since)
    ctx.frames[symbol] = (since, frame)
    return frame


def closed_bars(ctx: TickContext, spec: BookSpec, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(5m frame incl. in-progress bar, closed decision bars of the book's timeframe)."""
    need = max(int(spec.strategy.warmup_bars), 1) + 4
    frame = bars_5m(ctx, symbol, ctx.now.floor(spec.tf) - spec.tf * need)
    closed, _ = split_closed(frame, ctx.now, ctx.settle)
    if symbol == spec.symbol:
        _note_gaps(ctx, spec, closed)
    if closed.empty:
        return frame, closed
    return frame, resample(closed, spec.timeframe, incomplete="drop")


def _note_gaps(ctx: TickContext, spec: BookSpec, closed: pd.DataFrame) -> None:
    for gap in find_gaps(closed).itertuples():
        ctx.event(
            spec.book_id, "gap",
            f"Binance has no {coin(spec.symbol)} data for {gap.missing_bars} five-minute bar(s) after "
            f"{gap.after:%Y-%m-%d %H:%M} UTC. Any {spec.timeframe} bar missing data is skipped, not guessed.",
            key=iso(gap.after), missing_bars=int(gap.missing_bars),
        )  # fmt: skip


def _price_ok(ctx: TickContext, spec: BookSpec, price: float, ref: float | None, what: str) -> bool:
    ok = math.isfinite(price) and price > 0
    if ok and ref is not None and ref > 0:
        ok = abs(price / ref - 1) <= ctx.settings.max_price_jump
    if not ok:
        ctx.event(
            spec.book_id, "bad_price",
            f"A {coin(spec.symbol)} price from Binance looked wrong ({price:,.2f} against {ref or 0:,.2f}), "
            "so the book skipped this check and did not trade on it.",
            key=f"{what}:{price!r}", price=price, reference=ref,
        )  # fmt: skip
    return ok


def _is_stale(ctx: TickContext, spec: BookSpec, latest: pd.Timestamp) -> bool:
    age = ctx.now - (latest + spec.tf)
    if age <= spec.tf * ctx.settings.stale_after_bars:
        return False
    ctx.event(
        spec.book_id, "stale_data",
        f"The newest complete {coin(spec.symbol)} {spec.timeframe} bar ended at {latest + spec.tf:%Y-%m-%d %H:%M} "
        "UTC, which is too old to trade on. The book kept its position and will try again.",
        key=iso(latest), latest_bar=iso(latest),
    )  # fmt: skip
    return True


def _run_strategy(ctx: TickContext, spec: BookSpec, window: pd.DataFrame) -> Decision:
    other = None
    if spec.strategy.other_symbol:
        _, other_bars = closed_bars(ctx, spec, spec.strategy.other_symbol)
        other = other_window(other_bars, window)
    return normalize_decision(spec.strategy.decide(window, other))


def _paused_decision(paused_until: pd.Timestamp, raw: Decision) -> Decision:
    text = (
        "The book is paused after hitting its daily loss limit, so it stays in cash until "
        f"{paused_until:%Y-%m-%d %H:%M} UTC."
    )
    return Decision(0.0, text, {"strategy_target": raw.target, "strategy_explanation": raw.explanation})


def _catch_up(spec: BookSpec, last: pd.Timestamp | None, latest: pd.Timestamp) -> tuple[int, tuple[EventSpec, ...]]:
    skipped = 0 if last is None else max(round((latest - last) / spec.tf) - 1, 0)
    if not skipped:
        return 0, ()
    text = (
        f"{skipped} {spec.timeframe} decision bar(s) passed without a check (the Mac was asleep or offline). "
        "The book acted on the latest closed bar only; missed bars are not replayed."
    )
    return skipped, (EventSpec("catch_up", text, {"skipped_bars": skipped}, iso(latest)),)


def _fill_text(spec: BookSpec, fill) -> str:
    verb = "bought" if fill.side == "buy" else "sold"
    return f"{spec.title}: paper {verb} {abs(fill.units_delta):.6f} {coin(spec.symbol)} at ${fill.price:,.2f}."


class _Skip(Exception):
    """A guard decided this bar cannot be traded on this tick; ``status`` says why."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


@dataclass(frozen=True)
class ReadyBar:
    latest: pd.Timestamp  # the decision bar (latest closed bar of the book's timeframe)
    fill_ts: pd.Timestamp  # the next bar's open time: where the decision is filled
    price: float  # that open
    close: float  # the decision bar's close
    window: pd.DataFrame  # what decide() sees


def ready_bar(ctx: TickContext, spec: BookSpec, last: pd.Timestamp | None) -> ReadyBar:
    """The new decision bar and its fill, or raise _Skip with the reason it cannot be traded."""
    frame, bars = closed_bars(ctx, spec, spec.symbol)
    if bars.empty:
        raise _Skip("waiting_for_data")
    latest = bars.index[-1]
    if last is not None and latest <= last:
        raise _Skip("up_to_date")
    if _is_stale(ctx, spec, latest):
        raise _Skip("stale_data")
    if latest < latest_due_bar(ctx.now, spec.tf, ctx.settle):
        # The newest bar is missing 5m data (logged as a gap). Deciding on an older bar now
        # would fill at a price from the past, so wait for the next complete bar instead.
        raise _Skip("bar_incomplete")
    fill_ts = latest + spec.tf
    if fill_ts not in frame.index:
        raise _Skip("waiting_for_next_bar")
    price, close = float(frame.at[fill_ts, "open"]), float(bars.at[latest, "close"])
    if not _price_ok(ctx, spec, price, close, "fill"):
        raise _Skip("bad_price")
    window = decision_window(bars, spec.strategy)
    if window is None:
        raise _Skip("warming_up")
    return ReadyBar(latest, fill_ts, price, close, window)


def decide_book(ctx: TickContext, spec: BookSpec, book: dict) -> str:
    """Process the latest closed decision bar, if it is new. Returns a short status word."""
    last = parse_ts(book["last_bar_ts"])
    try:
        bar = ready_bar(ctx, spec, last)
    except _Skip as skip:
        return skip.status
    decision = _run_strategy(ctx, spec, bar.window)
    paused = parse_ts(book["paused_until"])
    if paused is not None and ctx.now < paused:
        decision = _paused_decision(paused, decision)
    skipped, events = _catch_up(spec, last, bar.latest)
    commit = ctx.store.commit_bar(
        spec.book_id,
        BarCommit(bar.latest, bar.fill_ts, decision, bar.price, bar.close, spec.cost_bps_per_side, ctx.now,
                  spec.strategy.meta.version, skipped, events),
    )  # fmt: skip
    if not commit.claimed:
        return "already_done"
    if commit.fill is not None:
        ctx.notifier.queue(ctx.store, spec.book_id, _fill_text(spec, commit.fill), ctx.now)
        return "traded"
    return "held"


def _kill_if_needed(ctx: TickContext, spec: BookSpec, mark: dict, ts: pd.Timestamp, frame: pd.DataFrame) -> bool:
    limit, start = spec.max_daily_loss, mark["day_start_equity"]
    paused = parse_ts(mark["book"]["paused_until"])
    if not 0 < limit < 1 or not start or (paused is not None and ctx.now < paused):
        return False
    change = mark["equity"] / start - 1
    if change > -limit or ts not in frame.index:  # the next 5m bar must have started: it is the fill
        return False
    resume = ts.floor("D") + pd.Timedelta(days=1)
    text = (
        f"{spec.title} lost {-change:.1%} today, past its {limit:.0%} daily limit. It sold to cash and is "
        f"paused until {resume:%Y-%m-%d %H:%M} UTC."
    )
    kill = {
        "now": ctx.now, "price": float(frame.at[ts, "open"]), "fill_ts": ts, "paused_until": resume,
        "cost_bps": spec.cost_bps_per_side,
        "event": EventSpec("kill_switch", text, {"day_change": change, "limit": limit}, ts.strftime("%Y-%m-%d")),
    }  # fmt: skip
    if not ctx.store.trigger_kill_switch(spec.book_id, kill).claimed:
        return False
    ctx.notifier.queue(ctx.store, spec.book_id, text, ctx.now)
    return True


def mark_book(ctx: TickContext, spec: BookSpec) -> dict:
    """Mark at the latest closed 5m close (once per bar), then apply the daily loss limit."""
    frame = bars_5m(ctx, spec.symbol, ctx.now.floor(FIVE_MIN) - MARK_LOOKBACK)
    closed, _ = split_closed(frame, ctx.now, ctx.settle)
    if closed.empty:
        return {"marked": False}
    ts, price = closed.index[-1] + FIVE_MIN, float(closed["close"].iloc[-1])
    book = ctx.store.book(spec.book_id)
    last_ts = parse_ts(book["last_mark_ts"])
    if last_ts is not None and ts <= last_ts:
        return {"marked": False}
    near = last_ts is not None and ts - last_ts <= MARK_JUMP_WINDOW
    if not _price_ok(ctx, spec, price, book["last_mark_price"] if near else None, "mark"):
        return {"marked": False}
    mark = ctx.store.record_mark(spec.book_id, ts, price, ctx.now)
    killed = _kill_if_needed(ctx, spec, mark, ts, frame)
    return {"marked": True, "equity": round(mark["equity"], 2), "kill_switch": killed}


def _resume_if_due(ctx: TickContext, spec: BookSpec) -> None:
    text = "The daily loss pause ended; the book follows its strategy again."
    ctx.store.resume_if_due(spec.book_id, ctx.now, EventSpec("resumed", text, {}, iso(ctx.now.floor("D"))))


def process_book(ctx: TickContext, spec: BookSpec) -> dict:
    _resume_if_due(ctx, spec)
    book = ctx.store.book(spec.book_id)
    last = parse_ts(book["last_bar_ts"])
    status = "up_to_date"
    if last is None or latest_due_bar(ctx.now, spec.tf, ctx.settle) > last:
        status = decide_book(ctx, spec, book)
    return {"book_id": spec.book_id, "decision": status, **mark_book(ctx, spec)}


def _bucket(now: pd.Timestamp, minutes: int = 10) -> str:
    return iso(now.floor(f"{minutes}min"))


def safe_process(ctx: TickContext, spec: BookSpec) -> dict:
    """process_book, with every failure turned into a plain-language event (never a crash)."""
    base = {"book_id": spec.book_id, "ok": False}
    try:
        return {**process_book(ctx, spec), "ok": True}
    except DataUnavailable as exc:
        log.warning("market data unavailable", extra={"book_id": spec.book_id, "error": str(exc)})
        msg = f"Couldn't get {coin(spec.symbol)} prices from Binance. Nothing was traded; it will retry shortly."
        ctx.event(spec.book_id, "data_error", msg, key=_bucket(ctx.now), error=type(exc).__name__)
        return {**base, "decision": "data_error"}
    except InvalidTarget as exc:
        log.error("strategy returned an invalid target", extra={"book_id": spec.book_id, "error": str(exc)})
        msg = "The strategy gave an answer the runner cannot trade (outside 0-100% invested). Nothing was traded."
        ctx.event(spec.book_id, "strategy_error", msg, key=_bucket(ctx.now))
        return {**base, "decision": "strategy_error"}
    except Exception as exc:
        log.exception("book check failed", extra={"book_id": spec.book_id})
        msg = "Something went wrong while checking this book. Nothing was traded for this bar; it will retry."
        ctx.event(spec.book_id, "error", msg, key=_bucket(ctx.now), error=type(exc).__name__)
        return {**base, "decision": "error"}


def _exchange_time(store: Store, source: BarSource, local: pd.Timestamp) -> pd.Timestamp | None:
    try:
        now = source.server_time()
    except DataUnavailable as exc:
        log.warning("exchange clock unavailable", extra={"error": str(exc)})
        msg = "Couldn't reach Binance. No book was checked; the runner will retry with a growing delay."
        store.add_event(None, EventSpec("data_error", msg, {"error": type(exc).__name__}, _bucket(local)), local)
        store.beat("runner", "offline", local, {"reason": "exchange_unreachable"})
        return None
    skew = (local - now).total_seconds()
    if abs(skew) > SKEW_WARN_S:
        side = "ahead of" if skew > 0 else "behind"
        msg = f"This Mac's clock is {abs(skew):.0f}s {side} Binance. Bars are judged on Binance's clock."
        store.add_event(None, EventSpec("clock_skew", msg, {"skew_s": skew}, iso(local.floor("h"))), local)
    return now


def tick(
    config: LiveConfig, store: Store, source: BarSource, notifier: Notifier | None = None,
    local_now: pd.Timestamp | None = None,
) -> dict:  # fmt: skip
    """One pass over every enabled book. Safe to call any number of times, from any process."""
    settings = config.settings
    notifier = notifier or Notifier(settings.notifications, settings.notify_min_interval_s)
    local = local_now or pd.Timestamp.now(tz="UTC")
    now = _exchange_time(store, source, local)
    if now is None:
        return {"ok": False, "status": "exchange_unreachable", "books": []}
    ctx = TickContext(store, source, settings, notifier, now)
    books = [spec for spec in config.books if spec.enabled]
    for spec in books:
        store.upsert_book(spec, now)
    results = [safe_process(ctx, spec) for spec in books]
    notifier.flush(store, now, {spec.book_id: spec.title for spec in books})
    ok = all(r["ok"] for r in results)
    detail = {"exchange_time": iso(now), "clock_skew_s": round((local - now).total_seconds(), 3), "n_books": len(books)}
    store.beat("runner", "ok" if ok else "degraded", local, detail)
    return {"ok": ok, "status": "ok" if ok else "degraded", **detail, "books": results}
