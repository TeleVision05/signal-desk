"""One pass over the paper books: :func:`tick`.

Per enabled book:

1. If no decision bar can have closed since the last one processed (judged on the
   EXCHANGE's clock), skip straight to marking. Otherwise fetch recent 5m bars, drop the
   in-progress ones (:func:`market.split_closed`), resample with the harness ``resample``
   (``incomplete="drop"``) and take the latest CLOSED decision bar.
2. Guards: stale feed (latest closed bar more than ``stale_after_bars`` timeframes old, checked
   even when that bar was already processed, so a frozen feed is never silent) and price
   sanity skip with an event; the fill price must exist. The history fetched is widened (up to
   :data:`MAX_WIDEN` doublings) until it holds ``warmup_bars`` complete bars, so a Binance data
   gap never stops a warmed-up book from deciding; if it still cannot, a ``warming_up`` event
   says so.
3. Decide on the latest closed bar ONLY and fill at the open of the bar now in progress, as
   the harness fills at the next bar that exists: right after the decision bar normally, or
   after bars dropped for missing 5m data. A halt is judged like the harness, on the whole fill
   bar: while none of its 5m bars seen so far has volume, the book waits (``halted`` event once
   a closed 5m bar shows it); as soon as one trades, it fills at the fill bar's open, which is
   the harness price. A fill bar that ends with no trading is skipped and the next decision
   supersedes it. Decision bars not acted on are counted in a ``catch_up`` event and never
   traded; a stateful strategy's own state (the trend band) is first run through them without
   trading (:mod:`catchup`), so it decides from where the research would be. Cost
   ``cost_bps_per_side`` on the traded fraction. All of it is one :meth:`Store.commit_bar`
   transaction, so the bar is traded at most once.
4. Mark equity at the latest closed 5m close. If equity is down ``max_daily_loss`` from the
   equity at 00:00 UTC (the 23:55 bar's close, even if the Mac slept through midnight), go
   flat at the next 5m open and pause until next UTC midnight. A breach seen before that
   next 5m bar exists is re-checked on later ticks of the same bar.

A book whose stored coin or timeframe differs from books.json is not touched
(``config_conflict`` event): its position belongs to the old market.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import pandas as pd

from profit_engine.intraday.data import find_gaps
from profit_engine.intraday.resample import resample
from profit_engine.live import catchup
from profit_engine.live.accounting import equity
from profit_engine.live.market import FIVE_MIN, BarSource, DataUnavailable, split_closed
from profit_engine.live.notify import Notifier
from profit_engine.live.registry import BookSpec, LiveConfig, Settings
from profit_engine.live.store import BarCommit, BookConflict, EventSpec, Store, iso, parse_ts
from profit_engine.live.strategy import (
    Decision,
    InvalidTarget,
    StrategyUnavailable,
    call_decide,
    decision_window,
    normalize_decision,
    other_window,
)

log = logging.getLogger("signal_desk.runner")
MARK_LOOKBACK = pd.Timedelta("30min")
MARK_JUMP_WINDOW = pd.Timedelta("1h")
SKEW_WARN_S = 60.0
MAX_WIDEN = 3  # history is doubled at most this many times to find warmup_bars complete bars
UNHEALTHY = {"stale_data", "model_unavailable"}  # statuses that make the runner heartbeat "degraded"


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


def _resampled(ctx: TickContext, spec: BookSpec, symbol: str, since: pd.Timestamp):
    frame = bars_5m(ctx, symbol, since)
    closed, _ = split_closed(frame, ctx.now, ctx.settle)
    bars = closed if closed.empty else resample(closed, spec.timeframe, incomplete="drop")
    return frame, closed, bars


def closed_bars(
    ctx: TickContext, spec: BookSpec, symbol: str, since: pd.Timestamp | None = None, need: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(5m frame incl. in-progress bar, closed decision bars of the book's timeframe).

    Without ``since``, the lookback is widened until it holds ``need`` (default ``warmup_bars``)
    complete bars (a data gap drops incomplete bars), or a wider fetch adds no older bar, or MAX_WIDEN."""
    if since is not None:
        frame, _, bars = _resampled(ctx, spec, symbol, since)
        return frame, bars
    n = need or max(int(spec.strategy.warmup_bars), 1)
    span, first = n + max(4, n // 50), None  # a 2% margin absorbs the odd gap without a doubling
    for _ in range(MAX_WIDEN + 1):
        since = ctx.now.floor(spec.tf) - spec.tf * span
        frame, closed, bars = _resampled(ctx, spec, symbol, since)
        if len(bars) >= n or frame.empty or frame.index[0] == first:  # enough, or nothing older exists
            break
        span, first = span * 2, frame.index[0]
    if symbol == spec.symbol:
        _note_gaps(ctx, spec, closed)
    return frame, bars


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


def _run_strategy(ctx: TickContext, spec: BookSpec, bar: ReadyBar, last: pd.Timestamp | None) -> Decision:
    """Decide on ``bar.window``. A stateful strategy continues from its own last persisted target,
    first run through the bars it did not decide (:func:`catchup.advance`); details say how many."""
    s, other_bars, other = spec.strategy, None, None
    if s.other_symbol:
        since = (bar.bars if catchup.state_bars(s) else bar.window).index[0]
        _, other_bars = closed_bars(ctx, spec, s.other_symbol, since=since)
        other = other_window(other_bars, bar.window)
    prev = ctx.store.strategy_target(spec.book_id) if getattr(s, "stateful", False) else 0.0
    prev, replayed = catchup.advance(s, bar.bars, last, prev, other_bars)
    decision = normalize_decision(call_decide(s, bar.window, other, prev))
    if not replayed:
        return decision
    return Decision(decision.target, decision.explanation, {**decision.details, "replayed_bars": replayed})


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
    fill_ts: pd.Timestamp  # open time of the bar now in progress: where the decision is filled
    price: float  # that open
    close: float  # the decision bar's close
    window: pd.DataFrame  # what decide() sees
    skipped: int  # decision bars since the last decision that were not acted on
    bars: pd.DataFrame  # every closed bar fetched (the window and the bars a stateful strategy replays)


def _halted(ctx: TickContext, spec: BookSpec, frame: pd.DataFrame, latest: pd.Timestamp, fill_ts) -> str | None:
    """None if the fill bar has traded (any 5m bar of it with volume), else why the book waits.

    The harness defers a fill only when the whole fill bar has no volume, and otherwise fills
    at its open, so the runner waits while no 5m bar of the fill bar has traded yet."""
    if "volume" not in frame.columns:
        return None
    seen = frame[(frame.index >= fill_ts) & (frame.index < fill_ts + spec.tf)]
    if (seen["volume"] > 0).any():
        return None
    closed, _ = split_closed(seen, ctx.now, ctx.settle)
    if closed.empty:
        return "waiting_for_trades"  # only the bar in progress, seconds old: no trades yet is normal
    ctx.event(
        spec.book_id, "halted",
        f"Binance shows no {coin(spec.symbol)} trading since {fill_ts:%Y-%m-%d %H:%M} UTC (an exchange halt), so "
        f"the book has not filled yet. It fills at that {spec.timeframe} bar's open once trading resumes within "
        "it, like the backtest; otherwise it keeps its position and acts on the next bar.",
        key=iso(fill_ts), decision_bar=iso(latest), fill_bar=iso(fill_ts),
    )  # fmt: skip
    return "halted"


def _warming_up(ctx: TickContext, spec: BookSpec, bars: pd.DataFrame) -> None:
    need = max(int(spec.strategy.warmup_bars), 1)
    ctx.event(
        spec.book_id, "warming_up",
        f"The strategy needs {need} complete {spec.timeframe} bars of {coin(spec.symbol)} history and Binance has "
        f"{len(bars)} recent ones, so the book kept its position and did not decide on this bar.",
        key=iso(bars.index[-1]), need=need, have=len(bars),
    )  # fmt: skip


def ready_bar(ctx: TickContext, spec: BookSpec, last: pd.Timestamp | None) -> ReadyBar:
    """The new decision bar and its fill, or raise _Skip with the reason it cannot be traded."""
    need = catchup.history_needed(spec.strategy, last, latest_due_bar(ctx.now, spec.tf, ctx.settle), spec.tf)
    frame, bars = closed_bars(ctx, spec, spec.symbol, need=need)
    if bars.empty:
        raise _Skip("waiting_for_data")
    latest = bars.index[-1]
    if _is_stale(ctx, spec, latest):  # before "up to date": a frozen feed must not look healthy
        raise _Skip("stale_data")
    if last is not None and latest <= last:
        raise _Skip("up_to_date")
    # The bar in progress. Bars between it and ``latest`` were dropped for missing 5m data (a
    # logged gap); like the harness, the decision fills at the next bar that exists.
    fill_ts = latest_due_bar(ctx.now, spec.tf, ctx.settle) + spec.tf
    if fill_ts not in frame.index:
        raise _Skip("waiting_for_next_bar")
    halted = _halted(ctx, spec, frame, latest, fill_ts)
    if halted:
        raise _Skip(halted)
    price, close = float(frame.at[fill_ts, "open"]), float(bars.at[latest, "close"])
    if not _price_ok(ctx, spec, price, close, "fill"):
        raise _Skip("bad_price")
    window = decision_window(bars, spec.strategy)
    if window is None:
        _warming_up(ctx, spec, bars)
        raise _Skip("warming_up")
    skipped = catchup.skipped(spec.tf, frame, bars, last, latest)
    return ReadyBar(latest, fill_ts, price, close, window, skipped, bars)


def decide_book(ctx: TickContext, spec: BookSpec, book: dict) -> str:
    """Process the latest closed decision bar, if it is new. Returns a short status word.

    A daily-loss pause is applied inside :meth:`Store.commit_bar`, on the book row read in the
    same transaction, so a pause set by another process a moment ago is never missed."""
    last = parse_ts(book["last_bar_ts"])
    try:
        if getattr(spec.strategy, "problem", ""):  # e.g. no trained model: say so before fetching history
            raise StrategyUnavailable(spec.strategy.problem)
        bar = ready_bar(ctx, spec, last)
        decision = _run_strategy(ctx, spec, bar, last)
    except _Skip as skip:
        return skip.status
    except StrategyUnavailable as exc:
        detail = getattr(spec.strategy, "problem_detail", "")  # technical detail: the log only
        log.warning("strategy unavailable", extra={"book_id": spec.book_id, "error": str(exc), "detail": detail})
        ctx.event(spec.book_id, "model_unavailable", str(exc), key=_bucket(ctx.now, 60))
        return "model_unavailable"
    events = catchup.catch_up_events(spec.timeframe, bar.skipped, decision.details.get("replayed_bars", 0), bar.latest)
    commit = ctx.store.commit_bar(
        spec.book_id,
        BarCommit(bar.latest, bar.fill_ts, decision, bar.price, bar.close, spec.cost_bps_per_side, ctx.now,
                  spec.strategy.meta.version, bar.skipped, events),
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


def _midnight_price(ctx: TickContext, spec: BookSpec, closed: pd.DataFrame, book: dict, ts) -> float | None:
    """The 00:00 UTC price of ``ts``'s day (the 23:55 bar's close) when this mark starts a new
    UTC day for the book; None otherwise, or when Binance has no 23:55 bar."""
    if book["day"] is None or book["day"] == ts.strftime("%Y-%m-%d"):
        return None
    bar = ts.floor("D") - FIVE_MIN
    if bar not in closed.index:
        return None
    price = float(closed.at[bar, "close"])
    return price if _price_ok(ctx, spec, price, None, "midnight") else None


def _recheck_kill(ctx: TickContext, spec: BookSpec, book: dict, ts: pd.Timestamp, frame: pd.DataFrame) -> dict:
    """Same 5m bar as the last mark: re-run the daily loss check, whose fill bar may only have
    appeared since (the check needs the next 5m open)."""
    price = book["last_mark_price"]
    if price is None or ts not in frame.index:
        return {"marked": False}
    mark = {"equity": equity(book["cash"], book["units"], price), "day_start_equity": book["day_start_equity"],
            "book": book}  # fmt: skip
    return {"marked": False, "kill_switch": _kill_if_needed(ctx, spec, mark, ts, frame)}


def mark_book(ctx: TickContext, spec: BookSpec) -> dict:
    """Mark at the latest closed 5m close (once per bar), then apply the daily loss limit."""
    book = ctx.store.book(spec.book_id)
    since = ctx.now.floor(FIVE_MIN) - MARK_LOOKBACK
    if book["day"] is not None and book["day"] != (ctx.now - ctx.settle).floor(FIVE_MIN).strftime("%Y-%m-%d"):
        since = min(since, ctx.now.floor("D") - FIVE_MIN)  # a new UTC day: also fetch the 23:55 bar
    frame = bars_5m(ctx, spec.symbol, since)
    closed, _ = split_closed(frame, ctx.now, ctx.settle)
    if closed.empty:
        return {"marked": False}
    ts, price = closed.index[-1] + FIVE_MIN, float(closed["close"].iloc[-1])
    last_ts = parse_ts(book["last_mark_ts"])
    if last_ts is not None and ts == last_ts:
        return _recheck_kill(ctx, spec, book, ts, frame)
    if last_ts is not None and ts < last_ts:
        return {"marked": False}
    near = last_ts is not None and ts - last_ts <= MARK_JUMP_WINDOW
    if not _price_ok(ctx, spec, price, book["last_mark_price"] if near else None, "mark"):
        return {"marked": False}
    midnight = _midnight_price(ctx, spec, closed, book, ts)
    mark = ctx.store.record_mark(spec.book_id, ts, price, ctx.now, midnight)
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
        result = process_book(ctx, spec)
        return {**result, "ok": result["decision"] not in UNHEALTHY}
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


def _conflict(ctx: TickContext, spec: BookSpec, exc: BookConflict) -> dict:
    (was_sym, was_tf), (now_sym, now_tf) = exc.stored, exc.wanted
    log.error("book settings conflict with its history", extra={"book_id": spec.book_id, "error": str(exc)})
    msg = (
        f"{spec.title} was started as {coin(was_sym)} {was_tf} bars, but the books settings now give it "
        f"{coin(now_sym)} {now_tf}. It was not checked, so its paper position is never priced in the wrong market. "
        "Change it back, or give the new setup its own book_id."
    )
    ctx.event(spec.book_id, "config_conflict", msg, key=f"{was_sym}:{was_tf}->{now_sym}:{now_tf}")
    return {"book_id": spec.book_id, "ok": False, "decision": "config_conflict"}


def _register(ctx: TickContext, specs: tuple[BookSpec, ...]) -> tuple[list[BookSpec], list[dict]]:
    """Upsert every configured book (so turning one off shows on the dashboard) and turn off any
    book no longer in the settings; return the enabled books that are safe to run, and results
    for the ones refused."""
    ctx.store.turn_off_unlisted([spec.book_id for spec in specs], ctx.now)
    runnable, refused = [], []
    for spec in specs:
        try:
            ctx.store.upsert_book(spec, ctx.now)
        except BookConflict as exc:
            if spec.enabled:
                refused.append(_conflict(ctx, spec, exc))
            continue
        if spec.enabled:
            runnable.append(spec)
    return runnable, refused


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
    books, refused = _register(ctx, config.books)
    results = refused + [safe_process(ctx, spec) for spec in books]
    notifier.flush(store, now, {spec.book_id: spec.title for spec in books})
    ok = all(r["ok"] for r in results)
    detail = {
        "exchange_time": iso(now),
        "clock_skew_s": round((local - now).total_seconds(), 3),
        "n_books": len(results),
    }
    store.beat("runner", "ok" if ok else "degraded", local, detail)
    return {"ok": ok, "status": "ok" if ok else "degraded", **detail, "books": results}
