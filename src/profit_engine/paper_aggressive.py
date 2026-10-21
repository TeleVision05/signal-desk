"""Paper driver for promoted aggressive TQQQ momentum + stop-loss strategy."""

from __future__ import annotations

from datetime import datetime, timezone

from .aggressive import weekly_momentum
from .daily_backtest import StopConfig, _stop_hit
from .data import fetch_ohlcv
from .ledger import PaperLedger


# Frozen promoted config (do not retune on paper)
TICKER = "TQQQ"
LOOKBACK = 10
HARD_PCT = 0.10
TRAIL_PCT = 0.15
COST_BPS = 10.0


def sync_aggressive_paper(
    run_id: str = "tqqq-mom10-paper-v1",
    *,
    initial_cash: float = 100_000.0,
    db_path=None,
) -> dict:
    df = fetch_ohlcv(TICKER, start="2015-01-01")
    sig = weekly_momentum(df["Close"], LOOKBACK, 0.0)
    want = int(sig.iloc[-1])
    bar = df.iloc[-1]
    mark = float(bar["Close"])
    open_px = float(bar["Open"])
    ts = df.index[-1].isoformat()

    ledger = PaperLedger(db_path) if db_path else PaperLedger()
    try:
        pos = ledger.get_position(run_id)
    except KeyError:
        ledger.start_run(
            run_id,
            "tqqq_mom10_hard10_trail15",
            {
                "ticker": TICKER,
                "lookback": LOOKBACK,
                "hard_pct": HARD_PCT,
                "trailing_pct": TRAIL_PCT,
                "cost_bps": COST_BPS,
            },
            initial_cash=initial_cash,
        )
        pos = ledger.get_position(run_id)

    cash = float(pos["cash"])
    shares = float(pos["shares"])
    asset = pos["asset"]
    half = COST_BPS / 10_000.0 / 2.0
    stop = StopConfig(hard_pct=HARD_PCT, trailing_pct=TRAIL_PCT)

    # Track peak in a simple side table via equity marks: use high watermark from marks
    # For paper, approximate peak as max(entry proxy, today's high).
    entry_px = float(pos.get("entry_px") or 0)  # may be absent
    # Store entry on buy via fill price; recover from last BUY if needed
    if shares > 0 and entry_px <= 0:
        with ledger._conn() as conn:
            row = conn.execute(
                "SELECT price FROM fills WHERE run_id=? AND side='BUY' ORDER BY id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            entry_px = float(row["price"]) if row else mark

    peak = max(entry_px, float(bar["High"])) if shares > 0 else 0.0
    action = "hold"

    if shares > 0 and asset == TICKER:
        hit, exit_px = _stop_hit(bar, entry_px, peak, stop)
        if hit and exit_px is not None:
            cost = shares * exit_px * half
            ledger.apply_fill(
                run_id,
                side="SELL",
                asset=TICKER,
                shares=shares,
                price=exit_px,
                cost=cost,
                source="yfinance-stop",
                ts=ts,
            )
            action = "stop_exit"
            shares = 0.0
            cash = float(ledger.get_position(run_id)["cash"])

    pos = ledger.get_position(run_id)
    cash = float(pos["cash"])
    shares = float(pos["shares"])

    if shares > 0 and want == 0:
        cost = shares * open_px * half
        ledger.apply_fill(
            run_id,
            side="SELL",
            asset=TICKER,
            shares=shares,
            price=open_px,
            cost=cost,
            source="yfinance-signal",
            ts=ts,
        )
        action = "signal_exit"
    elif shares == 0 and want == 1:
        cost = cash * half
        investable = cash - cost
        new_shares = investable / open_px
        ledger.apply_fill(
            run_id,
            side="BUY",
            asset=TICKER,
            shares=new_shares,
            price=open_px,
            cost=cost,
            source="yfinance-signal",
            ts=ts,
        )
        action = "signal_entry"

    pos = ledger.get_position(run_id)
    mark_px = mark if pos["asset"] == TICKER else None
    equity = ledger.mark(run_id, mark_price=mark_px or 0.0, source="yfinance", ts=ts)
    return {
        "run_id": run_id,
        "strategy": "tqqq_mom10_hard10_trail15",
        "signal": want,
        "action": action,
        "equity": equity,
        "cash": pos["cash"],
        "asset": pos["asset"],
        "shares": pos["shares"],
        "mark": mark,
        "hard_pct": HARD_PCT,
        "trailing_pct": TRAIL_PCT,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }
