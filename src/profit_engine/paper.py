"""Paper driver: apply latest GEM signal into the fake-money ledger using live Yahoo data."""

from __future__ import annotations

from datetime import datetime, timezone

from .data import fetch_panel, month_end_closes
from .ledger import PaperLedger
from .strategies import gem_dual_momentum


def sync_gem_paper(
    run_id: str = "gem-qqq-paper-v1",
    *,
    initial_cash: float = 100_000.0,
    round_trip_cost_bps: float = 7.5,
    db_path=None,
    risk_on: tuple[str, str] = ("QQQ", "VEU"),
    risk_off: str = "AGG",
) -> dict:
    """Default paper book: QQQ/VEU dual momentum (beats SPY + CPI in our locked tests)."""
    tickers = list(risk_on) + [risk_off]
    panel = fetch_panel(tickers, start="2007-01-01")
    month_closes = month_end_closes(panel)
    signals = gem_dual_momentum(month_closes, risk_on=risk_on, risk_off=risk_off)
    target = str(signals.iloc[-1])
    signal_date = signals.index[-1]

    # Fill at latest available open (proxy for next-open if signal already aged)
    px = float(panel[target]["Open"].iloc[-1])
    mark = float(panel[target]["Close"].iloc[-1])
    fill_ts = panel[target].index[-1].isoformat()

    ledger = PaperLedger(db_path) if db_path else PaperLedger()
    try:
        pos = ledger.get_position(run_id)
    except KeyError:
        ledger.start_run(
            run_id,
            "gem_qqq_veu",
            {
                "lookback": 12,
                "absolute_cutoff": 0.0,
                "risk_on": list(risk_on),
                "risk_off": risk_off,
                "universe": tickers,
                "round_trip_cost_bps": round_trip_cost_bps,
            },
            initial_cash=initial_cash,
        )
        pos = ledger.get_position(run_id)

    cur_asset = pos["asset"]
    cur_shares = float(pos["shares"])
    cash = float(pos["cash"])
    half = round_trip_cost_bps / 10_000.0 / 2.0

    if cur_asset != target:
        if cur_asset and cur_shares > 0:
            exit_px = float(panel[cur_asset]["Open"].iloc[-1])
            cost = cur_shares * exit_px * half
            ledger.apply_fill(
                run_id,
                side="SELL",
                asset=cur_asset,
                shares=cur_shares,
                price=exit_px,
                cost=cost,
                source="yfinance",
                ts=fill_ts,
            )
            pos = ledger.get_position(run_id)
            cash = float(pos["cash"])

        cost = cash * half
        investable = cash - cost
        shares = investable / px
        ledger.apply_fill(
            run_id,
            side="BUY",
            asset=target,
            shares=shares,
            price=px,
            cost=cost,
            source="yfinance",
            ts=fill_ts,
        )

    equity = ledger.mark(run_id, mark_price=mark, source="yfinance", ts=fill_ts)
    pos = ledger.get_position(run_id)
    return {
        "run_id": run_id,
        "signal_date": str(signal_date.date()),
        "target": target,
        "equity": equity,
        "cash": pos["cash"],
        "asset": pos["asset"],
        "shares": pos["shares"],
        "mark": mark,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }