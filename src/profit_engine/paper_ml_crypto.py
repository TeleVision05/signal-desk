"""Paper-only driver for BTCUSDT walk-forward ML (24/7 second book).

Uses Binance *public* klines for prices and simulates fills in PaperLedger.
Never places live exchange orders.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .daily_backtest import StopConfig, _stop_hit
from .data import fetch_binance_klines
from .features import build_features, make_labels
from .ledger import PaperLedger
from .ml_walkforward import WFConfig, walk_forward_signals

OUT = Path(__file__).resolve().parents[2] / "output"

RUN_ID = "ml-btc-hgb-paper-v1"
STRATEGY = "ml_BTCUSDT_hgb_h5_t0.55"
ASSET = "BTCUSDT"

PROMOTED = WFConfig(
    ticker="BTCUSDT",
    model="hgb",
    horizon=5,
    proba_threshold=0.55,
    hard_pct=0.08,
    trailing_pct=0.12,
    cost_bps=20.0,
    start="2018-01-01",
    embargo=5,
)


def sync_crypto_paper(
    run_id: str = RUN_ID,
    *,
    initial_cash: float = 1000.0,
    db_path=None,
) -> dict:
    cfg = PROMOTED
    btc = fetch_binance_klines("BTCUSDT", start=cfg.start)
    extras: dict = {}
    try:
        eth = fetch_binance_klines("ETHUSDT", start=cfg.start)
        extras["eth"] = eth["Close"]
    except Exception as exc:  # noqa: BLE001
        print(f"[crypto] ETH extras skipped: {exc}", flush=True)

    feat = build_features(btc, extras=extras or None)
    y = make_labels(btc["Close"], horizon=cfg.horizon, mode=cfg.label_mode)
    signal, proba = walk_forward_signals(btc, feat, y, cfg)

    p_raw = proba.iloc[-1]
    if p_raw == p_raw:  # not NaN
        want = int(signal.iloc[-1])
        p = float(p_raw)
    else:
        valid = proba.dropna()
        if len(valid):
            p = float(valid.iloc[-1])
            want = int(signal.loc[valid.index[-1]])
        else:
            want, p = 0, None

    bar = btc.iloc[-1]
    mark = float(bar["Close"])
    open_px = float(bar["Open"])
    ts = btc.index[-1].isoformat()
    stop = StopConfig(hard_pct=cfg.hard_pct, trailing_pct=cfg.trailing_pct)
    half = cfg.cost_bps / 10_000 / 2

    ledger = PaperLedger(db_path) if db_path else PaperLedger()
    try:
        pos = ledger.get_position(run_id)
    except KeyError:
        ledger.start_run(run_id, STRATEGY, cfg.__dict__, initial_cash=initial_cash)
        pos = ledger.get_position(run_id)

    cash = float(pos["cash"])
    shares = float(pos["shares"])
    pos_before = {"cash": cash, "shares": shares, "asset": pos.get("asset")}
    action = "hold"
    entry_px = mark
    if shares > 0:
        buy_px = ledger.last_buy_price(run_id)
        if buy_px is not None:
            entry_px = buy_px
        peak = max(entry_px, float(bar["High"]))
        hit, exit_px = _stop_hit(bar, entry_px, peak, stop)
        if hit and exit_px is not None:
            cost = shares * exit_px * half
            ledger.apply_fill(
                run_id,
                side="SELL",
                asset=ASSET,
                shares=shares,
                price=exit_px,
                cost=cost,
                source="stop",
                ts=ts,
            )
            action = "stop_exit"
            pos = ledger.get_position(run_id)
            cash, shares = float(pos["cash"]), float(pos["shares"])

    if shares > 0 and want == 0:
        cost = shares * open_px * half
        ledger.apply_fill(
            run_id,
            side="SELL",
            asset=ASSET,
            shares=shares,
            price=open_px,
            cost=cost,
            source="ml_signal",
            ts=ts,
        )
        action = "signal_exit"
    elif shares == 0 and want == 1:
        pos = ledger.get_position(run_id)
        cash = float(pos["cash"])
        cost = cash * half
        new_shares = (cash - cost) / open_px
        ledger.apply_fill(
            run_id,
            side="BUY",
            asset=ASSET,
            shares=new_shares,
            price=open_px,
            cost=cost,
            source="ml_signal",
            ts=ts,
        )
        action = "signal_entry"

    pos = ledger.get_position(run_id)
    mark_px = mark if pos["asset"] == ASSET else None
    equity = ledger.mark(run_id, mark_price=mark_px or 0.0, source="binance", ts=ts)

    OUT.mkdir(exist_ok=True)
    thr = float(cfg.proba_threshold)
    gap = (p - thr) if p is not None else None
    from .decisions import append_decision, explain_decision

    info = {
        "run_id": run_id,
        "book": "btc",
        "strategy": STRATEGY,
        "signal": want,
        "proba": p,
        "threshold": thr,
        "gap_to_threshold": gap,
        "comparison": None if p is None else ("pass" if p >= thr else "fail"),
        "bar_date": str(btc.index[-1].date()),
        "action": action,
        "equity": equity,
        "cash": pos["cash"],
        "asset": pos["asset"] or ASSET,
        "shares": pos["shares"],
        "mark": mark,
        "open": open_px,
        "pos_before": pos_before,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }
    info["explanation"] = explain_decision(info)
    append_decision(
        {
            "run_id": run_id,
            "book": "btc",
            "bar_date": info["bar_date"],
            "synced_at": info["synced_at"],
            "proba": p,
            "threshold": thr,
            "gap_to_threshold": gap,
            "comparison": info["comparison"],
            "signal": want,
            "action": action,
            "equity": equity,
            "cash": pos["cash"],
            "shares": pos["shares"],
            "asset": info["asset"],
            "mark": mark,
            "explanation": info["explanation"],
        }
    )
    (OUT / "ml_crypto_paper_last.json").write_text(json.dumps(info, indent=2))
    try:
        from .notify import send_trade_alert

        send_trade_alert(info)
    except Exception as exc:  # noqa: BLE001
        print(f"[notify] crypto hook failed: {exc}", flush=True)
    return info
