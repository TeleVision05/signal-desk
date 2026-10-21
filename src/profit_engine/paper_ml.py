"""Paper driver for promoted walk-forward ML TQQQ strategy."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .daily_backtest import StopConfig, _stop_hit
from .data import fetch_ohlcv
from .ledger import PaperLedger
from .ml_walkforward import WFConfig, run_ml_experiment, walk_forward_signals
from .features import build_features, make_labels

OUT = Path(__file__).resolve().parents[2] / "output"

# Frozen promoted config (HGB: strong OOS, better DD than logreg max-aggression cell)
PROMOTED = WFConfig(
    ticker="TQQQ",
    model="hgb",
    horizon=5,
    proba_threshold=0.55,
    hard_pct=0.08,
    trailing_pct=0.12,
    embargo=5,
)


def sync_ml_paper(
    run_id: str = "ml-tqqq-hgb-paper-v1",
    *,
    initial_cash: float = 1000.0,
    db_path=None,
) -> dict:
    cfg = PROMOTED
    # Refresh signals via full experiment path pieces
    qqq = fetch_ohlcv("QQQ", start=cfg.start)
    spy = fetch_ohlcv("SPY", start=cfg.start)
    tqqq = fetch_ohlcv("TQQQ", start=max(cfg.start, "2010-02-11"))
    try:
        vix = fetch_ohlcv("^VIX", start=cfg.start)["Close"]
    except Exception:
        vix = None
    extras = {"spy": spy["Close"], "qqq": qqq["Close"]}
    if vix is not None:
        extras["vix"] = vix
    feat = build_features(tqqq, extras=extras)
    y = make_labels(tqqq["Close"], horizon=cfg.horizon, mode=cfg.label_mode)
    signal, proba = walk_forward_signals(tqqq, feat, y, cfg)
    # Prefer the latest bar; if still empty, fall back to last scored day
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
    bar = tqqq.iloc[-1]
    mark = float(bar["Close"])
    open_px = float(bar["Open"])
    ts = tqqq.index[-1].isoformat()
    stop = StopConfig(hard_pct=cfg.hard_pct, trailing_pct=cfg.trailing_pct)
    half = cfg.cost_bps / 10_000 / 2

    ledger = PaperLedger(db_path) if db_path else PaperLedger()
    try:
        pos = ledger.get_position(run_id)
    except KeyError:
        ledger.start_run(
            run_id,
            "ml_TQQQ_hgb_h5_t0.55",
            cfg.__dict__,
            initial_cash=initial_cash,
        )
        pos = ledger.get_position(run_id)

    cash = float(pos["cash"])
    shares = float(pos["shares"])
    pos_before = {
        "cash": cash,
        "shares": shares,
        "asset": pos.get("asset"),
    }
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
                run_id, side="SELL", asset="TQQQ", shares=shares, price=exit_px, cost=cost, source="stop", ts=ts
            )
            action = "stop_exit"
            pos = ledger.get_position(run_id)
            cash, shares = float(pos["cash"]), float(pos["shares"])

    if shares > 0 and want == 0:
        cost = shares * open_px * half
        ledger.apply_fill(
            run_id, side="SELL", asset="TQQQ", shares=shares, price=open_px, cost=cost, source="ml_signal", ts=ts
        )
        action = "signal_exit"
    elif shares == 0 and want == 1:
        pos = ledger.get_position(run_id)
        cash = float(pos["cash"])
        cost = cash * half
        new_shares = (cash - cost) / open_px
        ledger.apply_fill(
            run_id, side="BUY", asset="TQQQ", shares=new_shares, price=open_px, cost=cost, source="ml_signal", ts=ts
        )
        action = "signal_entry"

    pos = ledger.get_position(run_id)
    mark_px = mark if pos["asset"] == "TQQQ" else None
    equity = ledger.mark(run_id, mark_price=mark_px or 0.0, source="yfinance", ts=ts)

    # Persist latest summary pointer
    OUT.mkdir(exist_ok=True)
    thr = float(cfg.proba_threshold)
    gap = (p - thr) if p is not None else None
    from .decisions import append_decision, explain_decision

    info = {
        "run_id": run_id,
        "strategy": "ml_TQQQ_hgb_h5_t0.55",
        "signal": want,
        "proba": p,
        "threshold": thr,
        "gap_to_threshold": gap,
        "comparison": (
            None
            if p is None
            else ("pass" if p >= thr else "fail")
        ),
        "bar_date": str(tqqq.index[-1].date()),
        "action": action,
        "equity": equity,
        "cash": pos["cash"],
        "asset": pos["asset"],
        "shares": pos["shares"],
        "mark": mark,
        "open": open_px,
        "pos_before": pos_before,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }
    info["explanation"] = explain_decision(info)
    append_decision(
        {
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
            "mark": mark,
            "explanation": info["explanation"],
        }
    )
    (OUT / "ml_paper_last.json").write_text(json.dumps(info, indent=2))
    try:
        from .notify import send_trade_alert

        send_trade_alert(info)
    except Exception as exc:  # noqa: BLE001
        print(f"[notify] hook failed: {exc}", flush=True)
    return info
