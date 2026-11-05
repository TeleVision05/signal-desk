"""Paper driver for promoted walk-forward ML TQQQ strategy."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .daily_backtest import StopConfig
from .data import fetch_ohlcv
from .features import build_features, make_labels
from .ledger import PaperLedger
from .ml_walkforward import WFConfig, walk_forward_signals
from .notify import send_trade_alert
from .paper_step import BookSpec, step

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
SPEC = BookSpec(
    run_id="ml-tqqq-hgb-paper-v1",
    book="tqqq",
    strategy="ml_TQQQ_hgb_h5_t0.55",
    asset="TQQQ",
    cost_bps=PROMOTED.cost_bps,
    stop=StopConfig(hard_pct=PROMOTED.hard_pct, trailing_pct=PROMOTED.trailing_pct),
    mark_source="yfinance",
    threshold=float(PROMOTED.proba_threshold),
    config=PROMOTED.__dict__,
)


def missing_inputs(main: pd.DataFrame, extras: dict[str, pd.Series | None]) -> str | None:
    """Why the inputs can't be trusted for the newest bar, or None if every series reaches it.

    Yahoo sometimes fails a download or returns a series a day short. The model then scores
    the same bar differently (Oct 30 2026: 0.37 to 0.66), so such a sync must not decide.
    """
    last = main.index[-1]
    for name, s in extras.items():
        if s is None or s.dropna().empty:
            return f"{name.upper()} prices could not be downloaded"
        if s.dropna().index[-1] < last:
            return f"{name.upper()} prices stop at {s.dropna().index[-1].date()}, before {last.date()}"
    return None


def score_latest(ohlc: pd.DataFrame, extras: dict[str, pd.Series], cfg: WFConfig) -> tuple[int | None, float | None]:
    """The model's (signal, p_up) for the newest bar of ``ohlc``; (None, None) if it has no score."""
    feat = build_features(ohlc, extras=extras)
    y = make_labels(ohlc["Close"], horizon=cfg.horizon, mode=cfg.label_mode)
    signal, proba = walk_forward_signals(ohlc, feat, y, cfg)
    p = proba.iloc[-1]
    if pd.isna(p):  # no score for the newest bar; never fall back to an older bar's score
        return None, None
    return int(signal.iloc[-1]), float(p)


def _fetch_close(ticker: str, start: str) -> pd.Series | None:
    try:
        return fetch_ohlcv(ticker, start=start)["Close"]
    except Exception as exc:
        print(f"[paper] {ticker} download failed: {exc}", flush=True)
        return None


def sync_ml_paper(
    run_id: str = SPEC.run_id,
    *,
    initial_cash: float = 1000.0,
    db_path=None,
) -> dict:
    cfg = PROMOTED
    spec = SPEC if run_id == SPEC.run_id else BookSpec(**{**SPEC.__dict__, "run_id": run_id})
    tqqq = fetch_ohlcv("TQQQ", start=max(cfg.start, "2010-02-11"))
    extras = {
        "spy": _fetch_close("SPY", cfg.start),
        "qqq": _fetch_close("QQQ", cfg.start),
        "vix": _fetch_close("^VIX", cfg.start),
    }
    problem = missing_inputs(tqqq, extras)
    want, p = (None, None) if problem else score_latest(tqqq, extras, cfg)
    ledger = PaperLedger(db_path) if db_path else PaperLedger()
    info = step(spec, tqqq, want, p, ledger=ledger, initial_cash=initial_cash, problem=problem)
    publish(info, OUT / "ml_paper_last.json")
    return info


def publish(info: dict, path: Path) -> None:
    """Write the dashboard's last-sync file (decisions only) and email on a real trade."""
    if info.get("action") == "skipped":
        print(f"[paper] {info['book']} {info['bar_date']} not decided: {info['problem']}", flush=True)
        return
    OUT.mkdir(exist_ok=True)
    path.write_text(json.dumps(info, indent=2, default=str))
    if not info.get("traded"):
        return
    try:
        send_trade_alert(info)
    except Exception as exc:
        print(f"[notify] hook failed: {exc}", flush=True)
