"""Build a JSON snapshot the dashboard can serve without retraining."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .ledger import PaperLedger

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "output"
STATE_PATH = DATA / "dashboard_state.json"
SEED_PATH = ROOT / "dashboard" / "seed_state.json"
PROMO_SEED = ROOT / "dashboard" / "ml_promoted.seed.json"
RUN_ID = "ml-tqqq-hgb-paper-v1"


def _ledger_rows(run_id: str = RUN_ID) -> dict:
    return PaperLedger().snapshot(run_id)


def _seed() -> dict:
    if SEED_PATH.exists():
        return json.loads(SEED_PATH.read_text())
    return {}


def _merge_seed_defaults(state: dict) -> dict:
    """Fill missing chart/metrics from committed seed so first paint never looks empty."""
    seed = _seed()
    if not state.get("backtest") or not (state.get("backtest") or {}).get("equity_weekly"):
        state["backtest"] = seed.get("backtest") or {}
    if not state.get("promoted_metrics"):
        if PROMO_SEED.exists():
            state["promoted_metrics"] = json.loads(PROMO_SEED.read_text())
        else:
            state["promoted_metrics"] = seed.get("promoted_metrics") or {}
    if not state.get("strategy"):
        state["strategy"] = seed.get("strategy") or {}
    if state.get("paper") is None:
        state["paper"] = {}
    if state.get("ledger") is None:
        state["ledger"] = {"fills": [], "equity": [], "position": None, "run": None}
    if not state.get("decisions"):
        state["decisions"] = []
    state.setdefault("mode", "paper_only")
    return state


def build_state(*, include_backtest: bool = True) -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    ledger = _ledger_rows()
    promoted = {}
    for promo_path in (OUT / "ml_promoted.json", PROMO_SEED):
        if promo_path.exists():
            promoted = json.loads(promo_path.read_text())
            break

    backtest = {}
    if include_backtest:
        eq_csv = OUT / "ml_TQQQ_hgb_h5_t0.55_equity.csv"
        if not eq_csv.exists():
            matches = list(OUT.glob("ml_TQQQ_hgb_h5_t0.55*_equity.csv"))
            eq_csv = matches[0] if matches else None
        if eq_csv and Path(eq_csv).exists():
            eq = pd.read_csv(eq_csv, index_col=0, parse_dates=True).iloc[:, 0]
            eq_w = eq.resample("W-FRI").last().dropna()
            backtest = {
                "strategy": "ml_TQQQ_hgb_h5_t0.55",
                "equity_weekly": [
                    {"date": str(i.date()), "equity": float(v)} for i, v in eq_w.items()
                ],
                "start": str(eq.index[0].date()),
                "end": str(eq.index[-1].date()),
                "start_equity": float(eq.iloc[0]),
                "end_equity": float(eq.iloc[-1]),
                "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
            }
        else:
            backtest = (_seed().get("backtest") or {})

    paper_last = {}
    pl = OUT / "ml_paper_last.json"
    if pl.exists():
        paper_last = json.loads(pl.read_text())

    from .decisions import load_decisions
    from .notify import load_notify_status

    decisions = load_decisions(limit=60)
    notify_status = load_notify_status()

    findings = {}
    fm = ROOT / "FINDINGS_ML.md"
    if fm.exists():
        findings["path"] = str(fm)
        findings["summary"] = (
            "Walk-forward HGB on TQQQ, P>=0.55, 8% hard / 12% trail stops. "
            "Last 3m paper backtest on $1k ≈ +16%."
        )

    state = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "strategy": {
            "name": "ml_TQQQ_hgb_h5_t0.55",
            "ticker": "TQQQ",
            "model": "HistGradientBoosting",
            "horizon_days": 5,
            "proba_threshold": 0.55,
            "hard_stop_pct": 0.08,
            "trailing_stop_pct": 0.12,
            "train_days": 504,
            "features": [
                "returns 1/2/3/5/10/20/60d",
                "vol 10/20 + ratio",
                "dist to SMA 10/20/50/100/200",
                "RSI 7/14",
                "ATR%",
                "volume z-score",
                "Donchian 20/55",
                "SPY/QQQ/VIX cross features",
            ],
        },
        "paper": paper_last,
        "decisions": decisions,
        "notify": notify_status,
        "ledger": ledger,
        "backtest": backtest,
        "promoted_metrics": promoted,
        "findings": findings,
        "mode": "paper_only",
    }
    state = _merge_seed_defaults(state)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
    return state


def load_state() -> dict:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
        fixed = _merge_seed_defaults(state)
        # Persist repair so subsequent loads stay complete
        if fixed.get("backtest") and not (state.get("backtest") or {}).get("equity_weekly"):
            STATE_PATH.write_text(json.dumps(fixed, indent=2, default=str))
        return fixed
    if SEED_PATH.exists():
        state = _merge_seed_defaults(json.loads(SEED_PATH.read_text()))
        DATA.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
        return state
    return build_state()


def ensure_dashboard_state() -> dict:
    """Idempotent startup helper used by the web server."""
    return load_state() if STATE_PATH.exists() or SEED_PATH.exists() else build_state()
