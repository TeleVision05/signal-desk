"""Build a JSON snapshot the dashboard can serve without retraining."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .books import BOOKS, DEFAULT_BOOK, resolve_book
from .ledger import PaperLedger

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "output"
STATE_PATH = DATA / "dashboard_state.json"
SEED_PATH = ROOT / "dashboard" / "seed_state.json"
PROMO_SEED = ROOT / "dashboard" / "ml_promoted.seed.json"

STRATEGY_META = {
    "tqqq": {
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
    "btc": {
        "name": "ml_BTCUSDT_hgb_h5_t0.55",
        "ticker": "BTCUSDT",
        "model": "HistGradientBoosting",
        "horizon_days": 5,
        "proba_threshold": 0.55,
        "hard_stop_pct": 0.08,
        "trailing_stop_pct": 0.12,
        "train_days": 504,
        "cost_bps": 20,
        "features": [
            "returns 1/2/3/5/10/20/60d",
            "vol 10/20 + ratio",
            "dist to SMA 10/20/50/100/200",
            "RSI 7/14",
            "ATR%",
            "volume z-score",
            "Donchian 20/55",
            "ETHUSDT cross features",
        ],
    },
}


def _ledger_rows(run_id: str) -> dict:
    return PaperLedger().snapshot(run_id)


def _seed() -> dict:
    if SEED_PATH.exists():
        return json.loads(SEED_PATH.read_text())
    return {}


def _paper_last(book_id: str) -> dict:
    path = OUT / ("ml_paper_last.json" if book_id == "tqqq" else "ml_crypto_paper_last.json")
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _tqqq_backtest() -> dict:
    eq_csv = OUT / "ml_TQQQ_hgb_h5_t0.55_equity.csv"
    if not eq_csv.exists():
        matches = list(OUT.glob("ml_TQQQ_hgb_h5_t0.55*_equity.csv"))
        eq_csv = matches[0] if matches else None
    if eq_csv and Path(eq_csv).exists():
        eq = pd.read_csv(eq_csv, index_col=0, parse_dates=True).iloc[:, 0]
        eq_w = eq.resample("W-FRI").last().dropna()
        return {
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
    return (_seed().get("backtest") or {})


def _promoted_metrics() -> dict:
    for promo_path in (OUT / "ml_promoted.json", PROMO_SEED):
        if promo_path.exists():
            return json.loads(promo_path.read_text())
    return (_seed().get("promoted_metrics") or {})


def build_book_slice(book_id: str) -> dict:
    meta = BOOKS[book_id]
    run_id = meta["run_id"]
    from .decisions import load_decisions

    return {
        "id": book_id,
        "label": meta["label"],
        "asset": meta["asset"],
        "run_id": run_id,
        "session": meta["session"],
        "tip": meta["tip"],
        "strategy": STRATEGY_META[book_id],
        "paper": _paper_last(book_id),
        "ledger": _ledger_rows(run_id),
        "decisions": load_decisions(limit=60, run_id=run_id),
        "backtest": _tqqq_backtest() if book_id == "tqqq" else {},
        "promoted_metrics": _promoted_metrics() if book_id == "tqqq" else {},
    }


def _merge_seed_defaults(state: dict) -> dict:
    """Fill missing chart/metrics from committed seed so first paint never looks empty."""
    seed = _seed()
    books = state.setdefault("books", {})
    if "tqqq" not in books:
        books["tqqq"] = {}
    tqqq = books["tqqq"]
    if not tqqq.get("backtest") or not (tqqq.get("backtest") or {}).get("equity_weekly"):
        tqqq["backtest"] = seed.get("backtest") or {}
    if not tqqq.get("promoted_metrics"):
        if PROMO_SEED.exists():
            tqqq["promoted_metrics"] = json.loads(PROMO_SEED.read_text())
        else:
            tqqq["promoted_metrics"] = seed.get("promoted_metrics") or {}
    if not tqqq.get("strategy"):
        tqqq["strategy"] = STRATEGY_META["tqqq"]
    tqqq.setdefault("paper", {})
    tqqq.setdefault("ledger", {"fills": [], "equity": [], "position": None, "run": None})
    tqqq.setdefault("decisions", [])

    if "btc" not in books:
        books["btc"] = {
            "id": "btc",
            "label": "BTC",
            "asset": "BTCUSDT",
            "run_id": BOOKS["btc"]["run_id"],
            "session": "crypto_24h",
            "tip": BOOKS["btc"]["tip"],
            "strategy": STRATEGY_META["btc"],
            "paper": {},
            "ledger": {"fills": [], "equity": [], "position": None, "run": None},
            "decisions": [],
            "backtest": {},
            "promoted_metrics": {},
        }

    # Backward-compat top-level mirrors active/default book
    active = resolve_book(state.get("active_book")).get("id", DEFAULT_BOOK)
    state["active_book"] = active
    active_slice = books.get(active) or books.get(DEFAULT_BOOK) or {}
    state["run_id"] = active_slice.get("run_id") or BOOKS[DEFAULT_BOOK]["run_id"]
    state["strategy"] = active_slice.get("strategy") or STRATEGY_META[DEFAULT_BOOK]
    state["paper"] = active_slice.get("paper") or {}
    state["ledger"] = active_slice.get("ledger") or {
        "fills": [],
        "equity": [],
        "position": None,
        "run": None,
    }
    state["decisions"] = active_slice.get("decisions") or []
    state["backtest"] = active_slice.get("backtest") or seed.get("backtest") or {}
    state["promoted_metrics"] = active_slice.get("promoted_metrics") or {}
    state.setdefault("mode", "paper_only")
    state.setdefault(
        "book_list",
        [{"id": b["id"], "label": b["label"], "asset": b["asset"]} for b in BOOKS.values()],
    )
    return state


def build_state(*, include_backtest: bool = True) -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    from .notify import load_notify_status

    books = {bid: build_book_slice(bid) for bid in BOOKS}
    if not include_backtest:
        books["tqqq"]["backtest"] = {}

    findings = {}
    fm = ROOT / "FINDINGS_ML.md"
    if fm.exists():
        findings["path"] = str(fm)
        findings["summary"] = (
            "Walk-forward HGB on TQQQ + BTCUSDT paper books. "
            "TQQQ: P>=0.55, 8%/12% stops. BTC: same model, Binance daily, 20bps costs."
        )

    state = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active_book": DEFAULT_BOOK,
        "book_list": [
            {"id": b["id"], "label": b["label"], "asset": b["asset"]} for b in BOOKS.values()
        ],
        "books": books,
        "findings": findings,
        "notify": load_notify_status(),
        "mode": "paper_only",
    }
    state = _merge_seed_defaults(state)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
    return state


def project_active_book(state: dict, book_id: str | None) -> dict:
    """Copy selected book fields to top-level for the UI."""
    book = resolve_book(book_id)
    bid = book["id"]
    books = state.get("books") or {}
    slice_ = books.get(bid) or build_book_slice(bid)
    out = dict(state)
    out["active_book"] = bid
    out["run_id"] = slice_.get("run_id")
    out["strategy"] = slice_.get("strategy") or STRATEGY_META.get(bid, {})
    out["paper"] = slice_.get("paper") or {}
    out["ledger"] = slice_.get("ledger") or {
        "fills": [],
        "equity": [],
        "position": None,
        "run": None,
    }
    out["decisions"] = slice_.get("decisions") or []
    out["backtest"] = slice_.get("backtest") or {}
    out["promoted_metrics"] = slice_.get("promoted_metrics") or {}
    out["book_tip"] = slice_.get("tip") or book.get("tip")
    out["book_session"] = slice_.get("session") or book.get("session")
    return out


def load_state() -> dict:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
        fixed = _merge_seed_defaults(state)
        if fixed.get("backtest") and not (state.get("backtest") or {}).get("equity_weekly"):
            STATE_PATH.write_text(json.dumps(fixed, indent=2, default=str))
        return fixed
    if SEED_PATH.exists():
        seed = json.loads(SEED_PATH.read_text())
        state = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "active_book": DEFAULT_BOOK,
            "books": {
                "tqqq": {
                    "id": "tqqq",
                    "label": "TQQQ",
                    "asset": "TQQQ",
                    "run_id": BOOKS["tqqq"]["run_id"],
                    "session": "us_rth",
                    "tip": BOOKS["tqqq"]["tip"],
                    "strategy": STRATEGY_META["tqqq"],
                    "paper": seed.get("paper") or {},
                    "ledger": seed.get("ledger")
                    or {"fills": [], "equity": [], "position": None, "run": None},
                    "decisions": seed.get("decisions") or [],
                    "backtest": seed.get("backtest") or {},
                    "promoted_metrics": seed.get("promoted_metrics") or {},
                }
            },
            "mode": "paper_only",
        }
        state = _merge_seed_defaults(state)
        DATA.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
        return state
    return build_state()


def ensure_dashboard_state() -> dict:
    """Idempotent startup helper used by the web server."""
    return load_state() if STATE_PATH.exists() or SEED_PATH.exists() else build_state()
