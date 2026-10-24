"""Signal Desk: paper trading backend + live dashboard API."""

from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .books import BOOKS, book_ids
from .dashboard_state import STATE_PATH, build_state, ensure_dashboard_state, load_state, project_active_book
from .db import using_postgres
from .notify import email_configured, load_notify_status

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "dashboard" / "static"
CRON_SECRET = os.environ.get("CRON_SECRET", "signal-desk-paper-sync")
INITIAL_CASH = float(os.environ.get("PAPER_CASH", "1000"))
ALLOW_PUBLIC_SYNC = os.environ.get("ALLOW_PUBLIC_SYNC", "1") == "1"
ET = ZoneInfo("America/New_York")
VERSION = "0.6.1"

_sync_lock = threading.Lock()
_scheduler: BackgroundScheduler | None = None

# Official equity update slots (UTC), matching in-process APScheduler jobs.
_SYNC_WEEKDAY_SLOTS: tuple[tuple[int, int], ...] = (
    (13, 35),  # preopen
    (14, 5),
    (15, 5),
    (16, 5),
    (17, 5),
    (18, 5),
    (19, 5),
    (20, 5),  # RTH hourly
    (21, 15),  # after close
)

# Crypto 24/7 slots every 4h UTC
_CRYPTO_SLOTS: tuple[tuple[int, int], ...] = (
    (0, 10),
    (4, 10),
    (8, 10),
    (12, 10),
    (16, 10),
    (20, 10),
)


def _us_equity_session_open(now: datetime | None = None) -> bool:
    """US cash session weekdays 9:30–16:00 ET (no holiday calendar)."""
    now = now or datetime.now(ET)
    if now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    else:
        now = now.astimezone(ET)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return (9 * 60 + 30) <= minutes <= (16 * 60)


def _next_scheduled_sync(now: datetime | None = None) -> datetime:
    """Next weekday equity sync slot in UTC."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    for day_offset in range(0, 14):
        day = (now + timedelta(days=day_offset)).date()
        if day.weekday() >= 5:
            continue
        for hour, minute in _SYNC_WEEKDAY_SLOTS:
            candidate = datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc)
            if candidate > now:
                return candidate
    days_ahead = (7 - now.weekday()) % 7 or 7
    monday = (now + timedelta(days=days_ahead)).date()
    return datetime(monday.year, monday.month, monday.day, 13, 35, tzinfo=timezone.utc)


def _next_crypto_sync(now: datetime | None = None) -> datetime:
    """Next 4h crypto sync slot in UTC (every day including weekends)."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    for day_offset in range(0, 3):
        day = (now + timedelta(days=day_offset)).date()
        for hour, minute in _CRYPTO_SLOTS:
            candidate = datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc)
            if candidate > now:
                return candidate
    day = (now + timedelta(days=1)).date()
    return datetime(day.year, day.month, day.day, 0, 10, tzinfo=timezone.utc)


def _sync_one_book(book_id: str) -> dict:
    from .paper_ml import sync_ml_paper
    from .paper_ml_crypto import sync_crypto_paper

    if book_id == "btc":
        return sync_crypto_paper(initial_cash=INITIAL_CASH)
    return sync_ml_paper(initial_cash=INITIAL_CASH)


def _run_paper_sync(*, reason: str = "manual", books: list[str] | None = None) -> dict:
    targets = books or book_ids()
    with _sync_lock:
        from .decisions import load_decisions

        print(f"[sync] start reason={reason} books={targets}", flush=True)
        results: dict[str, dict] = {}
        for bid in targets:
            try:
                info = _sync_one_book(bid)
                results[bid] = info
                print(
                    f"[sync] {bid} signal={info.get('signal')} equity={info.get('equity')} "
                    f"action={info.get('action')} p={info.get('proba')}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[sync] {bid} FAILED: {exc}", flush=True)
                results[bid] = {"ok": False, "error": str(exc), "book": bid}

        OUT = ROOT / "output"
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "ml_paper_sync_last.json").write_text(json.dumps(results, indent=2, default=str))

        state = build_state(include_backtest=True)
        # Keep top-level paper as TQQQ for older clients; books map has both.
        state["paper"] = (results.get("tqqq") if isinstance(results.get("tqqq"), dict) else None) or state.get(
            "paper"
        )
        state["decisions"] = load_decisions(limit=60, run_id=BOOKS["tqqq"]["run_id"])
        state["notify"] = load_notify_status()
        state["last_sync_at"] = datetime.now(timezone.utc).isoformat()
        state["last_sync_reason"] = reason
        state["last_sync_books"] = results
        STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
        return results


def _scheduled_sync(reason: str = "schedule", books: list[str] | None = None):
    try:
        _run_paper_sync(reason=reason, books=books)
    except Exception as exc:  # noqa: BLE001
        print(f"[scheduler] sync failed ({reason}): {exc}", flush=True)


def _maybe_intraday_sync():
    if _us_equity_session_open():
        _scheduled_sync("rth_hourly", books=["tqqq"])
    else:
        print("[scheduler] skip rth_hourly — market closed", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler
    try:
        ensure_dashboard_state()
        print("[startup] dashboard state ready", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] state ensure failed: {exc}", flush=True)

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        _scheduled_sync,
        "cron",
        day_of_week="mon-fri",
        hour=13,
        minute=35,
        args=["preopen", ["tqqq"]],
        id="preopen",
    )
    _scheduler.add_job(
        _maybe_intraday_sync, "cron", day_of_week="mon-fri", hour="14-20", minute=5, id="rth_hourly"
    )
    _scheduler.add_job(
        _scheduled_sync,
        "cron",
        day_of_week="mon-fri",
        hour=21,
        minute=15,
        args=["after_close", ["tqqq"]],
        id="after_close",
    )
    _scheduler.add_job(
        _scheduled_sync,
        "cron",
        hour="0,4,8,12,16,20",
        minute=10,
        args=["crypto_4h", ["btc"]],
        id="crypto_4h",
    )
    _scheduler.start()
    threading.Thread(target=lambda: _scheduled_sync("startup", book_ids()), daemon=True).start()

    yield
    if _scheduler:
        _scheduler.shutdown(wait=False)


app = FastAPI(title="Signal Desk", version=VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "mode": "paper_only",
        "utc": datetime.now(timezone.utc).isoformat(),
        "et": datetime.now(ET).isoformat(),
        "market_open_et": _us_equity_session_open(),
        "version": VERSION,
        "books": book_ids(),
        "email_ready": email_configured(),
        "trade_alerts": os.environ.get("TRADE_ALERTS", "1") == "1",
        "db_backend": "postgres" if using_postgres() else "sqlite",
    }


@app.get("/api/state")
def state(book: str | None = Query(default=None)):
    data = load_state()
    data = project_active_book(data, book)
    data["notify"] = load_notify_status()
    data["market_open_et"] = _us_equity_session_open()
    data["next_sync_at"] = (
        _next_crypto_sync().isoformat()
        if data.get("active_book") == "btc"
        else _next_scheduled_sync().isoformat()
    )
    data["next_equity_sync_at"] = _next_scheduled_sync().isoformat()
    data["next_crypto_sync_at"] = _next_crypto_sync().isoformat()
    data["version"] = VERSION
    return data


@app.post("/api/sync")
def sync(
    x_cron_secret: str | None = Header(default=None),
    book: str | None = Query(default=None),
):
    authed = x_cron_secret == CRON_SECRET or ALLOW_PUBLIC_SYNC
    if not authed:
        raise HTTPException(status_code=401, detail="invalid cron secret")
    if book and book.lower() not in ("all", *book_ids()):
        raise HTTPException(status_code=400, detail=f"unknown book={book}")
    targets = book_ids() if not book or book.lower() == "all" else [book.lower()]
    results = _run_paper_sync(reason="api", books=targets)
    return {"ok": True, "books": results, "notify": load_notify_status()}


@app.get("/api/config")
def config():
    return {
        "paper_cash": INITIAL_CASH,
        "market_open_et": _us_equity_session_open(),
        "books": [
            {
                "id": b["id"],
                "label": b["label"],
                "asset": b["asset"],
                "run_id": b["run_id"],
                "session": b["session"],
            }
            for b in BOOKS.values()
        ],
        "schedules": {
            "tqqq": ["preopen 13:35 UTC", "rth hourly 14-20 UTC", "after_close 21:15 UTC"],
            "btc": ["every 4h at :10 UTC (0/4/8/12/16/20)"],
        },
        "next_sync_at": _next_scheduled_sync().isoformat(),
        "next_crypto_sync_at": _next_crypto_sync().isoformat(),
        "trade_alert_email": os.environ.get("TRADE_ALERT_EMAIL", "tarun.vadapalli.1@gmail.com"),
        "email_ready": email_configured(),
        "db_backend": "postgres" if using_postgres() else "sqlite",
        "version": VERSION,
    }


@app.post("/api/test-alert")
def test_alert(x_cron_secret: str | None = Header(default=None)):
    authed = x_cron_secret == CRON_SECRET or ALLOW_PUBLIC_SYNC
    if not authed:
        raise HTTPException(status_code=401, detail="invalid cron secret")
    from .notify import send_trade_alert

    sample = {
        "run_id": "ml-tqqq-hgb-paper-v1",
        "book": "tqqq",
        "strategy": "ml_TQQQ_hgb_h5_t0.55",
        "signal": 1,
        "proba": 0.61,
        "threshold": 0.55,
        "action": "signal_entry",
        "equity": 1000.0,
        "cash": 0.0,
        "asset": "TQQQ",
        "shares": 14.2,
        "mark": 70.4,
        "bar_date": datetime.now(ET).date().isoformat(),
        "explanation": "Manual test alert from Signal Desk.",
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }
    delivery = send_trade_alert(sample)
    return {"ok": bool(delivery.get("ok")), "delivery": delivery}


@app.get("/")
def index():
    index_path = STATIC / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "dashboard not built")
    return FileResponse(index_path)


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("profit_engine.server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
