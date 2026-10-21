"""Always-on paper dashboard API + static UI."""

from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .dashboard_state import STATE_PATH, build_state, ensure_dashboard_state, load_state

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "dashboard" / "static"
CRON_SECRET = os.environ.get("CRON_SECRET", "signal-desk-paper-sync")
INITIAL_CASH = float(os.environ.get("PAPER_CASH", "1000"))
# Paper-only desk: allow browser Run sync without a shared secret by default.
ALLOW_PUBLIC_SYNC = os.environ.get("ALLOW_PUBLIC_SYNC", "1") == "1"
ET = ZoneInfo("America/New_York")

_sync_lock = threading.Lock()
_scheduler: BackgroundScheduler | None = None


def _us_equity_session_open(now: datetime | None = None) -> bool:
    """Rough US cash session: weekdays 9:30–16:00 America/New_York (no holiday calendar)."""
    now = now or datetime.now(ET)
    if now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    else:
        now = now.astimezone(ET)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return (9 * 60 + 30) <= minutes <= (16 * 60)


def _run_paper_sync(*, reason: str = "manual") -> dict:
    with _sync_lock:
        from .paper_ml import sync_ml_paper

        print(f"[sync] start reason={reason}", flush=True)
        info = sync_ml_paper(initial_cash=INITIAL_CASH)
        # Persist paper snapshot for next boot
        OUT = ROOT / "output"
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "ml_paper_last.json").write_text(json.dumps(info, indent=2, default=str))

        from .decisions import load_decisions

        state = build_state(include_backtest=True)
        state["paper"] = info
        state["decisions"] = load_decisions(limit=60)
        state["last_sync_at"] = datetime.now(timezone.utc).isoformat()
        state["last_sync_reason"] = reason
        STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
        print(
            f"[sync] done signal={info.get('signal')} equity={info.get('equity')} action={info.get('action')}",
            flush=True,
        )
        return info


def _scheduled_sync(reason: str = "schedule"):
    try:
        _run_paper_sync(reason=reason)
    except Exception as exc:  # noqa: BLE001
        print(f"[scheduler] sync failed ({reason}): {exc}", flush=True)


def _maybe_intraday_sync():
    """During RTH, refresh marks/signals; outside RTH only log skip unless forced jobs run."""
    if _us_equity_session_open():
        _scheduled_sync("rth_hourly")
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
    # Pre-open wake / sync (13:35 UTC ≈ 9:35 ET EDT)
    _scheduler.add_job(_scheduled_sync, "cron", day_of_week="mon-fri", hour=13, minute=35, args=["preopen"], id="preopen")
    # Hourly during typical RTH window UTC (14:00–20:00) — job self-gates on ET session
    _scheduler.add_job(_maybe_intraday_sync, "cron", day_of_week="mon-fri", hour="14-20", minute=5, id="rth_hourly")
    # After US cash close (21:15 UTC)
    _scheduler.add_job(_scheduled_sync, "cron", day_of_week="mon-fri", hour=21, minute=15, args=["after_close"], id="after_close")
    _scheduler.start()

    # Kick an initial sync in background so paper isn't empty forever
    threading.Thread(target=lambda: _scheduled_sync("startup"), daemon=True).start()

    yield
    if _scheduler:
        _scheduler.shutdown(wait=False)


app = FastAPI(title="Signal Desk", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://television05.github.io",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_origin_regex=r"https://.*\.github\.io",
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
        "version": "0.3.0",
        "allow_public_sync": ALLOW_PUBLIC_SYNC,
        "trade_alerts": os.environ.get("TRADE_ALERTS", "1") == "1",
    }


@app.get("/api/state")
def state():
    return load_state()


@app.post("/api/sync")
def sync(x_cron_secret: str | None = Header(default=None)):
    authed = x_cron_secret == CRON_SECRET or ALLOW_PUBLIC_SYNC
    if not authed:
        raise HTTPException(status_code=401, detail="invalid cron secret")
    info = _run_paper_sync(reason="api")
    return {"ok": True, "paper": info}


@app.get("/api/config")
def config():
    return {
        "paper_cash": INITIAL_CASH,
        "allow_public_sync": ALLOW_PUBLIC_SYNC,
        "market_open_et": _us_equity_session_open(),
        "schedules": ["preopen 13:35 UTC", "rth hourly 14-20 UTC", "after_close 21:15 UTC"],
        "trade_alert_email": os.environ.get("TRADE_ALERT_EMAIL", "tarun.vadapalli.1@gmail.com"),
    }


@app.post("/api/test-alert")
def test_alert(x_cron_secret: str | None = Header(default=None)):
    """Send a sample trade email so FormSubmit/SMTP can be activated."""
    authed = x_cron_secret == CRON_SECRET or ALLOW_PUBLIC_SYNC
    if not authed:
        raise HTTPException(status_code=401, detail="invalid cron secret")
    from .notify import send_trade_alert

    sample = {
        "run_id": "ml-tqqq-hgb-paper-v1",
        "strategy": "ml_TQQQ_hgb_h5_t0.55",
        "signal": 1,
        "proba": 0.61,
        "action": "signal_entry",
        "equity": 1000.0,
        "cash": 0.0,
        "asset": "TQQQ",
        "shares": 14.2,
        "mark": 70.4,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }
    send_trade_alert(sample)
    return {"ok": True, "emailed": sample}


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
