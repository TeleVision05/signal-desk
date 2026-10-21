"""Always-on paper dashboard API + static UI."""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .dashboard_state import STATE_PATH, build_state, load_state

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "dashboard" / "static"
CRON_SECRET = os.environ.get("CRON_SECRET", "dev-paper-secret")
INITIAL_CASH = float(os.environ.get("PAPER_CASH", "1000"))

_sync_lock = threading.Lock()
_scheduler: BackgroundScheduler | None = None


def _run_paper_sync() -> dict:
    with _sync_lock:
        from .paper_ml import sync_ml_paper

        info = sync_ml_paper(initial_cash=INITIAL_CASH)
        state = build_state(include_backtest=True)
        state["paper"] = info
        state["last_sync_at"] = datetime.now(timezone.utc).isoformat()
        STATE_PATH.write_text(__import__("json").dumps(state, indent=2, default=str))
        return info


def _scheduled_sync():
    try:
        _run_paper_sync()
    except Exception as exc:  # noqa: BLE001
        print(f"[scheduler] sync failed: {exc}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler
    # Ensure state exists for first paint
    if not STATE_PATH.exists():
        try:
            build_state(include_backtest=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[startup] state build deferred: {exc}", flush=True)

    _scheduler = BackgroundScheduler(timezone="UTC")
    # Weekdays 21:15 UTC ≈ after US cash close
    _scheduler.add_job(_scheduled_sync, "cron", day_of_week="mon-fri", hour=21, minute=15, id="paper_sync")
    _scheduler.start()
    yield
    if _scheduler:
        _scheduler.shutdown(wait=False)


app = FastAPI(title="Signal Desk", version="0.1.0", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"ok": True, "mode": "paper_only", "utc": datetime.now(timezone.utc).isoformat()}


@app.get("/api/state")
def state():
    return load_state()


@app.post("/api/sync")
def sync(x_cron_secret: str | None = Header(default=None)):
    if x_cron_secret != CRON_SECRET:
        raise HTTPException(status_code=401, detail="invalid cron secret")
    info = _run_paper_sync()
    return {"ok": True, "paper": info}


@app.get("/")
def index():
    index = STATIC / "index.html"
    if not index.exists():
        raise HTTPException(404, "dashboard not built")
    return FileResponse(index)


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("profit_engine.server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
