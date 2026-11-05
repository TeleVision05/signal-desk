"""Local, read-only dashboard for the live paper books: http://127.0.0.1:8765

Binds to localhost only. Every endpoint is a GET over :mod:`profit_engine.live.queries`
(``PRAGMA query_only``), so the dashboard cannot change a book. The page is a single static
HTML file (``static/index.html``) with inline JS: no build step, no external requests.
"""

from __future__ import annotations

import logging
import socket
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from profit_engine.live import paths, queries
from profit_engine.live.store import Store

HOST = "127.0.0.1"
PORT = 8765
STATIC = Path(__file__).resolve().parent / "static" / "index.html"
NOT_FOUND = "We couldn't find that book. It may have been renamed in books.json."
PORT_RETRY_S = 30.0
log = logging.getLogger("signal_desk.dashboard")


def create_app(db_path: Path | None = None) -> FastAPI:
    db = Path(db_path or paths.db_path())
    Store(db).close()  # make sure the file and schema exist so an empty desk renders
    app = FastAPI(title="Signal Desk Live", docs_url=None, redoc_url=None, openapi_url=None)

    def known(book_id: str) -> None:
        if not queries.book_exists(db, book_id):
            raise HTTPException(status_code=404, detail=NOT_FOUND)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC, media_type="text/html", headers={"Cache-Control": "no-store"})

    @app.get("/api/overview")
    def overview() -> dict:
        return queries.overview(db)

    @app.get("/api/heartbeat")
    def heartbeat() -> dict:
        return queries.overview(db)["heartbeat"]

    @app.get("/api/events")
    def events(book_id: str | None = None, limit: int = Query(100, ge=1, le=1000)) -> list[dict]:
        return queries.events(db, book_id, limit)

    @app.get("/api/books/{book_id}/equity")
    def equity(book_id: str, limit: int = Query(3000, ge=1, le=20000)) -> list[dict]:
        known(book_id)
        return queries.equity_curve(db, book_id, limit)

    @app.get("/api/books/{book_id}/fills")
    def fills(book_id: str, limit: int = Query(50, ge=1, le=1000)) -> list[dict]:
        known(book_id)
        return queries.fills(db, book_id, limit)

    @app.get("/api/books/{book_id}/decisions")
    def decisions(book_id: str, limit: int = Query(50, ge=1, le=1000)) -> list[dict]:
        known(book_id)
        return queries.decisions(db, book_id, limit)

    return app


def _port_free(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as uvicorn does: ignore TIME_WAIT
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def wait_for_port(host: str, port: int) -> None:
    """Block until ``port`` is free. Exiting instead would make launchd's KeepAlive relaunch the
    dashboard every 30 s for as long as something else holds the port."""
    if _port_free(host, port):
        return
    log.warning("port %s is in use; the dashboard will start when it is free", port)
    while not _port_free(host, port):
        time.sleep(PORT_RETRY_S)


def serve(host: str = HOST, port: int = PORT, db_path: Path | None = None) -> None:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("the live dashboard binds to localhost only")
    wait_for_port(host, port)
    uvicorn.run(create_app(db_path), host=host, port=port, log_level="warning", access_log=False)
