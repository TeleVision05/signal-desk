"""Dashboard endpoints: 200 on an empty database, real numbers after a tick, read-only."""

from __future__ import annotations

import asyncio
import json

import pytest
from live_helpers import Const, FakeSource, book, config, make_5m

from profit_engine.live import queries
from profit_engine.live.app import create_app, serve
from profit_engine.live.runner import tick
from profit_engine.live.store import Store


def get(app, path: str, query: str = "", host: bytes = b"127.0.0.1:8765") -> tuple[int, bytes]:
    """Minimal ASGI GET (httpx is not a dependency of this repo)."""
    sent: list[dict] = []
    state = {"requested": False}

    async def run() -> None:
        done = asyncio.Event()

        async def receive() -> dict:
            if not state["requested"]:
                state["requested"] = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await done.wait()  # like a real client: disconnect only after the response
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            sent.append(message)
            if message["type"] == "http.response.body" and not message.get("more_body"):
                done.set()

        await app(scope, receive, send)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
        "path": path, "raw_path": path.encode(), "query_string": query.encode(), "root_path": "",
        "headers": [(b"host", host)], "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8765),
    }  # fmt: skip
    asyncio.run(run())
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


@pytest.mark.parametrize("path", ["/", "/api/overview", "/api/heartbeat", "/api/events"])
def test_endpoints_200_on_empty_db(tmp_path, path):
    app = create_app(tmp_path / "fresh" / "live.sqlite")
    status, body = get(app, path)
    assert status == 200
    if path == "/":
        assert b"PAPER MONEY" in body
    elif path == "/api/overview":
        data = json.loads(body)
        assert data["paper"] is True and data["books"] == [] and data["heartbeat"] == {}


def test_unknown_book_is_a_friendly_404(tmp_path):
    status, body = get(create_app(tmp_path / "live.sqlite"), "/api/books/nope/fills")
    assert status == 404
    assert json.loads(body)["detail"].startswith("We couldn't find that book")


def test_endpoints_after_a_tick(tmp_path, quiet):
    db = tmp_path / "live.sqlite"
    store = Store(db)
    src = FakeSource({"BTCUSDT": make_5m(400)}, "2021-03-01 05:00:30+00:00")
    tick(config(book(Const(1.0), book_id="btc-test", title="BTC test")), store, src, quiet)
    store.close()
    app = create_app(db)
    status, body = get(app, "/api/overview")
    (b,) = json.loads(body)["books"]
    assert status == 200 and b["book_id"] == "btc-test" and b["label"] == "demo — not promoted"
    assert b["units"] > 0 and b["last_decision"]["explanation"] == "constant 1.0"
    assert json.loads(get(app, "/api/heartbeat")[1])["runner"]["status"] == "ok"
    for part in ("equity", "fills", "decisions"):
        status, body = get(app, f"/api/books/btc-test/{part}")
        assert status == 200 and len(json.loads(body)) == 1


def test_queries_cannot_write(tmp_path):
    db = tmp_path / "live.sqlite"
    Store(db).close()
    conn = queries.connect_ro(db)
    with pytest.raises(Exception, match=r"readonly|read-only|query_only"):
        conn.execute("DELETE FROM books")


def test_dashboard_refuses_non_local_bind():
    with pytest.raises(ValueError, match="localhost"):
        serve(host="0.0.0.0")
