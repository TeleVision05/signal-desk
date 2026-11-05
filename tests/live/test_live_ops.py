"""Operations: the Binance source vs a wrong Mac clock, log volume under launchd, lock and port
contention without a restart loop, and the installer's --no-dashboard and env handling."""

from __future__ import annotations

import logging
import socket
import threading
import time
from pathlib import Path

import pandas as pd
from live_helpers import FakeSource, make_5m

from profit_engine.live import app, cli, macos, market, paths, service
from profit_engine.live.store import Store


def _raw(start: pd.Timestamp, n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        o = start + pd.Timedelta(minutes=5 * i)
        ms = int(o.value // 1_000_000)
        rows.append([ms, 100, 101, 99, 100.5, 10, ms + 299_999, 1000, 5, 4, 400, 0])
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"]  # fmt: skip
    return pd.DataFrame(rows, columns=cols).astype(str)


def test_source_keeps_bars_when_the_mac_clock_is_days_behind(monkeypatch):
    ahead = pd.Timestamp.now(tz="UTC").floor("5min") + pd.Timedelta(days=2)  # exchange time, Mac 2 days slow
    monkeypatch.setattr(market.binance, "fetch_rest", lambda *a, **k: _raw(ahead - pd.Timedelta("1h"), 13))
    frame = market.BinanceBarSource(retries=0).klines_5m("BTCUSDT", ahead - pd.Timedelta("1h"))
    assert len(frame) == 13


def test_launchd_runner_logs_to_the_rotated_file_only(monkeypatch):
    monkeypatch.setattr(cli, "run_forever", lambda: 0)
    assert "--no-console-log" in macos.runner_plist(_plan())["ProgramArguments"]
    assert cli.main(["run", "--no-console-log"]) == 0
    handlers = logging.getLogger("signal_desk").handlers
    assert handlers and not any(type(h) is logging.StreamHandler for h in handlers)
    assert cli.main(["run"]) == 0
    assert any(type(h) is logging.StreamHandler for h in logging.getLogger("signal_desk").handlers)


def test_second_runner_waits_for_the_lock_instead_of_exiting(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "LOCK_RETRY_S", 0.05)
    monkeypatch.setenv("SIGNAL_DESK_BOOKS", str(tmp_path / "books.json"))
    (tmp_path / "books.json").write_text('{"books": []}')
    holder = service.InstanceLock(paths.lock_path())
    assert holder.acquire()
    stop = threading.Event()
    src = FakeSource({"BTCUSDT": make_5m(100)}, "2021-03-01 05:00:30+00:00")
    t = threading.Thread(target=service.run_forever, kwargs={"stop": stop, "source": src})
    t.start()
    time.sleep(0.3)
    assert t.is_alive()  # waiting, not exited (launchd would relaunch an exit every 30 s)
    holder.release()
    deadline = time.monotonic() + 5
    store = Store(paths.db_path())
    while time.monotonic() < deadline and not store.conn.execute("SELECT 1 FROM heartbeat").fetchone():
        time.sleep(0.05)
    stop.set()
    t.join(5)
    assert not t.is_alive()
    assert store.conn.execute("SELECT status FROM heartbeat WHERE component='runner'").fetchone()[0] == "stopped"
    store.close()


def test_dashboard_waits_for_a_busy_port(monkeypatch):
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen()
    port = busy.getsockname()[1]
    started = []
    monkeypatch.setattr(app, "PORT_RETRY_S", 0.05)
    monkeypatch.setattr(app.uvicorn, "run", lambda *a, **k: started.append(k["port"]))
    t = threading.Thread(target=app.serve, kwargs={"port": port})
    t.start()
    time.sleep(0.3)
    assert started == [] and t.is_alive()
    busy.close()
    t.join(5)
    assert started == [port]


def _plan(tmp: Path | None = None, **kw) -> macos.InstallPlan:
    home = tmp or Path("/Users/demo")
    return macos.InstallPlan(python=Path("/opt/sd/.venv/bin/python"), repo=Path("/opt/sd"), home=home,
                             logs=home / "Library" / "Logs" / "SignalDesk", **kw)  # fmt: skip


def test_install_without_dashboard_retires_an_old_dashboard_agent(tmp_path):
    ran: list[list[str]] = []
    full = _plan(tmp_path)
    macos.install(full, run=lambda cmd: ran.append(cmd) or 0, out=lambda _s: None, uid=501)
    old = full.agents_dir / f"{macos.DASHBOARD_LABEL}.plist"
    assert old.exists()
    ran.clear()
    macos.install(_plan(tmp_path, dashboard=False), run=lambda cmd: ran.append(cmd) or 0, out=lambda _s: None, uid=501)
    assert ["launchctl", "bootout", f"gui/501/{macos.DASHBOARD_LABEL}"] in ran
    assert not old.exists()
    said: list[str] = []
    macos.install(_plan(tmp_path, dashboard=False), dry_run=True, run=ran.append, out=said.append, uid=501)
    assert any(macos.DASHBOARD_LABEL in s and "bootout" in s for s in said)


def test_forwarded_paths_are_made_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SIGNAL_DESK_LIVE_DB", "data/probe.sqlite")
    monkeypatch.setenv("SIGNAL_DESK_BOOKS", "~/books.json")
    monkeypatch.setenv("SIGNAL_DESK_NOTIFY", "0")
    env = macos.default_plan().env
    assert env["SIGNAL_DESK_LIVE_DB"] == str((tmp_path / "data" / "probe.sqlite").resolve())
    assert env["SIGNAL_DESK_BOOKS"] == str((Path.home() / "books.json").resolve())
    assert env["SIGNAL_DESK_NOTIFY"] == "0"
