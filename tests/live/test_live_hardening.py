"""Hardening: typed books.json settings, a loop that survives a bad edit, shutdown during a
network stall, the kline cache vs the settle margin, unusable database files, the installer's
running check and log paths, the dashboard's Host check and its offline heartbeat."""

from __future__ import annotations

import json
import plistlib
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
import requests
from live_helpers import FakeSource, make_5m, rows
from test_live_app import get

from profit_engine.intraday.binance import RAW_COLUMNS
from profit_engine.live import app, cli, macos, market, paths, service
from profit_engine.live.registry import ConfigError, parse_config
from profit_engine.live.store import Store, StoreError

DEMO = {"book_id": "d", "strategy": "demo_sma_trend", "symbol": "BTCUSDT", "timeframe": "1h"}


@pytest.mark.parametrize(
    "settings",
    [{"poll_seconds": "30"}, {"poll_seconds": 0}, {"poll_seconds": -5}, {"jitter": None},
     {"notifications": "false"}, {"max_daily_loss": 1.5}, {"settle_seconds": True}, {"stale_after_bars": 0}],
)  # fmt: skip
def test_wrongly_typed_or_out_of_range_settings_are_refused(settings):
    with pytest.raises(ConfigError):
        parse_config({**settings, "books": [DEMO]})


@pytest.mark.parametrize("entry", [{"enabled": "false"}, {"starting_equity": "10000"}, {"max_daily_loss": -1}])
def test_wrongly_typed_book_values_are_refused(entry):
    with pytest.raises(ConfigError, match="book 'd'"):
        parse_config({"books": [{**DEMO, **entry}]})


def _run_until(stop: threading.Event, src, until, timeout: float = 5.0) -> threading.Thread:
    t = threading.Thread(target=service.run_forever, kwargs={"stop": stop, "source": src})
    t.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and t.is_alive() and not until():
        time.sleep(0.05)
    return t


def test_a_wrongly_typed_edit_keeps_the_loop_alive_and_logs_it(monkeypatch, tmp_path):
    books = tmp_path / "books.json"
    books.write_text(json.dumps({"poll_seconds": "30", "books": []}))
    monkeypatch.setenv("SIGNAL_DESK_BOOKS", str(books))
    stop = threading.Event()
    src = FakeSource({"BTCUSDT": make_5m(100)}, "2021-03-01 05:00:30+00:00")
    db = paths.db_path()

    def logged() -> bool:
        return db.exists() and bool(rows(Store(db), "SELECT * FROM events WHERE kind='config_error'"))

    t = _run_until(stop, src, logged)
    alive = t.is_alive()
    stop.set()
    t.join(5)
    assert alive and not t.is_alive()  # it waited for a fix instead of crashing into a relaunch loop
    (ev,) = rows(Store(db), "SELECT * FROM events WHERE kind='config_error'")
    assert "kept the last good settings" in ev["message"]


def test_offline_heartbeat_says_when_the_next_try_is(monkeypatch, tmp_path):
    (tmp_path / "books.json").write_text(json.dumps({"poll_seconds": 30, "jitter": 0, "books": [DEMO]}))
    monkeypatch.setenv("SIGNAL_DESK_BOOKS", str(tmp_path / "books.json"))
    src = FakeSource({"BTCUSDT": make_5m(100)}, "2021-03-01 05:00:30+00:00")
    src.down = True
    stop, db = threading.Event(), paths.db_path()

    def detail() -> dict:
        got = db.exists() and rows(Store(db), "SELECT detail_json FROM heartbeat WHERE component='runner'")
        return json.loads(got[0]["detail_json"]) if got else {}

    t = _run_until(stop, src, lambda: "next_check_s" in detail())
    seen = detail()
    stop.set()
    t.join(5)
    assert seen["next_check_s"] == 60.0  # one failure: the 30 s poll doubled, so the page is not "late" yet
    page = (Path(app.__file__).parent / "static" / "index.html").read_text()
    assert "next_check_s" in page and "still retrying" in page


class _StalledSession:
    def __init__(self, stop: threading.Event) -> None:
        self.stop, self.calls = stop, 0

    def get(self, *_a, **_k):
        self.calls += 1
        self.stop.set()  # SIGTERM arrives while this request is stalled
        raise requests.ReadTimeout("read timed out")


def test_shutdown_during_a_network_stall_does_not_keep_retrying():
    stop = threading.Event()
    session = _StalledSession(stop)
    src = market.BinanceBarSource(session=session, retries=2, stop=stop)
    started = time.monotonic()
    with pytest.raises(market.DataUnavailable):
        src.server_time()
    with pytest.raises(market.DataUnavailable):
        src.klines_5m("BTCUSDT", pd.Timestamp("2021-03-01", tz="UTC"))
    assert time.monotonic() - started < 0.5  # no 1 s + 2 s retry sleeps, no further requests
    assert session.calls == 1
    assert macos.runner_plist(_plan())["ExitTimeOut"] == macos.EXIT_TIMEOUT_S


def test_cache_refreshes_a_bar_that_closed_just_before_the_last_fetch(monkeypatch):
    t0 = pd.Timestamp("2026-11-05 13:00", tz="UTC")
    truth = {t0 + pd.Timedelta(minutes=5 * i): 100.0 for i in range(13)}  # 13:00 .. 14:00

    def fetch(_symbol, _interval, start_ms, _session, timeout=30, end_ms=None):
        out = [[str(int(o.value // 1_000_000)), "100", "101", "99", str(c), "1",
                str(int(o.value // 1_000_000) + 299_999), "1", "1", "1", "1", "0"]
               for o, c in sorted(truth.items()) if o.value // 1_000_000 >= start_ms]  # fmt: skip
        return pd.DataFrame(out, columns=list(RAW_COLUMNS)).astype(str)

    monkeypatch.setattr(market.binance, "fetch_rest", fetch)
    bar = pd.Timestamp("2026-11-05 13:55", tz="UTC")
    truth[bar] = 111.0  # served at 14:00:00.5, before Binance finalized the 13:55 bar
    src = market.BinanceBarSource(retries=0)
    assert src.klines_5m("BTCUSDT", t0).at[bar, "close"] == 111.0
    truth[bar] = 222.0  # the final close, published a moment later
    assert src.klines_5m("BTCUSDT", t0).at[bar, "close"] == 222.0


@pytest.mark.parametrize("version", [7, 1])
def test_unusable_database_files_are_refused_in_plain_words(tmp_path, version, capsys, monkeypatch):
    path = tmp_path / "live.sqlite"
    with sqlite3.connect(path) as conn:  # a downgrade (7), or some other file marked version 1
        conn.execute(f"PRAGMA user_version={version}")
    with pytest.raises(StoreError, match="SIGNAL_DESK_LIVE_DB"):
        Store(path)
    monkeypatch.setenv("SIGNAL_DESK_LIVE_DB", str(path))
    assert cli.main(["status"]) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err and "SIGNAL_DESK_LIVE_DB" in err


def _plan(tmp: Path | None = None) -> macos.InstallPlan:
    home = tmp or Path("/Users/demo")
    return macos.InstallPlan(python=Path("/opt/sd/.venv/bin/python"), repo=Path("/opt/sd"), home=home,
                             logs=home / "Library" / "Logs" / "SignalDesk")  # fmt: skip


RUNNING = "gui/501/com.signaldesk.live = {\n\tstate = running\n\tpid = 4242\n}\n"
CRASHED = "gui/501/com.signaldesk.live = {\n\tstate = not running\n\tlast exit code = 1\n}\n"


@pytest.mark.parametrize(("printed", "code"), [(RUNNING, 0), (CRASHED, 1)])
def test_install_reports_success_only_when_the_agent_stays_running(tmp_path, printed, code):
    said, probed, paused = [], [], []
    plan = _plan(tmp_path)
    got = macos.install(
        plan, run=lambda _cmd: 0, out=said.append, uid=501,
        probe=lambda cmd: probed.append(cmd) or (0, printed), pause=paused.append,
    )  # fmt: skip
    assert got == code
    assert ["launchctl", "print", "gui/501/com.signaldesk.live"] in probed and paused == [macos.STARTUP_CHECK_S]
    text = "\n".join(said)
    if code:
        assert "not staying up" in text and "Documents" in text and "runner.err.log" in text
        assert "Installed and started com.signaldesk.live" not in text
    else:
        assert "it is running" in text


def test_launchctl_output_never_reaches_the_terminal(capfd):
    code = macos._run(["/bin/sh", "-c", "echo 'Boot-out failed: 3: No such process' >&2; exit 3"])
    assert code == 3
    captured = capfd.readouterr()
    assert "Boot-out failed" not in captured.out + captured.err


def test_relative_log_folder_becomes_absolute_in_the_plist(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SIGNAL_DESK_LOGS", "mylogs")
    plist = macos.runner_plist(macos.default_plan())
    want = str(tmp_path.resolve() / "mylogs")
    assert plist["EnvironmentVariables"]["SIGNAL_DESK_LOGS"] == want
    assert plist["StandardErrorPath"] == want + "/runner.err.log"
    assert plistlib.loads(macos.render(plist))["StandardOutPath"] == want + "/runner.out.log"


def test_dashboard_refuses_a_foreign_host_header(tmp_path):
    dashboard = app.create_app(tmp_path / "live.sqlite")
    real = get(dashboard, "/api/overview")[0]
    status, body = get(dashboard, "/api/overview", host=b"attacker.example:8765")
    assert real == 200 and status == 400
    assert b"127.0.0.1" in body and b"{" not in body
    for ok in (b"localhost:8765", b"127.0.0.1", b"[::1]:8765"):
        assert get(dashboard, "/api/heartbeat", host=ok)[0] == 200
