"""Daily loss limit: flatten at the next 5m open, pause until next UTC midnight, then resume."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from live_helpers import Const, FakeSource, book, config, path_bars, rows

from profit_engine.live.notify import Notifier
from profit_engine.live.runner import tick

START = pd.Timestamp("2021-03-01 00:00+00:00")


def _crash_path() -> pd.DataFrame:
    """Flat at 100 until 05:00, then -1% per 5m bar for 12 bars, then flat; 2 days long."""
    n = 2 * 288
    prices = np.full(n, 100.0)
    drop_from = 5 * 12  # 05:00
    prices[drop_from : drop_from + 12] = 100 * 0.99 ** np.arange(1, 13)
    prices[drop_from + 12 :] = prices[drop_from + 11]
    return path_bars(prices)


def test_kill_switch_flattens_pauses_and_resumes_next_utc_day(store):
    sent: list[str] = []
    notifier = Notifier(True, 300, sender=lambda _title, text: sent.append(text) or True)
    bars = _crash_path()
    cfg = config(book(Const(1.0), title="BTC test"), notifications=True)
    src = FakeSource({"BTCUSDT": bars}, START + pd.Timedelta("5h30s"))

    first = tick(cfg, store, src, notifier)  # decides on 04:00, buys at the 05:00 open
    assert first["books"][0]["decision"] == "traded"
    src.now = START + pd.Timedelta("5h45m30s")  # 9 bars of -1%: equity about -8.6% today
    second = tick(cfg, store, src, notifier)
    assert second["books"][0]["kill_switch"] is True

    fills = rows(store, "SELECT * FROM fills ORDER BY id")
    assert [f["reason"] for f in fills] == ["signal", "kill_switch"]
    kill = fills[1]
    assert kill["fill_bar_ts"] == "2021-03-01T05:45:00Z"  # next 5m open after the 05:40 bar's close
    assert kill["price"] == bars.at[START + pd.Timedelta("5h45m"), "open"]
    assert kill["units_after"] == pytest.approx(0.0, abs=1e-12)
    b = store.book("b1")
    assert b["paused_until"] == "2021-03-02T00:00:00Z" and b["target"] == 0.0
    (ev,) = rows(store, "SELECT * FROM events WHERE kind='kill_switch'")
    assert "daily limit" in ev["message"]
    assert any("daily limit" in t for t in sent)

    src.now = START + pd.Timedelta("6h00m30s")  # new 05:00 bar: the strategy still wants 100%...
    third = tick(cfg, store, src, notifier)
    assert third["books"][0]["decision"] == "held"  # ...but the paused book stays in cash
    dec = rows(store, "SELECT * FROM decisions WHERE bar_open_ts='2021-03-01T05:00:00Z'")[0]
    assert dec["target"] == 0.0 and "paused" in dec["explanation"]
    assert len(rows(store, "SELECT * FROM fills")) == 2
    src.now = START + pd.Timedelta("23h55m30s")
    tick(cfg, store, src, notifier)
    assert len(rows(store, "SELECT * FROM fills")) == 2

    src.now = START + pd.Timedelta("1D00h00m30s")  # next UTC day
    fourth = tick(cfg, store, src, notifier)
    assert fourth["books"][0]["decision"] == "traded"
    assert store.book("b1")["paused_until"] is None
    assert len(rows(store, "SELECT * FROM events WHERE kind='resumed'")) == 1
    assert [f["side"] for f in rows(store, "SELECT side FROM fills ORDER BY id")] == ["buy", "sell", "buy"]


def test_kill_switch_triggers_once(store):
    bars = _crash_path()
    cfg = config(book(Const(1.0)))
    src = FakeSource({"BTCUSDT": bars}, START + pd.Timedelta("5h30s"))
    tick(cfg, store, src, Notifier(False))
    for minutes in (45, 50, 55):
        src.now = START + pd.Timedelta(hours=5, minutes=minutes, seconds=30)
        tick(cfg, store, src, Notifier(False))
    assert len(rows(store, "SELECT * FROM fills WHERE reason='kill_switch'")) == 1
    assert len(rows(store, "SELECT * FROM events WHERE kind='kill_switch'")) == 1
