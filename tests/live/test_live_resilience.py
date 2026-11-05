"""Runner resilience: data gaps, edited books, the UTC-day loss baseline, kill re-checks, halts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from live_helpers import Const, FakeSource, book, config, make_5m, path_bars, rows

from profit_engine.intraday.backtest import run_backtest
from profit_engine.intraday.resample import resample
from profit_engine.live.notify import Notifier
from profit_engine.live.runner import tick
from profit_engine.live.store import iso
from profit_engine.live.strategy import Decision, SmaTrendDemo, target_series

START = pd.Timestamp("2021-03-01 00:00+00:00")
DAY = pd.Timedelta("1D")


def test_book_keeps_deciding_after_a_data_gap(store, quiet):
    bars5 = make_5m(600, seed=5, vol=0.004)
    gap = (bars5.index >= START + DAY + pd.Timedelta("10h")) & (bars5.index < START + DAY + pd.Timedelta("11h30m"))
    holed = bars5[~gap]
    strategy = SmaTrendDemo("BTCUSDT", "15m", fast=10, slow=30)
    cfg = config(book(strategy, max_daily_loss=0.0))
    src = FakeSource({"BTCUSDT": holed}, START)
    statuses = {}
    for t in pd.date_range(START + DAY + pd.Timedelta("8h"), START + DAY + pd.Timedelta("18h45m"), freq="15min"):
        src.now = t + pd.Timedelta("30s")
        statuses[t] = tick(cfg, store, src, quiet)["books"][0]["decision"]
    after = {t: s for t, s in statuses.items() if t >= START + DAY + pd.Timedelta("11h45m")}
    assert "warming_up" not in after.values(), after

    fifteen = resample(holed, "15m", incomplete="drop")
    want = target_series(strategy, fifteen).dropna()
    want = want[
        (want.index >= START + DAY + pd.Timedelta("11h30m")) & (want.index <= START + DAY + pd.Timedelta("18h30m"))
    ]
    got = rows(store, "SELECT bar_open_ts, target FROM decisions WHERE bar_open_ts >= ? AND bar_open_ts <= ?",
               iso(want.index[0]), iso(want.index[-1]))  # fmt: skip
    assert [(d["bar_open_ts"], d["target"]) for d in got] == [(iso(t), x) for t, x in want.items()]


def test_too_little_history_is_logged_not_silent(store, quiet):
    short = make_5m(12 * 6, seed=1)  # six hours of data, the strategy wants 30 hourly bars
    cfg = config(book(SmaTrendDemo("BTCUSDT", "1h", fast=3, slow=30)))
    result = tick(cfg, store, FakeSource({"BTCUSDT": short}, START + pd.Timedelta("5h30s")), quiet)
    assert result["books"][0]["decision"] == "warming_up"
    (ev,) = rows(store, "SELECT * FROM events WHERE kind='warming_up'")
    assert "30" in ev["message"] and "{" not in ev["message"]


def _eth(bars: pd.DataFrame) -> pd.DataFrame:
    eth = bars.copy()
    eth[["open", "high", "low", "close"]] = 2000.0
    return eth


@pytest.mark.parametrize("change", [{"symbol": "ETHUSDT"}, {"timeframe": "4h"}])
def test_editing_a_books_market_is_refused(store, quiet, change):
    bars5 = path_bars(np.full(600, 50_000.0))
    src = FakeSource({"BTCUSDT": bars5, "ETHUSDT": _eth(bars5)}, START + pd.Timedelta("5h30s"))
    assert tick(config(book(Const(1.0))), store, src, quiet)["books"][0]["decision"] == "traded"

    src.now = START + pd.Timedelta("7h30s")
    result = tick(config(book(Const(1.0, **change))), store, src, quiet)
    assert result["books"][0]["decision"] == "config_conflict" and not result["ok"]
    assert [f["reason"] for f in rows(store, "SELECT reason FROM fills")] == ["signal"]
    b = store.book("b1")
    assert (b["symbol"], b["timeframe"], b["paused_until"]) == ("BTCUSDT", "1h", None)
    (ev,) = rows(store, "SELECT * FROM events WHERE kind='config_conflict'")
    assert "book_id" in ev["message"] and "ETHUSDT" not in ev["message"]
    tick(config(book(Const(1.0, **change))), store, src, quiet)
    assert len(rows(store, "SELECT * FROM events WHERE kind='config_conflict'")) == 1


def _overnight_drop() -> pd.DataFrame:
    """Flat at 100 on day 1, -9.7% between 00:00 and 01:00 on day 2, then flat."""
    prices = np.full(2 * 288 + 60, 100.0)
    prices[288 : 288 + 12] = 100 * 0.9915 ** np.arange(1, 13)
    prices[288 + 12 :] = prices[288 + 11]
    return path_bars(prices)


def test_daily_loss_baseline_is_midnight_equity_after_sleep(store, quiet):
    cfg = config(book(Const(1.0)))
    src = FakeSource({"BTCUSDT": _overnight_drop()}, START + pd.Timedelta("5h30s"))
    tick(cfg, store, src, quiet)
    src.now = START + pd.Timedelta("23h50m30s")
    tick(cfg, store, src, quiet)
    src.now = START + DAY + pd.Timedelta("10h30s")  # the Mac slept through midnight
    result = tick(cfg, store, src, quiet)
    assert result["books"][0]["kill_switch"] is True
    assert store.book("b1")["day_start_equity"] == pytest.approx(9995.0, rel=1e-6)
    (kill,) = rows(store, "SELECT * FROM fills WHERE reason='kill_switch'")
    assert kill["fill_bar_ts"] == "2021-03-02T10:00:00Z"


class SellOnDayTwo(Const):
    def decide(self, bars, other=None):
        return Decision(0.0 if bars.index[-1] >= START + DAY else 1.0, "test")


def test_daily_loss_baseline_uses_the_position_held_at_midnight(store, quiet):
    cfg = config(book(SellOnDayTwo()))
    src = FakeSource({"BTCUSDT": _overnight_drop()}, START + pd.Timedelta("5h30s"))
    tick(cfg, store, src, quiet)
    src.now = START + DAY + pd.Timedelta("10h30s")  # wakes, sells at the 10:00 open, then marks
    tick(cfg, store, src, quiet)
    assert [f["side"] for f in rows(store, "SELECT side FROM fills ORDER BY id")] == ["buy", "sell"]
    assert store.book("b1")["day_start_equity"] == pytest.approx(9995.0, rel=1e-6)
    assert store.book("b1")["paused_until"] == "2021-03-03T00:00:00Z"
    assert len(rows(store, "SELECT * FROM events WHERE kind='kill_switch'")) == 1


def test_kill_switch_rechecks_when_the_fill_bar_arrives_late(store):
    prices = np.full(2 * 288, 100.0)
    prices[60:72] = 100 * 0.99 ** np.arange(1, 13)
    prices[72:] = prices[71]
    bars = path_bars(prices)
    cfg = config(book(Const(1.0)))
    src = FakeSource({"BTCUSDT": bars}, START + pd.Timedelta("5h30s"))
    tick(cfg, store, src, Notifier(False))
    src.now = START + pd.Timedelta("5h45m30s")
    src.frames = {"BTCUSDT": bars[bars.index < START + pd.Timedelta("5h45m")]}  # 05:45 not served yet
    assert tick(cfg, store, src, Notifier(False))["books"][0]["kill_switch"] is False
    src.frames = {"BTCUSDT": bars}
    tick(cfg, store, src, Notifier(False))
    (kill,) = rows(store, "SELECT * FROM fills WHERE reason='kill_switch'")
    assert kill["fill_bar_ts"] == "2021-03-01T05:45:00Z"


def test_fill_on_a_halted_bar_is_deferred_like_the_harness(store, quiet):
    bars5 = make_5m(12 * 24 * 7, seed=11, vol=0.004)
    strategy = SmaTrendDemo("BTCUSDT", "1h", fast=3, slow=8)
    clean = resample(bars5, "1h", incomplete="drop")
    held = run_backtest(clean, target_series(strategy, clean), 5.0).bars["position"].to_numpy()
    fill_hours = np.flatnonzero(np.diff(np.concatenate([[0.0], held])) != 0)
    halt = clean.index[fill_hours[len(fill_hours) // 2]]
    bars5.loc[(bars5.index >= halt) & (bars5.index < halt + pd.Timedelta("1h")), "volume"] = 0.0
    hourly = resample(bars5, "1h", incomplete="drop")

    harness = run_backtest(hourly, target_series(strategy, hourly), 5.0)
    assert harness.params["fills_deferred"] >= 1
    pos = harness.bars["position"].to_numpy()
    changed = np.flatnonzero(np.diff(np.concatenate([[0.0], pos])) != 0)
    changed = changed[changed < len(hourly) - 1]
    expected = [(iso(hourly.index[j]), pos[j], hourly["open"].iloc[j]) for j in changed]

    cfg = config(book(strategy, max_daily_loss=0.0))
    src = FakeSource({"BTCUSDT": bars5}, bars5.index[0])
    for now in pd.date_range(
        bars5.index[0] + pd.Timedelta("15min"), hourly.index[-1] - pd.Timedelta("5min"), freq="15min"
    ):
        src.now = now + pd.Timedelta("30s")
        tick(cfg, store, src, quiet)
    fills = rows(store, "SELECT * FROM fills WHERE reason='signal' ORDER BY fill_bar_ts")
    assert [(f["fill_bar_ts"], f["target"], f["price"]) for f in fills] == expected
    (ev,) = rows(store, "SELECT * FROM events WHERE kind='halted'")
    assert json.loads(ev["data_json"])["fill_bar"] == iso(halt)


def test_turning_a_book_off_shows_on_the_dashboard(store, quiet):
    src = FakeSource({"BTCUSDT": make_5m(600, seed=2)}, START + pd.Timedelta("5h30s"))
    tick(config(book(Const(1.0))), store, src, quiet)
    assert store.book("b1")["enabled"] == 1
    src.now = START + pd.Timedelta("6h30s")
    result = tick(config(book(Const(1.0), enabled=False)), store, src, quiet)
    assert result["books"] == []
    assert store.book("b1")["enabled"] == 0
    assert len(rows(store, "SELECT * FROM decisions")) == 1
