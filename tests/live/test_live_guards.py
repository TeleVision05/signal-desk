"""Guards that must never be silent: a frozen feed, halts and gaps at the fill bar (parity with
the harness), honest catch-up counts, and a daily-loss pause set by another process."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from live_helpers import Const, FakeSource, book, config, make_5m, rows

from profit_engine.intraday.backtest import run_backtest
from profit_engine.intraday.resample import resample
from profit_engine.live import runner
from profit_engine.live.runner import tick
from profit_engine.live.store import EventSpec, iso
from profit_engine.live.strategy import SmaTrendDemo, target_series

H = pd.Timedelta("1h")


def test_frozen_feed_after_a_decision_is_reported_stale(store, quiet):
    bars5 = make_5m(600, seed=3)
    cfg = config(book(Const(1.0)))
    first = tick(cfg, store, FakeSource({"BTCUSDT": bars5}, "2021-03-01 05:00:30+00:00"), quiet)
    assert first["books"][0]["decision"] == "traded"
    frozen = bars5[bars5.index < pd.Timestamp("2021-03-01 05:10+00:00")]  # Binance stops publishing
    src = FakeSource({"BTCUSDT": frozen}, "2021-03-01 10:00:30+00:00")  # five hours later
    for _ in range(3):
        result = tick(cfg, store, src, quiet)
        assert result["books"][0]["decision"] == "stale_data"
        assert result["books"][0]["ok"] is False and result["status"] == "degraded"
    (ev,) = rows(store, "SELECT * FROM events WHERE kind='stale_data'")  # logged once, not per tick
    assert "too old to trade on" in ev["message"]
    assert rows(store, "SELECT status FROM heartbeat WHERE component='runner'") == [{"status": "degraded"}]


def _harness_fills(bars5: pd.DataFrame, strategy) -> list[tuple]:
    hourly = resample(bars5, "1h", incomplete="drop")
    pos = run_backtest(hourly, target_series(strategy, hourly), 5.0).bars["position"].to_numpy()
    changed = np.flatnonzero(np.diff(np.concatenate([[0.0], pos])) != 0)
    changed = changed[changed < len(hourly) - 1]  # the harness force-exits on its last bar; live does not
    return [(iso(hourly.index[j]), pos[j], hourly["open"].iloc[j]) for j in changed]


def _replay(store, quiet, bars5: pd.DataFrame, strategy) -> list[tuple]:
    cfg = config(book(strategy, max_daily_loss=0.0))
    src = FakeSource({"BTCUSDT": bars5}, bars5.index[0])
    end = resample(bars5, "1h", incomplete="drop").index[-1]
    for now in pd.date_range(bars5.index[0] + pd.Timedelta("15min"), end - pd.Timedelta("5min"), freq="15min"):
        src.now = now + pd.Timedelta("30s")
        tick(cfg, store, src, quiet)
    fills = rows(store, "SELECT * FROM fills WHERE reason='signal' ORDER BY fill_bar_ts")
    return [(f["fill_bar_ts"], f["target"], f["price"]) for f in fills]


def _a_fill_hour(bars5: pd.DataFrame, strategy) -> pd.Timestamp:
    clean = _harness_fills(bars5, strategy)
    return pd.Timestamp(clean[3][0])


@pytest.mark.parametrize("damage", ["halt_first_5m", "drop_first_5m"])
def test_halt_or_gap_on_the_first_5m_of_the_fill_bar_matches_the_harness(store, quiet, damage):
    bars5 = make_5m(12 * 24 * 7, seed=11, vol=0.004)
    strategy = SmaTrendDemo("BTCUSDT", "1h", fast=3, slow=8)
    hit = _a_fill_hour(bars5, strategy)
    if damage == "halt_first_5m":
        bars5.loc[hit, "volume"] = 0.0  # Binance prints a stale open; the rest of the hour trades
    else:
        bars5 = bars5.drop(hit)  # Binance has no data for that 5m bar; the hour is dropped
    expected = _harness_fills(bars5, strategy)
    assert len(expected) >= 6
    assert _replay(store, quiet, bars5, strategy) == expected
    assert rows(store, "SELECT * FROM events WHERE kind='catch_up'") == []  # nothing was missed


def test_in_progress_fill_bar_without_trades_yet_waits_without_a_halt_event(store, quiet):
    bars5 = make_5m(600, seed=3)
    bars5.loc[pd.Timestamp("2021-03-01 05:00+00:00"), "volume"] = 0.0  # 2 s old: no trades yet is normal
    src = FakeSource({"BTCUSDT": bars5}, "2021-03-01 05:00:02+00:00")
    result = tick(config(book(Const(1.0))), store, src, quiet)
    assert result["books"][0]["decision"] == "waiting_for_trades"
    assert rows(store, "SELECT * FROM events WHERE kind='halted'") == []
    src.now = pd.Timestamp("2021-03-01 05:05:30+00:00")  # the next 5m bar trades: fill at the 05:00 open
    assert tick(config(book(Const(1.0))), store, src, quiet)["books"][0]["decision"] == "traded"
    (fill,) = rows(store, "SELECT * FROM fills")
    assert fill["fill_bar_ts"] == "2021-03-01T05:00:00Z"
    assert fill["price"] == bars5.at[pd.Timestamp("2021-03-01 05:00+00:00"), "open"]


def test_catch_up_does_not_count_bars_dropped_for_missing_data(store, quiet):
    bars5 = make_5m(600, seed=3)
    holed = bars5.drop(pd.Timestamp("2021-03-01 06:30+00:00"))  # the 06:00 hour has no complete bar
    cfg = config(book(Const(1.0)))
    src = FakeSource({"BTCUSDT": holed}, "2021-03-01 06:00:30+00:00")
    tick(cfg, store, src, quiet)  # decides 05:00
    src.now = pd.Timestamp("2021-03-01 08:00:30+00:00")  # 06:00 is missing data, 07:00 is due
    assert tick(cfg, store, src, quiet)["books"][0]["decision"] == "held"
    decided = [r["bar_open_ts"] for r in rows(store, "SELECT bar_open_ts FROM decisions ORDER BY bar_open_ts")]
    assert decided == ["2021-03-01T05:00:00Z", "2021-03-01T07:00:00Z"]
    assert rows(store, "SELECT * FROM events WHERE kind='catch_up'") == []
    src.now = pd.Timestamp("2021-03-01 11:00:30+00:00")  # two real bars (08:00, 09:00) go unchecked
    tick(cfg, store, src, quiet)
    (ev,) = rows(store, "SELECT * FROM events WHERE kind='catch_up'")
    assert json.loads(ev["data_json"]) == {"skipped_bars": 2}
    assert "asleep or offline, or those bars could not be traded" in ev["message"]


def test_a_pause_set_by_another_process_is_applied_in_the_bar_commit(store, quiet):
    bars5 = make_5m(600, seed=3)
    spec = book(Const(1.0))
    now = pd.Timestamp("2021-03-01 05:00:30+00:00")
    src = FakeSource({"BTCUSDT": bars5}, now)
    store.upsert_book(spec, now)
    ctx = runner.TickContext(store, src, config(spec).settings, quiet, now)
    seen = store.book("b1")  # this process reads the book: not paused
    kill = {
        "now": now, "price": 100.0, "fill_ts": pd.Timestamp("2021-03-01 04:55+00:00"),
        "paused_until": pd.Timestamp("2021-03-02", tz="UTC"), "cost_bps": 5.0,
        "event": EventSpec("kill_switch", "stopped for the day", {}, "2021-03-01"),
    }  # fmt: skip
    assert store.trigger_kill_switch("b1", kill).claimed  # ...another process trips the loss limit
    assert runner.decide_book(ctx, spec, seen) == "held"
    b = store.book("b1")
    assert b["target"] == 0.0 and b["units"] == 0.0
    assert rows(store, "SELECT reason FROM fills WHERE reason='signal'") == []
    (dec,) = rows(store, "SELECT * FROM decisions")
    assert dec["target"] == 0.0 and "paused" in dec["explanation"]
    assert json.loads(dec["details_json"])["strategy_target"] == 1.0
