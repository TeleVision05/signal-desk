"""Timing contract: closed bars only, fill at the next bar's open, catch-up, stale data."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from live_helpers import POISON, Const, FakeSource, book, config, make_5m, rows

from profit_engine.live.market import split_closed
from profit_engine.live.runner import tick

H = pd.Timedelta("1h")


@pytest.fixture
def bars5():
    return make_5m(600, seed=3)


def test_fill_at_next_bar_open_with_cost(store, quiet, bars5):
    tick(config(book(Const(1.0))), store, FakeSource({"BTCUSDT": bars5}, "2021-03-01 05:00:30+00:00"), quiet)
    (fill,) = rows(store, "SELECT * FROM fills")
    fill_bar = pd.Timestamp("2021-03-01 05:00+00:00")
    price = bars5.at[fill_bar, "open"]
    assert fill["decision_bar_ts"] == "2021-03-01T04:00:00Z"
    assert fill["fill_bar_ts"] == "2021-03-01T05:00:00Z"
    assert fill["price"] == price
    assert fill["cost"] == pytest.approx(10_000 * 5 / 1e4)  # 5 bps on a 0 -> 100% move
    assert fill["units_after"] == pytest.approx((10_000 - 5.0) / price)
    b = store.book("b1")
    assert b["cash"] == pytest.approx(0.0, abs=1e-9) and b["target"] == 1.0


def test_flat_to_flat_does_not_trade(store, quiet, bars5):
    tick(config(book(Const(0.0))), store, FakeSource({"BTCUSDT": bars5}, "2021-03-01 05:00:30+00:00"), quiet)
    assert rows(store, "SELECT * FROM fills") == []
    assert len(rows(store, "SELECT * FROM decisions")) == 1


def test_in_progress_bars_are_never_used(store, quiet, bars5):
    strategy = Const(1.0)
    now = pd.Timestamp("2021-03-01 05:20:30+00:00")  # mid-way through the 05:00 1h bar and the 05:20 5m bar
    tick(config(book(strategy)), store, FakeSource({"BTCUSDT": bars5}, now), quiet)
    (window,) = strategy.windows
    assert window.index[-1] == pd.Timestamp("2021-03-01 04:00+00:00")  # latest CLOSED 1h bar
    assert (window["close"] < POISON).all() and (window["high"] < POISON).all()
    assert window["close"].iloc[-1] == bars5.at[pd.Timestamp("2021-03-01 04:55+00:00"), "close"]
    (dec,) = rows(store, "SELECT * FROM decisions")
    assert dec["bar_open_ts"] == "2021-03-01T04:00:00Z"
    (mark,) = rows(store, "SELECT * FROM equity_marks")
    assert mark["ts"] == "2021-03-01T05:20:00Z"  # close of the 05:15 bar, not the in-progress 05:20 bar
    assert mark["price"] == bars5.at[pd.Timestamp("2021-03-01 05:15+00:00"), "close"]


def test_split_closed_waits_for_the_settle_margin(bars5):
    frame = bars5.iloc[:10]
    end = frame.index[4] + pd.Timedelta("5min")
    closed, live = split_closed(frame, end + pd.Timedelta("1s"), pd.Timedelta("2s"))
    assert closed.index[-1] == frame.index[3] and live.index[0] == frame.index[4]
    closed, _ = split_closed(frame, end + pd.Timedelta("2s"), pd.Timedelta("2s"))
    assert closed.index[-1] == frame.index[4]


def test_no_fill_until_the_next_bar_has_started(store, quiet, bars5):
    cut = bars5[bars5.index < pd.Timestamp("2021-03-01 05:00+00:00")]  # the 05:00 bar does not exist yet
    result = tick(config(book(Const(1.0))), store, FakeSource({"BTCUSDT": cut}, "2021-03-01 05:00:03+00:00"), quiet)
    assert result["books"][0]["decision"] == "waiting_for_next_bar"
    assert rows(store, "SELECT * FROM decisions") == []


def test_catch_up_after_six_hour_gap_decides_once(store, quiet, bars5):
    strategy = Const(1.0)
    cfg = config(book(strategy))
    src = FakeSource({"BTCUSDT": bars5}, "2021-03-01 05:00:30+00:00")
    tick(cfg, store, src, quiet)
    src.now = pd.Timestamp("2021-03-01 11:00:30+00:00")  # the Mac slept for 6 hours
    result = tick(cfg, store, src, quiet)
    assert result["books"][0]["decision"] == "held"
    decided = [r["bar_open_ts"] for r in rows(store, "SELECT bar_open_ts FROM decisions ORDER BY bar_open_ts")]
    assert decided == ["2021-03-01T04:00:00Z", "2021-03-01T10:00:00Z"]  # latest bar only, no replay
    assert len(strategy.windows) == 2
    (ev,) = rows(store, "SELECT * FROM events WHERE kind='catch_up'")
    assert json.loads(ev["data_json"]) == {"skipped_bars": 5}
    (dec,) = rows(store, "SELECT skipped_bars FROM decisions WHERE bar_open_ts='2021-03-01T10:00:00Z'")
    assert dec["skipped_bars"] == 5
    tick(cfg, store, src, quiet)
    assert len(rows(store, "SELECT * FROM events WHERE kind='catch_up'")) == 1


def test_stale_data_is_skipped_with_an_event(store, quiet, bars5):
    feed = bars5[bars5.index < pd.Timestamp("2021-03-01 02:00+00:00")]  # the feed stopped at 02:00
    cfg = config(book(Const(1.0)))
    src = FakeSource({"BTCUSDT": feed}, "2021-03-01 05:00:30+00:00")
    result = tick(cfg, store, src, quiet)
    assert result["books"][0]["decision"] == "stale_data"
    assert rows(store, "SELECT * FROM decisions") == []
    tick(cfg, store, src, quiet)
    assert len(rows(store, "SELECT * FROM events WHERE kind='stale_data'")) == 1  # logged once per bar


def test_missing_5m_bar_drops_the_decision_bar_and_logs_a_gap(store, quiet, bars5):
    holed = bars5.drop(pd.Timestamp("2021-03-01 04:30+00:00"))
    strategy = Const(1.0)
    result = tick(config(book(strategy)), store, FakeSource({"BTCUSDT": holed}, "2021-03-01 05:00:40+00:00"), quiet)
    assert result["books"][0]["decision"] == "traded"
    # Like the harness (whose 1h index skips the incomplete 04:00 bar): decide on 03:00 and fill
    # at the next bar that exists, 05:00, which is the bar now in progress (never a past price).
    assert strategy.windows[0].index[-1] == pd.Timestamp("2021-03-01 03:00+00:00")
    (fill,) = rows(store, "SELECT * FROM fills")
    assert (fill["decision_bar_ts"], fill["fill_bar_ts"]) == ("2021-03-01T03:00:00Z", "2021-03-01T05:00:00Z")
    assert fill["price"] == holed.at[pd.Timestamp("2021-03-01 05:00+00:00"), "open"]
    assert len(rows(store, "SELECT * FROM events WHERE kind='gap'")) == 1


def test_exchange_down_records_event_and_trades_nothing(store, quiet, bars5):
    src = FakeSource({"BTCUSDT": bars5}, "2021-03-01 05:00:30+00:00")
    src.down = True
    result = tick(config(book(Const(1.0))), store, src, quiet)
    assert result["status"] == "exchange_unreachable"
    assert rows(store, "SELECT * FROM fills") == []
    (ev,) = rows(store, "SELECT * FROM events WHERE kind='data_error'")
    assert "ConnectionError" not in ev["message"]


def test_bad_fill_price_is_refused(store, quiet, bars5):
    bad = bars5.copy()
    bad.loc[pd.Timestamp("2021-03-01 05:00+00:00"), "open"] *= 2  # +100% against the decision close
    result = tick(config(book(Const(1.0))), store, FakeSource({"BTCUSDT": bad}, "2021-03-01 05:00:30+00:00"), quiet)
    assert result["books"][0]["decision"] == "bad_price"
    assert rows(store, "SELECT * FROM fills") == []
    assert len(rows(store, "SELECT * FROM events WHERE kind='bad_price'")) == 1
