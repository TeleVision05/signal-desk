"""Paper books decide each daily bar once and fill like run_long_only_daily (next open, stops)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from profit_engine import decisions
from profit_engine.daily_backtest import StopConfig, run_long_only_daily
from profit_engine.ledger import PaperLedger
from profit_engine.paper_ml import missing_inputs
from profit_engine.paper_ml_crypto import closed_days
from profit_engine.paper_step import BookSpec, step

SPEC = BookSpec(
    run_id="test-book",
    book="tqqq",
    strategy="test",
    asset="TQQQ",
    cost_bps=10.0,
    stop=StopConfig(hard_pct=0.08, trailing_pct=0.12),
    mark_source="test",
    threshold=0.55,
    config={},
)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    monkeypatch.setattr(decisions, "LOG_PATH", tmp_path / "decision_log.json")
    return PaperLedger(tmp_path / "paper.sqlite")


def _ohlc(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 60 * np.exp(np.cumsum(rng.normal(0, 0.04, n)))
    open_ = np.r_[60.0, close[:-1]] * np.exp(rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.02, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.02, n)))
    idx = pd.bdate_range("2026-01-02", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def _fills(ledger: PaperLedger) -> list[tuple]:
    with ledger._sqlite_conn() as conn:
        rows = conn.execute("SELECT ts, side, price, source FROM fills ORDER BY id").fetchall()
    return [(str(pd.Timestamp(r["ts"]).date()), r["side"], round(r["price"], 9), r["source"]) for r in rows]


def test_the_same_bar_is_decided_once_however_often_the_score_moves(ledger):
    """Oct 30 2026: p for one bar went 0.544, 0.600, 0.544, 0.556, 0.544, 0.600 and it traded each time."""
    ohlc = _ohlc(5)
    first = step(SPEC, ohlc, 0, 0.544, ledger=ledger)
    assert first["action"] == "hold"
    for p in (0.600, 0.544, 0.556, 0.600):
        again = step(SPEC, ohlc, int(p >= 0.55), p, ledger=ledger)
        assert again["action"] == "already_decided" and not again["traded"]
        assert again["signal"] == 0 and again["proba"] == 0.544
    assert _fills(ledger) == []
    assert len([r for r in decisions.load_decisions(run_id=SPEC.run_id) if r.get("state_version")]) == 1


def test_a_decision_fills_at_the_next_bars_open_not_its_own(ledger):
    ohlc = _ohlc(6)
    step(SPEC, ohlc.iloc[:4], 1, 0.61, ledger=ledger)
    assert _fills(ledger) == []  # decided at bar 4's close; nothing is bought at a past price
    out = step(SPEC, ohlc.iloc[:5], 1, 0.62, ledger=ledger)
    assert out["action"] == "signal_entry"
    assert _fills(ledger)[0] == (str(ohlc.index[4].date()), "BUY", round(ohlc["Open"].iloc[4], 9), "ml_signal")


def test_a_sync_after_the_latest_day_slid_back_does_not_trade(ledger):
    ohlc = _ohlc(6)
    step(SPEC, ohlc.iloc[:5], 1, 0.61, ledger=ledger)
    out = step(SPEC, ohlc.iloc[:4], 0, 0.40, ledger=ledger)  # Yahoo returned one day fewer
    assert out["action"] == "already_decided"
    assert _fills(ledger) == []


def test_missing_inputs_record_nothing_and_the_next_sync_decides(ledger):
    ohlc = _ohlc(5)
    out = step(SPEC, ohlc, 1, 0.9, ledger=ledger, problem="VIX prices could not be downloaded")
    assert out["action"] == "skipped" and "VIX" in out["problem"]
    assert decisions.load_decisions(run_id=SPEC.run_id) == []
    assert step(SPEC, ohlc, 1, 0.6, ledger=ledger)["action"] == "hold"


@pytest.mark.parametrize("seed", range(6))
def test_daily_syncs_reproduce_the_backtest_fills(ledger, seed):
    ohlc = _ohlc(160, seed)
    rng = np.random.default_rng(100 + seed)
    signals = pd.Series((rng.random(len(ohlc)) < 0.55).astype(int), index=ohlc.index)
    for i in range(1, len(ohlc) + 1):
        want = int(signals.iloc[i - 1])
        step(SPEC, ohlc.iloc[:i], want, 0.6 if want else 0.4, ledger=ledger, initial_cash=1000.0)
        step(SPEC, ohlc.iloc[:i], 1 - want, 0.5, ledger=ledger)  # a second sync with another answer
    bt = run_long_only_daily(ohlc, signals, initial_cash=1000.0, round_trip_cost_bps=10.0, stop=SPEC.stop)
    want = [
        (str(pd.Timestamp(r.date).date()), r.side, round(r.price, 9), "stop" if r.reason == "stop" else "ml_signal")
        for r in bt.trades.itertuples()
    ]
    assert len(want) > 10 and any(w[3] == "stop" for w in want)
    assert _fills(ledger) == want
    equity = ledger.get_position(SPEC.run_id)
    final = equity["cash"] + equity["shares"] * ohlc["Close"].iloc[-1]
    assert final == pytest.approx(bt.equity.iloc[-1], rel=1e-9)


def test_missing_inputs_names_the_short_or_failed_series():
    ohlc = _ohlc(5)
    full = ohlc["Close"]
    assert missing_inputs(ohlc, {"spy": full, "vix": full}) is None
    assert missing_inputs(ohlc, {"spy": full, "vix": None}) == "VIX prices could not be downloaded"
    assert "QQQ prices stop at" in missing_inputs(ohlc, {"qqq": full.iloc[:-1]})


def test_the_utc_day_still_in_progress_is_dropped():
    idx = pd.date_range("2026-11-03", periods=3, freq="D")
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx)
    now = datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc)
    assert list(closed_days(df, now).index) == list(idx[:2])
    assert list(closed_days(df, datetime(2026, 11, 6, 0, 0, tzinfo=timezone.utc)).index) == list(idx)
