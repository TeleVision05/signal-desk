"""A stateful strategy's own state (the trend book's band) is brought up to date through bars the
runner never decided (Mac asleep or offline, stale feed, halted fill bar), and a new book starts
from a band run over the history before its first decision, not from flat.

Nothing is traded for those bars: only the state fed to the decision on the newest bar changes.
"""

from __future__ import annotations

import pandas as pd
from live_helpers import FakeSource, book, config, make_5m, rows
from parity_helpers import bars, raw_5m, research

from profit_engine.intraday.resample import resample
from profit_engine.live.runner import tick
from profit_engine.live.store import iso
from profit_engine.live.strategies.voltarget import VolTargetTrend
from profit_engine.live.strategy import Decision, StrategyMeta, target_series

H = pd.Timedelta("1h")


class Counter:
    """Stateful test strategy: every decided bar adds 0.01 to its own previous target."""

    name, stateful, other_symbol = "counter", True, None

    def __init__(self, warmup: int = 10, state_bars: int = 50) -> None:
        self.symbol, self.timeframe, self.warmup_bars, self.state_bars = "BTCUSDT", "1h", warmup, state_bars
        self.meta = StrategyMeta(version="1")
        self.seen: list[tuple[pd.Timestamp, int]] = []

    def decide(self, bars: pd.DataFrame, other=None, prev_target: float = 0.0) -> Decision:
        self.seen.append((bars.index[-1], len(bars)))
        return Decision(round(min(1.0, prev_target + 0.01), 10), "step")


def _run(strategy, five: pd.DataFrame, bar_times, store, quiet) -> None:
    cfg = config(book(strategy, max_daily_loss=0.0))
    src = FakeSource({strategy.symbol: five}, five.index[0])
    for bar in bar_times:
        src.now = bar + H + pd.Timedelta("30s")
        assert tick(cfg, store, src, quiet)["ok"]


def _decided(store) -> list[tuple[str, float]]:
    return [(d["bar_open_ts"], d["target"]) for d in rows(store, "SELECT * FROM decisions ORDER BY bar_open_ts")]


def test_bars_slept_through_advance_the_state_with_full_windows_and_no_trades(store, quiet):
    s = Counter()
    five = make_5m(12 * 400, seed=5)
    hourly = resample(five, "1h", incomplete="drop")
    a, b = hourly.index[60], hourly.index[66]  # asleep for the five bars between them
    _run(s, five, [a], store, quiet)
    first = store.strategy_target("b1")
    s.seen.clear()
    _run(s, five, [b], store, quiet)
    assert [t for t, _ in s.seen] == list(hourly.index[61:67])  # the five missed bars, then b
    assert {n for _, n in s.seen} == {s.warmup_bars}  # each with the window the runner would have used
    assert store.strategy_target("b1") == round(first + 0.06, 10)
    (catch_up,) = rows(store, "SELECT message, data_json FROM events WHERE kind='catch_up'")
    assert "without trading" in catch_up["message"]
    assert len(rows(store, "SELECT * FROM fills")) == 2  # one fill per decided bar, none for the missed ones


def test_a_long_absence_fetches_the_older_history_it_needs_and_stops_at_state_bars(store, quiet):
    s = Counter(warmup=10, state_bars=50)
    five = make_5m(12 * 400, seed=6)
    hourly = resample(five, "1h", incomplete="drop")
    a, b = hourly.index[60], hourly.index[260]  # 199 bars missed: more than state_bars
    _run(s, five, [a], store, quiet)
    s.seen.clear()
    _run(s, five, [b], store, quiet)
    assert [t for t, _ in s.seen] == list(hourly.index[210:261])
    assert {n for _, n in s.seen} == {10}


def test_a_new_book_runs_the_state_over_the_history_before_its_first_decision(store, quiet):
    s = Counter(warmup=10, state_bars=50)
    five = make_5m(12 * 400, seed=7)
    hourly = resample(five, "1h", incomplete="drop")
    _run(s, five, [hourly.index[200]], store, quiet)
    assert [t for t, _ in s.seen] == list(hourly.index[150:201])
    assert _decided(store) == [(iso(hourly.index[200]), 0.51)]


def test_a_stateless_strategy_is_not_replayed(store, quiet):
    s = Counter()
    s.stateful = False
    s.decide = lambda bars, other=None: s.seen.append((bars.index[-1], len(bars))) or 0.5
    five = make_5m(12 * 400, seed=8)
    hourly = resample(five, "1h", incomplete="drop")
    _run(s, five, [hourly.index[60], hourly.index[66]], store, quiet)
    assert [t for t, _ in s.seen] == [hourly.index[60], hourly.index[66]]


def test_trend_book_that_sleeps_every_night_follows_the_research_band(store, quiet):
    """Real cached BTC bars, ticked hourly but never 23:00-05:00 UTC and not at all for four days:
    every decision equals the band run over every bar (what the research does)."""
    strategy = VolTargetTrend("BTCUSDT")
    n, end = strategy.warmup_bars, pd.Timestamp("2026-01-21", tz="UTC")
    five = raw_5m("BTCUSDT")
    five = five[(five.index >= end - H * (n + 24 * 26)) & (five.index < end)]
    hourly = resample(five, "1h", incomplete="drop")
    want = target_series(strategy, hourly)  # every bar, starting flat at the first full window
    t, hour = hourly.index, hourly.index.hour
    away = (t >= pd.Timestamp("2026-01-05", tz="UTC")) & (t < pd.Timestamp("2026-01-09", tz="UTC"))
    awake = t[(t >= t[n - 1]) & ~away & (hour >= 6) & (hour < 23)]
    _run(strategy, five, awake[:-1], store, quiet)
    got = _decided(store)
    assert len(got) > 300
    assert got == [(iso(t), want[t]) for t in awake[:-1]]
    replayed = [d for d in rows(store, "SELECT details_json FROM decisions") if '"replayed_bars"' in d["details_json"]]
    assert len(replayed) >= 20  # the band really was advanced, every morning


def test_a_new_trend_book_starts_where_the_research_band_is(store, quiet):
    """ETH, 2025-06-09 02:00 UTC: one band step from flat gives 0%, the research held 20%."""
    vt = research("voltarget_trend")
    hourly = bars("ETHUSDT", "1h")
    want = vt.vt_target(hourly, "1h", vt.GRID_BY_NAME["vt_1h_slow4_z_s30"]["params"])
    t = pd.Timestamp("2025-06-09 02:00", tz="UTC")
    strategy = VolTargetTrend("ETHUSDT")
    window = hourly.loc[:t].iloc[-strategy.warmup_bars :]
    assert want[t] == 0.2 and strategy.decide(window, None, prev_target=0.0).target == 0.0  # the old flat start
    five = raw_5m("ETHUSDT")
    five = five[(five.index >= t - H * (2 * strategy.warmup_bars + 200)) & (five.index < t + 2 * H)]
    _run(strategy, five, [t], store, quiet)
    assert _decided(store) == [(iso(t), 0.2)]
