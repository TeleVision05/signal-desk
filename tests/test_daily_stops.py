import pandas as pd

from profit_engine.daily_backtest import StopConfig, _stop_hit, run_long_only_daily


def test_hard_stop_triggers_on_low():
    cfg = StopConfig(hard_pct=0.10, trailing_pct=None)
    bar = pd.Series({"Open": 100, "High": 101, "Low": 89, "Close": 90})
    hit, px = _stop_hit(bar, entry=100.0, peak=101.0, cfg=cfg)
    assert hit
    assert abs(px - 90.0) < 1e-9  # hard floor = 90


def test_trailing_stop_binds_after_rally():
    cfg = StopConfig(hard_pct=0.50, trailing_pct=0.10)
    bar = pd.Series({"Open": 140, "High": 140, "Low": 125, "Close": 126})
    # peak 150 → trail floor 135; low 125 hits
    hit, px = _stop_hit(bar, entry=100.0, peak=150.0, cfg=cfg)
    assert hit
    assert abs(px - 135.0) < 1e-9


def test_backtest_respects_flat_signal():
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    px = pd.Series(range(100, 130), index=idx, dtype=float)
    ohlc = pd.DataFrame({"Open": px, "High": px + 1, "Low": px - 1, "Close": px})
    sig = pd.Series(0, index=idx)
    sig.iloc[5:15] = 1
    res = run_long_only_daily(ohlc, sig, stop=StopConfig(hard_pct=0.5, trailing_pct=0.5))
    assert not res.trades.empty
    assert set(res.trades["side"]) <= {"BUY", "SELL"}
