import pandas as pd

from profit_engine.strategies import faber_sma, gem_dual_momentum


def _toy_closes():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    spy = pd.Series(range(100, 124), index=idx, dtype=float)
    veu = spy * 0.95
    agg = pd.Series(100.0, index=idx)
    bil = pd.Series(100.0, index=idx)
    # Make VEU stronger in last year
    veu.iloc[-12:] = spy.iloc[-12:] * 1.2
    return pd.DataFrame({"SPY": spy, "VEU": veu, "AGG": agg, "BIL": bil})


def test_gem_picks_relative_winner_when_positive():
    closes = _toy_closes()
    sig = gem_dual_momentum(closes)
    assert sig.iloc[-1] == "VEU"


def test_gem_goes_risk_off_when_both_negative():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    spy = pd.Series(range(124, 100, -1), index=idx, dtype=float)
    veu = spy * 0.9
    agg = pd.Series(100.0, index=idx)
    closes = pd.DataFrame({"SPY": spy, "VEU": veu, "AGG": agg})
    sig = gem_dual_momentum(closes)
    assert sig.iloc[-1] == "AGG"


def test_faber_uses_sma():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    # Rising then falling through SMA
    spy = pd.Series([100 + i for i in range(24)], index=idx, dtype=float)
    bil = pd.Series(100.0, index=idx)
    closes = pd.DataFrame({"SPY": spy, "BIL": bil})
    sig = faber_sma(closes, sma_months=10)
    assert sig.iloc[-1] == "SPY"