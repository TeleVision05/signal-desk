"""(i) every feature is causal: computing it on df[:k] gives the same rows as on the full df."""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import pytest

from profit_engine.intraday import features
from profit_engine.intraday.resample import resample

N_BARS = 1500
CUTS = 8


def _cuts(n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return sorted([*rng.integers(5, n - 1, CUTS).tolist(), n - 1])


def _assert_causal(fn, df: pd.DataFrame, seed: int) -> None:
    full = fn(df)
    for k in _cuts(len(df), seed):
        part = fn(df.iloc[:k])
        pd.testing.assert_frame_equal(part, full.iloc[:k], check_exact=False, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", sorted(features.FEATURES))
def test_i_feature_is_causal(name, bar_factory):
    df = bar_factory(n=N_BARS, seed=11)
    _assert_causal(features.FEATURES[name], df, seed=zlib.crc32(name.encode()))


@pytest.mark.parametrize("name", sorted(features.FEATURES))
def test_i_feature_does_not_mutate_input(name, bar_factory):
    df = bar_factory(n=400, seed=12)
    before = df.copy()
    features.FEATURES[name](df)
    pd.testing.assert_frame_equal(df, before)


def test_i_cross_asset_and_feature_frame_are_causal(bar_factory):
    btc = bar_factory(n=N_BARS, seed=13)
    eth = bar_factory(n=N_BARS, seed=14).drop(btc.index[[100, 700]])  # misaligned bars
    _assert_causal(lambda d: features.cross_asset(d, eth), btc, seed=1)
    full = features.make_feature_frame(btc, eth)
    assert btc.index[100] not in full.index
    for k in _cuts(N_BARS, 2):
        part = features.make_feature_frame(btc.iloc[:k], eth)
        pd.testing.assert_frame_equal(part, full.loc[part.index], check_exact=False, rtol=1e-9, atol=1e-12)
        assert part.index.max() <= btc.index[k - 1]


def test_i_resample_complete_bars_are_causal(bar_factory):
    df = bar_factory(n=N_BARS, seed=15)
    full = resample(df, "1h", incomplete="drop")
    for k in _cuts(N_BARS, 3):
        part = resample(df.iloc[:k], "1h", incomplete="drop")
        pd.testing.assert_frame_equal(part, full.iloc[: len(part)])


def test_i_checker_catches_a_lookahead_feature(bar_factory):
    """The harness must fail a centred window; otherwise the tests above prove nothing."""
    df = bar_factory(n=N_BARS, seed=16)

    def leaky(d: pd.DataFrame) -> pd.DataFrame:
        return d[["close"]].rolling(5, center=True).mean()

    with pytest.raises(AssertionError):
        _assert_causal(leaky, df, seed=4)


def test_make_feature_frame_params_and_skip(bar_factory):
    df = bar_factory(n=400, seed=17)
    out = features.make_feature_frame(df, params={"realized_vol": {"windows": (6,)}, "calendar": None})
    assert "rv_6" in out.columns and "rv_12" not in out.columns and "hour_sin" not in out.columns
