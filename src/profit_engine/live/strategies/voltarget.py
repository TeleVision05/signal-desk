"""Vol-targeted multi-lookback trend, ported from ``experiments/intraday/voltarget_trend.py``.

The research rule (family ``voltarget_trend``) computes a raw exposure per bar,

    raw = mean over lookbacks of clip(log return over n bars / (EWMA vol * sqrt(n)), 0, 1)
          * min(1, sigma_target / annualised EWMA vol)

and then moves the target to ``round(raw / 0.05) * 0.05`` only when ``|raw - target| > band``
(a raw that rounds to 0 exits at once). That last step is path dependent, so the live
strategy never re-runs it over its window: it takes ``prev_target`` (the book's own last
target, persisted by the runner) and applies ONE step of the band to the newest bar.
Replayed bar by bar from the first bar with ``prev_target`` fed back, it reproduces the
research target exactly (``tests/live/test_live_strategy_parity.py``).

Warm-up. The EWMA vol has infinite memory, so a window of ``warmup_bars`` bars gives a vol
that differs from the full-history one by the weight of the bars left out. Measured on the
cached 2018..2026-06 1h bars (300 random bars per symbol, 2026-11-05), the worst relative
difference of the raw exposure was 1.5e-09 at 2400 bars, 6e-10 at 2600, 3e-11 at 3000 and
4e-14 at 3500 bars. :data:`WARMUP_BARS` is 3600 (150 days of 1h bars). The parity test
re-measures it on every bar of 2025-01..2026-06.

Only the configuration the user chose for paper trading is listed in :data:`CONFIGS`; its
parameters are copied from ``GRID_BY_NAME`` in the experiment (the parity test checks they
still match).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from profit_engine.live.strategy import Decision, StrategyMeta

FAMILY = "Trend (vol-targeted)"
STATUS = "not promoted — best research candidate (forward test)"
BARS_PER_DAY = {"1h": 24, "4h": 6}
STEP = 0.05
CONFIGS: dict[str, dict] = {
    "vt_1h_slow4_z_s30": {
        "tf": "1h",
        "params": {
            "rule": "voltarget_trend",
            "lookbacks_days": [7, 14, 30, 60],
            "signal": "z",
            "sigma_target": 0.30,
            "band": 0.2,
            "vol_halflife_days": 3,
            "regime_ema_days": None,
            "step": STEP,
        },
    },
}
WARMUP_BARS = {"vt_1h_slow4_z_s30": 3600}


def ewma_sigma(df: pd.DataFrame, hl_bars: float) -> pd.Series:
    """Mirrors ``voltarget_trend.ewma_sigma``: per-bar EWMA vol of 1-bar log close returns."""
    r1 = np.log(df["close"]).diff()
    return np.sqrt((r1**2).ewm(halflife=hl_bars, min_periods=int(2 * hl_bars)).mean())


def _signal(r: float, sig: float, n: int, signal: str) -> float:
    """One lookback's [0, 1] signal, element for element as ``voltarget_trend.trend_score``."""
    if signal == "vote":
        return float("nan") if np.isnan(r) else float(r > 0)
    if signal == "z":
        den = (sig if sig > 0 else np.nan) * np.sqrt(n)
        return float(np.clip(r / den, 0.0, 1.0))
    raise ValueError(f"unknown signal {signal}")


def latest_exposure(df: pd.DataFrame, tf: str, p: dict) -> dict[str, float]:
    """``voltarget_trend.raw_exposure`` evaluated at the LAST bar of ``df`` only.

    The EWMA vol is the research pandas call on the whole window; everything after it is the
    research's element-wise arithmetic on that one row (same IEEE operations in the same
    order: the lookback mean adds left to right and divides by the count, NaN if any is NaN).
    """
    bpd = BARS_PER_DAY[tf]
    sig = float(ewma_sigma(df, p["vol_halflife_days"] * bpd).iloc[-1])
    lc = np.log(df["close"].to_numpy(dtype="float64"))
    parts = []
    for n in (d * bpd for d in p["lookbacks_days"]):
        r = lc[-1] - lc[-1 - n] if len(lc) > n else np.nan
        parts.append(_signal(r, sig, n, p["signal"]))
    total = 0.0
    for x in parts:
        total += x
    score = total / len(parts)
    vol = (sig if sig > 0 else np.nan) * np.sqrt(bpd * 365)
    if p["sigma_target"] is None:
        scale = 1.0 if np.isfinite(sig) else np.nan
    else:
        scale = min(p["sigma_target"] / vol, 1.0) if np.isfinite(vol) else np.nan
    raw = score * scale
    if p["regime_ema_days"]:
        n = p["regime_ema_days"] * bpd
        ema = float(df["close"].ewm(span=n, adjust=False, min_periods=n).mean().iloc[-1])
        raw = raw * (float(df["close"].iloc[-1] > ema) if np.isfinite(ema) else np.nan)
    return {"raw": float(raw), "score": float(score), "scale": float(scale), "vol": float(vol)}


def hysteresis_step(x: float, cur: float, band: float, step: float = STEP) -> float:
    """One bar of ``voltarget_trend.hysteresis``: the next target given the current one.

    Same expressions as the research loop body, so the float result is identical."""
    if np.isfinite(x):
        q = min(1.0, round(x / step) * step)
        if q <= 0.0:
            cur = 0.0
        elif abs(x - cur) > band:
            cur = q
    return cur


def _explain(coin: str, last: dict, prev: float, target: float, band: float) -> str:
    if not np.isfinite(last["raw"]):
        return f"Not enough {coin} price history yet to measure the trend, so the book keeps {prev:.0%} in {coin}."
    trend = f"The {coin} trend score is {last['score']:.0%} and its recent volatility is {last['vol']:.0%} a year"
    if min(1.0, round(last["raw"] / STEP) * STEP) <= 0.0:
        return f"{trend}, which points to holding no {coin}, so the book {'sells to' if prev > 0 else 'stays in'} cash."
    want = f"which points to holding about {last['raw']:.0%} of the book in {coin}"
    if target == prev:
        return (
            f"{trend}, {want}. That is within {band:.0%} of the current {prev:.0%}, "
            "so the book keeps its position to avoid small trades."
        )
    return f"{trend}, {want}, so the book moves from {prev:.0%} to {target:.0%} in {coin}."


class VolTargetTrend:
    """Live port of the ``voltarget_trend`` rule for one coin (long/flat spot, fractional)."""

    name = "voltarget_trend"
    stateful = True

    def __init__(self, symbol: str, timeframe: str = "1h", config: str = "vt_1h_slow4_z_s30") -> None:
        if config not in CONFIGS:
            raise ValueError(f"unknown voltarget_trend config {config!r}; known: {sorted(CONFIGS)}")
        cfg = CONFIGS[config]
        if timeframe != cfg["tf"]:
            raise ValueError(f"{config} trades {cfg['tf']} bars, not {timeframe}")
        self.symbol, self.timeframe, self.config = symbol, timeframe, config
        self.params = dict(cfg["params"])
        self.warmup_bars = WARMUP_BARS[config]
        self.other_symbol = None
        self.meta = StrategyMeta(
            version=f"{config}/1",
            params={"config": config, **self.params, "warmup_bars": self.warmup_bars},
            status=STATUS,
            family=FAMILY,
        )

    def decide(self, bars: pd.DataFrame, other: pd.DataFrame | None = None, prev_target: float = 0.0) -> Decision:
        last = latest_exposure(bars, self.timeframe, self.params)
        x = np.float64(last["raw"])
        target = float(hysteresis_step(x, float(prev_target), self.params["band"], self.params["step"]))
        text = _explain(self.symbol.removesuffix("USDT"), last, float(prev_target), target, self.params["band"])
        return Decision(target, text, {**last, "prev_target": float(prev_target)})
