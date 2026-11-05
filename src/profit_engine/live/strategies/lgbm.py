"""Live LightGBM 4h book: the newest trained artifact, decided on closed bars only.

The research rule (``experiments/intraday/lgbm.py``, config ``lgbm_4h_h30_pt2_sl1``) is long
for the next bar when the fold's model gives P(triple-barrier win) > the fold's threshold,
and flat otherwise, including on a bar where any feature is missing. Features use the
book's coin AND the other coin (cross-asset returns) on the bars both have.

Warm-up. ``emadist_180`` is an EWM with infinite memory. On the cached 4h bars (150 random
bars per symbol, 2026-11-05) the worst relative feature difference between a window and the
full history was 1.7e-07 at 2000 bars, 1.3e-09 at 2500 and 5e-10 at 3000 (there the worst
column is a rolling-sum rounding difference, not the EWM). ``warmup_bars`` is 3000 (500 days).

If no usable artifact exists (never trained, altered, or trained under other library
versions), :meth:`LgbmStrategy.decide` raises :class:`StrategyUnavailable` and the book keeps
its position until ``signal-desk-live train lgbm`` or ``retrain`` produces one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from profit_engine.live import paths
from profit_engine.live.models import ArtifactError
from profit_engine.live.strategies import lgbm_artifacts as art
from profit_engine.live.strategies import lgbm_core as core
from profit_engine.live.strategy import Decision, StrategyMeta, StrategyUnavailable

FAMILY = "LightGBM 4h"
STATUS = "not promoted — comparison model (backtest trailed buy-and-hold)"
DEFAULT_CONFIG = "lgbm_4h_h30_pt2_sl1"


def _coin(symbol: str) -> str:
    return symbol.removesuffix("USDT")


class LgbmStrategy:
    """Long when the latest model's probability for the newest closed bar beats its threshold."""

    name = "lgbm"

    def __init__(self, symbol: str, timeframe: str = "4h", config: str = DEFAULT_CONFIG, models_dir=None) -> None:
        if config not in core.CONFIGS:
            raise ValueError(f"unknown lgbm config {config!r}; known: {sorted(core.CONFIGS)}")
        if timeframe != core.CONFIGS[config]["tf"]:
            raise ValueError(f"{config} trades {core.CONFIGS[config]['tf']} bars, not {timeframe}")
        if symbol not in art.OTHER:
            raise ValueError(f"lgbm needs BTCUSDT or ETHUSDT, got {symbol!r}")
        self.symbol, self.timeframe, self.config = symbol, timeframe, config
        self.other_symbol = art.OTHER[symbol]
        self.warmup_bars = art.WARMUP_BARS[config]
        self.models_dir = Path(models_dir or paths.models_dir())
        self.loaded, self.problem = None, ""
        try:
            self.loaded = art.load_latest(self.models_dir, config, symbol)
        except FileNotFoundError:
            self.problem = (
                f"No trained {_coin(symbol)} LightGBM model yet, so the book stays as it is. "
                f"Run: signal-desk-live train lgbm --symbol {symbol}"
            )
        except (ArtifactError, ValueError, OSError) as exc:
            self.problem = (
                f"The {_coin(symbol)} LightGBM model can't be used ({exc}), so the book stays as it is. "
                "Retrain it with: signal-desk-live retrain"
            )
        self.meta = self._meta()

    def _meta(self) -> StrategyMeta:
        base = {"config": self.config, "warmup_bars": self.warmup_bars}
        if self.loaded is None:
            return StrategyMeta(version="no model yet", params=base, status=STATUS, family=FAMILY, source="lgbm")
        m = self.loaded.meta
        params = {**base, "threshold": m["threshold"], "trades_month": m["trades_month"],
                  "train_window": m["train_window"], "feature_columns": m["feature_columns"]}  # fmt: skip
        return StrategyMeta(
            version=m["version"], params=params, artifact_hash=self.loaded.sha256,
            promoted=False, source=f"model:{self.loaded.folder.name}", status=STATUS, family=FAMILY,
        )  # fmt: skip

    def probability(self, bars: pd.DataFrame, other: pd.DataFrame | None) -> float | None:
        """P(win) for the newest bar, or None when a feature is missing (research: flat)."""
        if self.loaded is None:
            raise StrategyUnavailable(self.problem)
        if other is None or other.empty:
            return None
        _, x = core.feature_frame(bars, other, self.timeframe)
        if x.empty or x.index[-1] != bars.index[-1]:
            return None
        row = x.iloc[[-1]][self.loaded.meta["feature_columns"]]
        if row.isna().any(axis=None):
            return None
        return float(self.loaded.booster.predict(row)[0])

    def decide(self, bars: pd.DataFrame, other: pd.DataFrame | None = None) -> Decision:
        p = self.probability(bars, other)
        coin, thr = _coin(self.symbol), float(self.loaded.meta["threshold"])
        month = self.loaded.meta["trades_month"]
        if p is None:
            text = f"Some model inputs for this bar are missing (no matching {_coin(self.other_symbol)} bar or too "
            text += "little history), so the book stays in cash, as the backtest did."
            return Decision(0.0, text, {"threshold": thr, "model": self.meta.version})
        up = bool(np.float64(p) > thr)
        text = (
            f"The model (trained for {month}) puts the chance that {coin} rises enough over the next "
            f"few days at {p:.0%}; it needs more than {thr:.0%}, so the book "
            f"{'holds ' + coin if up else 'stays in cash'}."
        )
        return Decision(1.0 if up else 0.0, text, {"p_up": p, "threshold": thr, "model": self.meta.version})
