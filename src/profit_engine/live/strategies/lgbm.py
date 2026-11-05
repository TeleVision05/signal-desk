"""Live LightGBM 4h book: each bar decided by its month's trained artifact, on closed bars only.

The research rule (``experiments/intraday/lgbm.py``, config ``lgbm_4h_h30_pt2_sl1``) is long
for the next bar when the fold's model gives P(triple-barrier win) > the fold's threshold,
and flat otherwise, including on a bar where any feature is missing. Features use the
book's coin AND the other coin (cross-asset returns) on the bars both have.

Which model. The research decides every bar OPENING in month M (UTC) with fold M's model
(``oos_positions``), so the live book does too: :func:`lgbm_artifacts.folder_for_month`. A bar
of a month whose model is not trained yet is NOT decided with last month's model: ``decide``
raises :class:`StrategyUnavailable` (the book keeps its position, the bar stays unclaimed and
is decided as soon as ``retrain`` publishes the model; the retrain job runs every hour). A
month whose training found no profitable threshold keeps the previous model (documented in
docs/live_runner.md, Known limitations; the research stayed in cash).

A bar the other coin lacks. The research builds features on the bars both coins have
(``features.align``), so such a bar is not in its target or backtest at all and the research
HOLDS its position across it. The book therefore keeps its own last target (``stateful``).

Warm-up. ``emadist_180`` is an EWM with infinite memory. On the cached 4h bars (150 random
bars per symbol, 2026-11-05) the worst relative feature difference between a window and the
full history was 1.7e-07 at 2000 bars, 1.3e-09 at 2500 and 5e-10 at 3000 (there the worst
column is a rolling-sum rounding difference, not the EWM). ``warmup_bars`` is 3000 (500 days).

If no usable artifact exists (never trained, altered, or trained under other library
versions), the book keeps its position: ``problem`` says why in plain words (the runner
raises it as :class:`StrategyUnavailable`) and ``problem_detail`` keeps the technical detail
for the log, until ``signal-desk-live train lgbm`` or ``retrain`` produces a usable one.
"""

from __future__ import annotations

import logging
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
log = logging.getLogger("signal_desk.lgbm")


def _coin(symbol: str) -> str:
    return symbol.removesuffix("USDT")


def _month_name(month: str) -> str:
    return f"{pd.Timestamp(month + '-01'):%B %Y}"


class LgbmStrategy:
    """Long when its month's model gives the newest closed bar a probability above its threshold."""

    name = "lgbm"
    stateful = True  # a bar the other coin lacks keeps the book's own last target

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
        self.loaded, self.problem, self.problem_detail = None, "", ""
        try:
            self.loaded = art.load_latest(self.models_dir, config, symbol)
        except FileNotFoundError:
            self.problem = (
                f"No trained {_coin(symbol)} LightGBM model yet, so the book stays as it is. "
                f"Run: signal-desk-live train lgbm --symbol {symbol}"
            )
        except (ArtifactError, ValueError, OSError) as exc:
            self.problem, self.problem_detail = self._unusable(exc), str(exc)
        self.meta = self._meta()

    def _unusable(self, exc: Exception) -> str:
        reason = getattr(exc, "reason", art.INCOMPLETE_REASON)
        return (
            f"The {_coin(self.symbol)} LightGBM model can't be used: {reason}. The book keeps its position "
            "until it is retrained. Retrain it with: signal-desk-live retrain"
        )

    def _meta(self) -> StrategyMeta:
        base = {"config": self.config, "warmup_bars": self.warmup_bars}
        if self.loaded is None:
            return StrategyMeta(version="none", params=base, status=STATUS, family=FAMILY, source="lgbm")
        m = self.loaded.meta
        params = {**base, "threshold": m["threshold"], "trades_month": m["trades_month"],
                  "train_window": m["train_window"], "feature_columns": m["feature_columns"]}  # fmt: skip
        return StrategyMeta(
            version=m["version"], params=params, artifact_hash=self.loaded.sha256,
            promoted=False, source=f"model:{self.loaded.folder.name}", status=STATUS, family=FAMILY,
        )  # fmt: skip

    def model_for(self, bar_open: pd.Timestamp) -> art.Loaded:
        """The artifact for the month ``bar_open`` falls in (UTC); StrategyUnavailable if none is usable."""
        if self.loaded is None:
            raise StrategyUnavailable(self.problem)
        month = f"{art.month_start(bar_open):%Y-%m}"
        folder = art.folder_for_month(self.models_dir, self.config, self.symbol, month)
        if folder is None:
            raise StrategyUnavailable(
                f"The {_coin(self.symbol)} LightGBM model for {_month_name(month)} is not trained yet, so the book "
                "keeps its position for now. The retrain job trains it within the hour after the month starts "
                "(UTC); to do it now, run: signal-desk-live retrain"
            )
        if folder != self.loaded.folder:
            try:
                self.loaded = art.load(folder)
            except (ArtifactError, ValueError, OSError) as exc:
                log.warning("lgbm artifact unusable", extra={"folder": str(folder), "error": str(exc)})
                raise StrategyUnavailable(self._unusable(exc)) from exc
            self.meta = self._meta()
        return self.loaded

    def probability(self, model: art.Loaded, bars: pd.DataFrame, other: pd.DataFrame) -> float | None:
        """P(win) for the newest bar, or None when a feature is missing (research: flat)."""
        _, x = core.feature_frame(bars, other, self.timeframe)
        row = x.iloc[[-1]][model.meta["feature_columns"]]
        if row.isna().any(axis=None):
            return None
        return float(model.booster.predict(row)[0])

    def decide(self, bars: pd.DataFrame, other: pd.DataFrame | None = None, prev_target: float = 0.0) -> Decision:
        model = self.model_for(bars.index[-1])
        coin, m = _coin(self.symbol), model.meta
        thr, month = float(m["threshold"]), m["trades_month"]
        details = {"threshold": thr, "model": m["version"]}
        if other is None or bars.index[-1] not in other.index:  # research: no such bar, position held
            text = (
                f"There is no closed {_coin(self.other_symbol)} bar for this bar, so the model cannot score it and "
                f"the book keeps its position ({prev_target:.0%} in {coin}), as the backtest did."
            )
            return Decision(float(prev_target), text, {**details, "no_matching_bar": True})
        p = self.probability(model, bars, other)
        if p is None:
            text = "Some model inputs for this bar are missing (too little history), so the book stays in cash, "
            return Decision(0.0, text + "as the backtest did.", details)
        up = bool(np.float64(p) > thr)
        bar_month = f"{art.month_start(bars.index[-1]):%Y-%m}"
        kept = "" if bar_month == month else (
            f" Training for {_month_name(bar_month)} found no setting that made money, so it keeps this model."
        )  # fmt: skip
        text = (
            f"The model (trained for {month}) puts the chance that {coin} rises enough over the next "
            f"few days at {p:.0%}; it needs more than {thr:.0%}, so the book "
            f"{'holds ' + coin if up else 'stays in cash'}.{kept}"
        )
        return Decision(1.0 if up else 0.0, text, {**details, "p_up": p})
