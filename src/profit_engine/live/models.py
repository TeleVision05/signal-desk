"""Versioned model artifacts: ``<models dir>/<name>/metadata.json`` (+ optional joblib file).

``metadata.json`` fields:

| field | meaning |
|---|---|
| ``name``, ``version`` | shown on the dashboard and stored with every decision |
| ``symbol``, ``timeframe``, ``warmup_bars`` | what the runner feeds ``decide`` |
| ``other_symbol`` | optional second symbol (same timeframe) for cross-asset features |
| ``kind`` | ``"proba_threshold"`` (joblib classifier, long when P(up) > ``threshold``) or ``"python"`` |
| ``feature_params``, ``feature_columns`` | ``intraday.features.make_feature_frame`` params and column order |
| ``artifact``, ``artifact_sha256`` | joblib file name and its SHA256; a mismatch refuses to load |
| ``entry`` | ``"package.module:factory"`` for ``kind="python"``; ``factory(metadata)`` returns a Strategy |
| ``promoted``, ``promotion.verdict`` | the book is labelled "promoted" only if both say so (``true`` / ``"PASS"``) |

Loading a joblib file executes pickled code: only put artifacts you built yourself in the
models folder.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import joblib
import pandas as pd

from profit_engine.intraday.features import make_feature_frame
from profit_engine.live.strategy import Decision, Strategy, StrategyMeta

REQUIRED = ("name", "version", "symbol", "timeframe", "warmup_bars", "kind")


class ArtifactError(ValueError):
    """The model folder is missing, incomplete, or its file does not match its hash."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_metadata(folder: Path) -> dict:
    path = Path(folder) / "metadata.json"
    if not path.exists():
        raise ArtifactError(f"{path} not found")
    meta = json.loads(path.read_text())
    missing = [k for k in REQUIRED if k not in meta]
    if missing:
        raise ArtifactError(f"{path} lacks {missing}")
    return meta


def artifact_hash(folder: Path, meta: dict) -> str:
    """SHA256 of the joblib file (checked against ``artifact_sha256``) or of metadata.json."""
    name = meta.get("artifact")
    if not name:
        return sha256_file(Path(folder) / "metadata.json")
    path = Path(folder) / name
    if not path.exists():
        raise ArtifactError(f"{path} not found")
    actual = sha256_file(path)
    expected = (meta.get("artifact_sha256") or "").lower()
    if not expected:
        raise ArtifactError(f"{folder}/metadata.json must record artifact_sha256 for {name}")
    if actual != expected:
        raise ArtifactError(f"{path} sha256 {actual} does not match metadata {expected}")
    return actual


def is_promoted(meta: dict) -> bool:
    return meta.get("promoted") is True and (meta.get("promotion") or {}).get("verdict") == "PASS"


def model_meta(folder: Path, meta: dict) -> StrategyMeta:
    return StrategyMeta(
        version=str(meta["version"]),
        params={k: meta[k] for k in ("threshold", "feature_params", "feature_columns") if k in meta},
        artifact_hash=artifact_hash(folder, meta),
        promoted=is_promoted(meta),
        source=f"model:{Path(folder).name}",
    )


class ProbaThresholdModel:
    """A fitted classifier with ``predict_proba``: long when P(up) of the last bar > threshold."""

    def __init__(self, folder: Path, meta: dict) -> None:
        self.meta = model_meta(folder, meta)
        self.name = str(meta["name"])
        self.symbol, self.timeframe = str(meta["symbol"]), str(meta["timeframe"])
        self.warmup_bars = int(meta["warmup_bars"])
        self.other_symbol = meta.get("other_symbol")
        self.threshold = float(meta.get("threshold", 0.5))
        self.feature_params = meta.get("feature_params") or {}
        self.columns = list(meta.get("feature_columns") or [])
        self.model = joblib.load(Path(folder) / meta["artifact"])

    def decide(self, bars: pd.DataFrame, other: pd.DataFrame | None = None) -> Decision:
        x = make_feature_frame(bars, other, params=self.feature_params)
        if x.empty or x.index[-1] != bars.index[-1]:
            return Decision(0.0, "Not enough matching data for the model on this bar, so the book stays in cash.")
        row = x.iloc[[-1]][self.columns] if self.columns else x.iloc[[-1]]
        if row.isna().any(axis=None) and not getattr(self.model, "accepts_nan", False):
            return Decision(0.0, "Some model inputs are not available yet, so the book stays in cash.")
        p = float(self.model.predict_proba(row)[0, 1])
        up = p > self.threshold
        text = (
            f"The model puts the chance of a rise at {p:.0%} (it needs more than {self.threshold:.0%}), "
            f"so the book {'holds the coin' if up else 'stays in cash'}."
        )
        return Decision(1.0 if up else 0.0, text, {"p_up": p, "threshold": self.threshold})


def load_model(folder: Path) -> Strategy:
    """Build the strategy described by ``folder/metadata.json``."""
    folder = Path(folder)
    meta = read_metadata(folder)
    if meta["kind"] == "proba_threshold":
        if not meta.get("artifact"):
            raise ArtifactError(f"{folder}: kind proba_threshold needs an artifact file")
        return ProbaThresholdModel(folder, meta)
    if meta["kind"] == "python":
        module, _, attr = str(meta.get("entry", "")).partition(":")
        if not module or not attr:
            raise ArtifactError(f"{folder}: kind python needs entry 'package.module:factory'")
        strategy = getattr(importlib.import_module(module), attr)(meta)
        strategy.meta = model_meta(folder, meta)
        return strategy
    raise ArtifactError(f"{folder}: unknown kind {meta['kind']!r}")
