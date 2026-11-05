"""Train, publish and load the live LightGBM artifacts.

Layout (outside git, under the runner's models folder, see ``paths.models_dir``)::

    <models dir>/<config>/<SYMBOL>/<YYYY-MM>-<created UTC>/model.txt      LightGBM text model
    <models dir>/<config>/<SYMBOL>/<YYYY-MM>-<created UTC>/metadata.json  see below

``YYYY-MM`` is the month the model trades: it is trained on every closed bar from
``TRAIN_START`` up to the start of that month (the research fold for that month). A new
artifact is written into a hidden temporary folder and renamed into place only after it
trained, produced a threshold, and passed a reload check, so a failed or flat retrain never
replaces the artifact in use. A bar is decided by the newest folder for the month it OPENS in
(:func:`folder_for_month`), as the research decides month M with fold M's model.

A month whose training found no profitable threshold leaves a marker file
``.no-threshold-<YYYY-MM>.json`` next to the folders: that month keeps the newest earlier model
(the research stayed in cash; see docs/live_runner.md, Known limitations), and the hourly
retrain does not train it again (the result would be the same: training is deterministic).

``metadata.json`` records the config, model parameters, feature columns, threshold and the
quantile it came from, the training window, a hash of the 5m data, the git sha and the
lightgbm / scikit-learn / numpy / pandas versions. :func:`load_latest` refuses an artifact
whose library versions differ from the installed ones (a different version may predict
differently) or whose model file does not match its SHA256.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from profit_engine.intraday import config
from profit_engine.intraday.provenance import data_hash, git_sha
from profit_engine.intraday.resample import resample
from profit_engine.live.models import ArtifactError, sha256_file
from profit_engine.live.strategies import lgbm_core as core

MODEL_FILE = "model.txt"
META_FILE = "metadata.json"
WARMUP_BARS = {"lgbm_4h_h30_pt2_sl1": 3000}
OTHER = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT"}


class NoThreshold(RuntimeError):
    """Inner CV found no threshold that made money: research trades flat that month."""


class UnusableArtifact(ArtifactError):
    """An artifact that cannot be used; ``str()`` is the technical detail (for the logs) and
    ``reason`` says why in plain words (for the dashboard)."""

    def __init__(self, detail: str, reason: str) -> None:
        super().__init__(detail)
        self.reason = reason


VERSIONS_REASON = "it was trained with different library versions than the ones installed now"
CHANGED_REASON = "its model file changed or was damaged after it was trained"
INCOMPLETE_REASON = "some of its files are missing or unreadable"


def library_versions() -> dict[str, str]:
    return {"lightgbm": lgb.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__}  # fmt: skip


def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    """First instant of ``ts``'s UTC month (the cutoff for the model trading that month)."""
    ts = pd.Timestamp(ts).tz_convert("UTC")
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz="UTC")


def symbol_dir(models_dir: Path, cfg_name: str, symbol: str) -> Path:
    return Path(models_dir) / cfg_name / symbol


@dataclass
class Trained:
    booster: lgb.Booster
    threshold: float
    quantile: float
    columns: list[str]
    meta: dict


def _closed_before(df5: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return df5[(df5.index >= config.TRAIN_START) & (df5.index < cutoff)]


def train(frames_5m: dict[str, pd.DataFrame], symbol: str, cutoff: pd.Timestamp, cfg_name: str) -> Trained:
    """Train the model that trades the month starting at ``cutoff`` (exactly the research fold).

    ``frames_5m`` holds closed 5m bars of both symbols; only bars in [TRAIN_START, cutoff) are used."""
    cfg = core.CONFIGS[cfg_name]
    raw = {s: _closed_before(frames_5m[s], cutoff) for s in (symbol, OTHER[symbol])}
    bars = {s: resample(f, cfg["tf"], incomplete="drop") for s, f in raw.items()}
    xy = core.build_xy(bars[symbol], bars[OTHER[symbol]], cfg)
    train_mask = core.training_rows(xy, cutoff)
    model, thr, q = core.fit_fold(xy, train_mask, cutoff)
    if not np.isfinite(thr):
        raise NoThreshold(f"no threshold made money for {symbol} before {cutoff:%Y-%m-%d}")
    rows = xy["df"].index[train_mask]
    meta = {
        "config": cfg_name, "symbol": symbol, "other_symbol": OTHER[symbol], "timeframe": cfg["tf"],
        "trades_month": f"{cutoff:%Y-%m}", "threshold": thr, "threshold_quantile": q,
        "feature_columns": list(xy["x"].columns),
        "feature_params": {k: {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in v.items()}
                           for k, v in core.feature_params(cfg["tf"])[0].items()},
        "labels": {"kind": "triple_barrier", "h": cfg["h"], "pt": cfg["pt"], "sl": cfg["sl"],
                   "vol_window": 7 * core.BARS_PER_DAY[cfg["tf"]], "label_min_ret": core.LABEL_MIN_RET},
        "weights": "uniqueness", "model_params": {**core.MODEL_PARAMS, "objective": "binary", "seed": core.SEED,
                                                   "deterministic": True, "force_row_wise": True, "n_jobs": 4},
        "threshold_search": {"inner_splits": core.INNER_SPLITS, "quantiles": core.QUANTILES.tolist(),
                             "embargo": str(core.DEFAULT_EMBARGO), "cost_bps": config.BASE_COST_BPS},
        "train_window": {"start": config.TRAIN_START.isoformat(), "end_exclusive": cutoff.isoformat(),
                         "first_row": rows[0].isoformat(), "last_row": rows[-1].isoformat(),
                         "n_rows": int(train_mask.sum())},
        "data_hash": {s: data_hash(f) for s, f in raw.items()},
        "research_ref": "experiments/intraday/lgbm.py: oos_positions -> fit_fold (one monthly fold)",
    }  # fmt: skip
    return Trained(model.booster_, thr, q, list(xy["x"].columns), meta)


def publish(trained: Trained, models_dir: Path, now: pd.Timestamp | None = None) -> Path:
    """Write the artifact to a temporary folder, reload-check it, then rename it into place."""
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    meta = trained.meta
    version = f"{meta['trades_month']}-{now:%Y%m%dT%H%M%SZ}"
    parent = symbol_dir(models_dir, meta["config"], meta["symbol"])
    parent.mkdir(parents=True, exist_ok=True)
    tmp, final = parent / f".tmp-{version}", parent / version
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    trained.booster.save_model(str(tmp / MODEL_FILE))
    full = {
        **meta, "name": f"{meta['config']}-{meta['symbol']}", "version": version, "kind": "lgbm_booster",
        "warmup_bars": WARMUP_BARS[meta["config"]], "artifact": MODEL_FILE,
        "artifact_sha256": sha256_file(tmp / MODEL_FILE), "versions": library_versions(),
        "python": platform.python_version(), "git_sha": git_sha(), "created_at": now.isoformat(), "promoted": False,
    }  # fmt: skip
    (tmp / META_FILE).write_text(json.dumps(full, indent=2, sort_keys=True))
    reloaded = lgb.Booster(model_file=str(tmp / MODEL_FILE))
    if reloaded.feature_name() != trained.columns or reloaded.num_trees() != trained.booster.num_trees():
        shutil.rmtree(tmp, ignore_errors=True)
        raise ArtifactError("the saved model did not reload with the same features and trees")
    os.replace(tmp, final)
    return final


def versions(models_dir: Path, cfg_name: str, symbol: str) -> list[Path]:
    """Published artifact folders for (config, symbol), oldest first."""
    parent = symbol_dir(models_dir, cfg_name, symbol)
    if not parent.is_dir():
        return []
    return sorted(p for p in parent.iterdir() if p.is_dir() and not p.name.startswith(".") and (p / META_FILE).exists())


@dataclass
class Loaded:
    booster: lgb.Booster
    meta: dict
    folder: Path
    sha256: str


def check_versions(meta: dict) -> None:
    have, want = library_versions(), meta.get("versions") or {}
    diff = {k: (want.get(k), v) for k, v in have.items() if want.get(k) != v}
    if diff:
        said = ", ".join(f"{k} {w} (installed {h})" for k, (w, h) in diff.items())
        raise UnusableArtifact(f"it was trained with {said}", VERSIONS_REASON)


def load(folder: Path) -> Loaded:
    """Load one artifact; ArtifactError if it is incomplete, altered, or from other library versions."""
    folder = Path(folder)
    try:
        meta = json.loads((folder / META_FILE).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UnusableArtifact(f"{folder / META_FILE} is missing or unreadable", INCOMPLETE_REASON) from exc
    check_versions(meta)
    path = folder / str(meta.get("artifact", MODEL_FILE))
    if not path.exists():
        raise UnusableArtifact(f"{path} is missing", INCOMPLETE_REASON)
    digest = sha256_file(path)
    if digest != meta.get("artifact_sha256"):
        raise UnusableArtifact(f"{path} does not match the SHA256 recorded when it was trained", CHANGED_REASON)
    try:
        booster = lgb.Booster(model_file=str(path))
    except lgb.basic.LightGBMError as exc:
        raise UnusableArtifact(f"{path} could not be read by lightgbm: {exc}", CHANGED_REASON) from exc
    if booster.feature_name() != list(meta.get("feature_columns") or []):
        raise UnusableArtifact(f"{path} has different feature columns than its metadata", CHANGED_REASON)
    return Loaded(booster, meta, folder, digest)


def load_latest(models_dir: Path, cfg_name: str, symbol: str) -> Loaded:
    found = versions(models_dir, cfg_name, symbol)
    if not found:
        raise FileNotFoundError(symbol_dir(models_dir, cfg_name, symbol))
    return load(found[-1])


def trades_month(folder: Path) -> str:
    """``YYYY-MM`` of the month an artifact folder trades (its name starts with it)."""
    return Path(folder).name[:7]


def _marker(models_dir: Path, cfg_name: str, symbol: str, month: str) -> Path:
    return symbol_dir(models_dir, cfg_name, symbol) / f".no-threshold-{month}.json"


def mark_no_threshold(models_dir: Path, cfg_name: str, symbol: str, month: str, now: pd.Timestamp) -> None:
    """Record that ``month``'s training found no profitable threshold (it keeps the earlier model)."""
    path = _marker(models_dir, cfg_name, symbol, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"month": month, "found_at": pd.Timestamp(now).isoformat(), "versions": library_versions()})
    )


def has_no_threshold(models_dir: Path, cfg_name: str, symbol: str, month: str) -> bool:
    return _marker(models_dir, cfg_name, symbol, month).exists()


def folder_for_month(models_dir: Path, cfg_name: str, symbol: str, month: str) -> Path | None:
    """The artifact that decides bars opening in ``month``: the newest folder trained for it, else
    (only if that month found no threshold) the newest earlier one; None if there is none yet."""
    found = versions(models_dir, cfg_name, symbol)
    same = [p for p in found if trades_month(p) == month]
    if same:
        return same[-1]
    if has_no_threshold(models_dir, cfg_name, symbol, month):
        earlier = [p for p in found if trades_month(p) < month]
        return earlier[-1] if earlier else None
    return None
