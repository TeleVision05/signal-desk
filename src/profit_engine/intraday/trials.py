"""Append-only trial registry.

Every configuration ever evaluated gets one JSON line in
``research/intraday/trials/<family>.jsonl`` and its daily OOS net returns in
``research/intraday/trials/returns/<family>/<trial_id>.parquet`` (gitignored).

* The DSR trial count is the number of DISTINCT configurations (family, params, timeframe,
  symbol) across ALL families: re-running a config, or scoring it at stress cost, does not
  add a trial; trying a new threshold does.
* PBO is computed over the matrix of saved daily returns at base cost.
* ``kind="baseline"`` marks rule baselines; promotion requires beating the best of them.

Never edit or delete lines by hand: a trial you would rather forget is exactly the one the
deflation needs to know about.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from profit_engine.intraday import config
from profit_engine.intraday.holdout import assert_pre_holdout
from profit_engine.intraday.provenance import git_sha

KINDS = ("model", "baseline", "benchmark")


def _dir(trials_dir: Path | None) -> Path:
    return Path(trials_dir or config.TRIALS_DIR)


def config_id(family: str, params: dict, timeframe: str, symbol: str) -> str:
    blob = json.dumps({"f": family, "p": params, "tf": timeframe, "s": symbol}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _jsonable(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def log_trial(
    family: str,
    params: dict,
    timeframe: str,
    symbol: str,
    cost_bps_per_side: float,
    metrics: dict,
    daily_returns: pd.Series,
    data_hash: str,
    seed: int | None,
    kind: str = "model",
    notes: str = "",
    trials_dir: Path | None = None,
) -> str:
    """Append one trial record and save its daily OOS returns. Returns the trial id."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if not family.replace("_", "").isalnum():
        raise ValueError("family must be [A-Za-z0-9_]")
    assert_pre_holdout(daily_returns.index, f"log_trial({family}) returns")
    cid = config_id(family, params, timeframe, symbol)
    trial_id = f"{cid}_{int(cost_bps_per_side * 100):04d}"
    root = _dir(trials_dir)
    ret_path = root / "returns" / family / f"{trial_id}.parquet"
    ret_path.parent.mkdir(parents=True, exist_ok=True)
    daily_returns.rename("ret").to_frame().to_parquet(ret_path)
    record = {
        "trial_id": trial_id,
        "config_id": cid,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "kind": kind,
        "params": params,
        "timeframe": timeframe,
        "symbol": symbol,
        "cost_bps_per_side": cost_bps_per_side,
        "metrics": metrics,
        "oos_start": str(daily_returns.index[0].date()),
        "oos_end": str(daily_returns.index[-1].date()),
        "data_hash": data_hash,
        "git_sha": git_sha(),
        "seed": seed,
        "returns_path": str(ret_path.relative_to(root)),
        "notes": notes,
    }
    with (root / f"{family}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(record), sort_keys=True, default=str) + "\n")
    return trial_id


def load_trials(trials_dir: Path | None = None) -> pd.DataFrame:
    """Every record from every family, one row per log_trial call (latest last)."""
    rows = []
    for path in sorted(_dir(trials_dir).glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return pd.DataFrame(
            columns=["trial_id", "config_id", "family", "kind", "symbol", "cost_bps_per_side", "metrics"]
        )
    df = pd.DataFrame(rows)
    flat = pd.json_normalize(df["metrics"]).add_prefix("m_")
    return pd.concat([df, flat], axis=1)


def n_trials(trials_dir: Path | None = None) -> int:
    """Distinct configurations across ALL families (the DSR ``N``); benchmarks excluded."""
    df = load_trials(trials_dir)
    return int(df.loc[df["kind"] != "benchmark", "config_id"].nunique())


def _base(trials_dir: Path | None, symbol: str | None) -> pd.DataFrame:
    df = load_trials(trials_dir)
    if df.empty:
        return df
    df = df[(df["cost_bps_per_side"] == config.BASE_COST_BPS) & (df["kind"] != "benchmark")]
    if symbol is not None:
        df = df[df["symbol"] == symbol]
    return df.drop_duplicates("trial_id", keep="last")


def trial_sharpe_variance(trials_dir: Path | None = None) -> float:
    """Variance of per-period (daily) Sharpe across distinct base-cost trials, all families."""
    df = _base(trials_dir, None)
    if len(df) < 2:
        return 0.0
    return float(df["m_sharpe_daily"].astype(float).var(ddof=1))


def returns_matrix(symbol: str | None = None, trials_dir: Path | None = None) -> pd.DataFrame:
    """Dates x trials matrix of daily base-cost returns on the dates common to all trials."""
    df = _base(trials_dir, symbol)
    root = _dir(trials_dir)
    cols = {r.trial_id: pd.read_parquet(root / r.returns_path)["ret"] for r in df.itertuples()}
    return pd.DataFrame(cols).dropna() if cols else pd.DataFrame()


def best_baseline_sharpe(symbol: str, trials_dir: Path | None = None) -> float | None:
    """Highest annualised OOS Sharpe among ``kind="baseline"`` trials for ``symbol`` at base cost."""
    df = _base(trials_dir, symbol)
    if df.empty:
        return None
    df = df[df["kind"] == "baseline"]
    return float(df["m_sharpe"].astype(float).max()) if len(df) else None
