"""Append-only trial registry.

Every ``log_trial`` call gets one JSON line in ``research/intraday/trials/<family>.jsonl``
and its own daily OOS net returns file
``research/intraday/trials/returns/<family>/<trial_id>.parquet`` (gitignored). ``trial_id``
is unique per call, so nothing is ever overwritten.

* The counted unit is a RUN. ``spec_id`` = hash(family, params, timeframe, symbol, seed,
  git sha, data hash) is what the caller declared; ``run_id`` = hash(spec_id, hash of the
  logged daily returns) is what actually ran. A code change the git sha cannot see (an
  untracked script) still changes the returns, so it is still a new trial. An identical
  re-run gives identical returns and dedupes.
* The DSR ``N`` is the number of distinct runs at base cost, plus every ``spec_id`` logged
  only at another cost. Re-scoring a run at stress cost (same spec, different net returns)
  therefore adds nothing. ``kind`` is a label only: benchmarks and baselines count like
  any other run, in ``N``, the variance and the PBO matrix.
* Sharpe ratios used by the registry (DSR variance, best baseline) are computed here from
  the saved returns. The caller's ``metrics["sharpe"]`` / ``["sharpe_daily"]`` are replaced
  by the harness values in the stored record.
* PBO is computed over the matrix of saved daily returns at base cost, one column per run.
* Returns must cover the walk-forward OOS window ``[EVAL_START, EVAL_END)``: start within
  ``COVERAGE_SLACK`` of ``EVAL_START`` and end within it of ``EVAL_END``, never at or after
  ``EVAL_END``. This holds inside ``open_holdout`` too: holdout results are recorded only by
  ``promotion.holdout_check`` in the holdout audit log, never here.
* ``kind="baseline"`` marks rule baselines; promotion requires beating the best of them.

Never edit or delete lines by hand: a trial you would rather forget is exactly the one the
deflation needs to know about.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from profit_engine.intraday import config
from profit_engine.intraday.holdout import assert_pre_holdout
from profit_engine.intraday.metrics import sharpe
from profit_engine.intraday.provenance import data_hash as content_hash
from profit_engine.intraday.provenance import git_sha

KINDS = ("model", "baseline", "benchmark")


def _dir(trials_dir: Path | None) -> Path:
    return Path(trials_dir or config.TRIALS_DIR)


def config_id(family: str, params: dict, timeframe: str, symbol: str) -> str:
    blob = json.dumps({"f": family, "p": params, "tf": timeframe, "s": symbol}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def spec_id(cid: str, seed: int | None, code_sha: str, data_hash: str) -> str:
    """What the caller declared: config + seed + code version (tracked files) + data version."""
    blob = json.dumps({"c": cid, "seed": seed, "git": code_sha, "data": data_hash}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_id(sid: str, returns_hash: str) -> str:
    """Identity of one evaluated run: its spec plus a hash of the returns it produced."""
    return hashlib.sha256(f"{sid}:{returns_hash}".encode()).hexdigest()[:16]


def _check_window(daily_returns: pd.Series, family: str) -> None:
    """The series must cover the walk-forward OOS window and nothing else.

    Enforced whether or not the holdout is open: only the guard's HoldoutViolation is lifted
    by ``open_holdout``; the EVAL_END bound here is not.
    """
    if len(daily_returns) == 0:
        raise ValueError(f"log_trial({family}): empty return series")
    idx = daily_returns.index
    assert_pre_holdout(idx, f"log_trial({family}) returns")  # TypeError unless a DatetimeIndex
    if idx.tz is None or not idx.is_monotonic_increasing or not idx.is_unique:
        raise ValueError(f"log_trial({family}): returns need a tz-aware, increasing, unique daily index")
    first, last, slack = idx[0], idx[-1], config.COVERAGE_SLACK
    if first < config.EVAL_START:
        raise ValueError(
            f"log_trial({family}): returns start {first}, before EVAL_START {config.EVAL_START}. "
            "Log only the walk-forward OOS series over [EVAL_START, EVAL_END)."
        )
    if last >= config.EVAL_END:
        raise ValueError(
            f"log_trial({family}): returns end {last}, at or after EVAL_END {config.EVAL_END}. "
            "The registry holds walk-forward OOS series only; holdout results are recorded by "
            "promotion.holdout_check in the holdout audit log."
        )
    if first > config.EVAL_START + slack or last < config.EVAL_END - slack:
        raise ValueError(
            f"log_trial({family}): returns {first.date()}..{last.date()} do not cover the walk-forward "
            f"window [{config.EVAL_START.date()}, {config.EVAL_END.date()}) within {slack}. A partial "
            "run would shrink the PBO matrix for every trial on this symbol."
        )


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
    """Append one trial record and save its daily OOS returns. Returns the trial id.

    ``metrics`` is stored as given except ``sharpe`` and ``sharpe_daily``, which the harness
    recomputes from ``daily_returns``.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if not family.replace("_", "").isalnum():
        raise ValueError("family must be [A-Za-z0-9_]")
    _check_window(daily_returns, family)
    cid, sha = config_id(family, params, timeframe, symbol), git_sha()
    sid, ret_hash = spec_id(cid, seed, sha, data_hash), content_hash(daily_returns)
    rid = run_id(sid, ret_hash)
    metrics = {
        **metrics,
        "sharpe": sharpe(daily_returns),
        "sharpe_daily": sharpe(daily_returns, annualize=False),
    }
    trial_id = f"{rid}_{int(cost_bps_per_side * 100):04d}_{uuid.uuid4().hex[:8]}"
    root = _dir(trials_dir)
    ret_path = root / "returns" / family / f"{trial_id}.parquet"
    ret_path.parent.mkdir(parents=True, exist_ok=True)
    daily_returns.rename("ret").to_frame().to_parquet(ret_path)
    record = {
        "trial_id": trial_id,
        "config_id": cid,
        "spec_id": sid,
        "run_id": rid,
        "returns_hash": ret_hash,
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
        "git_sha": sha,
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
            columns=[
                "trial_id",
                "config_id",
                "spec_id",
                "run_id",
                "family",
                "kind",
                "symbol",
                "cost_bps_per_side",
                "metrics",
            ]
        )
    df = pd.DataFrame(rows)
    for col in ("run_id", "spec_id"):
        if col not in df:
            df[col] = None
    df["run_id"] = df["run_id"].fillna(df["config_id"])  # records written before run_id existed
    df["spec_id"] = df["spec_id"].fillna(df["run_id"])  # records written before spec_id existed
    flat = pd.json_normalize(df["metrics"]).add_prefix("m_")
    return pd.concat([df, flat], axis=1)


def n_trials(trials_dir: Path | None = None) -> int:
    """The DSR ``N`` across ALL families and kinds.

    Distinct runs at base cost, plus each spec logged only at another cost. Scoring one run
    at several costs counts once.
    """
    df = load_trials(trials_dir)
    if df.empty:
        return 0
    base = df["cost_bps_per_side"] == config.BASE_COST_BPS
    base_specs = set(df.loc[base, "spec_id"])
    other_only = df.loc[~base & ~df["spec_id"].isin(base_specs), "spec_id"]
    return int(df.loc[base, "run_id"].nunique() + other_only.nunique())


def _base(trials_dir: Path | None, symbol: str | None) -> pd.DataFrame:
    df = load_trials(trials_dir)
    if df.empty:
        return df
    df = df[df["cost_bps_per_side"] == config.BASE_COST_BPS]
    if symbol is not None:
        df = df[df["symbol"] == symbol]
    return df.drop_duplicates("run_id", keep="last")  # one row per run (identical returns)


def _series(df: pd.DataFrame, trials_dir: Path | None) -> dict[str, pd.Series]:
    root = _dir(trials_dir)
    return {r.run_id: pd.read_parquet(root / r.returns_path)["ret"] for r in df.itertuples()}


def trial_sharpe_variance(trials_dir: Path | None = None) -> float:
    """Variance of per-period (daily) Sharpe across distinct base-cost runs, all families.

    Computed from each run's saved returns, so every counted run contributes.
    """
    runs = _series(_base(trials_dir, None), trials_dir)
    if len(runs) < 2:
        return 0.0
    return float(pd.Series([sharpe(r, annualize=False) for r in runs.values()]).var(ddof=1))


def returns_matrix(symbol: str | None = None, trials_dir: Path | None = None) -> pd.DataFrame:
    """Dates x runs matrix of daily base-cost returns on the dates common to all runs."""
    cols = _series(_base(trials_dir, symbol), trials_dir)
    return pd.DataFrame(cols).dropna() if cols else pd.DataFrame()


def best_baseline_sharpe(symbol: str, trials_dir: Path | None = None) -> float | None:
    """Highest annualised OOS Sharpe among ``kind="baseline"`` runs for ``symbol`` at base cost,
    computed from their saved returns."""
    df = _base(trials_dir, symbol)
    if df.empty:
        return None
    runs = _series(df[df["kind"] == "baseline"], trials_dir)
    return max(sharpe(r) for r in runs.values()) if runs else None
