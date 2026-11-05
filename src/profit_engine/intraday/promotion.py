"""Promotion gate: a candidate either clears every criterion on BOTH symbols or it FAILs.

Inputs are walk-forward OOS backtests over ``[EVAL_START, EVAL_END)`` at base (5 bps/side)
and stress (10 bps/side) cost, plus the bar frames they ran on (for the buy-and-hold
benchmarks). The gate checks its own inputs: the base backtest must start inside
``[EVAL_START, EVAL_START + 7D]`` (no pre-2020 P&L), the stress run must hold the same
positions on the same bars, and ``bars`` must be the base backtest's bars (benchmarks are
always computed on the base window, so a wider frame cannot flatter the strategy).
The candidate must already be in the trial registry: the DSR uses the total distinct-run
count across ALL families and PBO uses the registry's return matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from profit_engine.intraday import config, trials
from profit_engine.intraday.backtest import BacktestResult, buy_and_hold, vol_matched_buy_and_hold
from profit_engine.intraday.holdout import HoldoutViolation, assert_pre_holdout, holdout_is_open
from profit_engine.intraday.metrics import sharpe, summarize
from profit_engine.intraday.overfitting import dsr_from_daily, pbo_cscv, permutation_test

COVERAGE_SLACK = pd.Timedelta("7D")


@dataclass
class Candidate:
    family: str
    params: dict
    timeframe: str
    base: dict[str, BacktestResult]  # symbol -> OOS backtest at BASE_COST_BPS
    stress: dict[str, BacktestResult]  # symbol -> same positions at STRESS_COST_BPS
    bars: dict[str, pd.DataFrame]  # symbol -> the EXACT bar frame base/stress ran on (checked)


def _row(symbol: str, name: str, value: float, rule: str, passed: bool) -> dict:
    return {"symbol": symbol, "criterion": name, "value": value, "threshold": rule, "pass": bool(passed)}


def _registry_rows(c: Candidate, sym: str, base: BacktestResult, m: dict, trials_dir: Path | None) -> list[dict]:
    cid = trials.config_id(c.family, c.params, c.timeframe, sym)
    logged = cid in set(trials.load_trials(trials_dir).get("config_id", []))
    n, var = trials.n_trials(trials_dir), trials.trial_sharpe_variance(trials_dir)
    dsr = dsr_from_daily(base.daily, n, var)
    mat = trials.returns_matrix(sym, trials_dir)
    pbo = pbo_cscv(mat)["pbo"] if mat.shape[1] >= 2 and len(mat) >= 32 else float("nan")
    best = trials.best_baseline_sharpe(sym, trials_dir)
    return [
        _row(sym, "logged in trial registry", float(logged), "== 1", logged),
        _row(sym, f"DSR (N={n} trials)", dsr, ">= 0.95", dsr >= 0.95),
        _row(sym, f"PBO ({mat.shape[1]} trials)", pbo, "<= 0.20", pbo <= 0.20),
        _row(
            sym,
            "Sharpe vs best rule baseline",
            m["sharpe"] - (best if best is not None else np.inf),
            "> 0",
            best is not None and m["sharpe"] > best,
        ),
    ]


def _consistency_rows(sym: str, base: BacktestResult, stress: BacktestResult, bars: pd.DataFrame) -> list[dict]:
    """The stress run and the benchmark bars must belong to the base backtest, bar for bar."""
    first, last = base.bars.index[0], base.bars.index[-1]
    covers = config.EVAL_START <= first <= config.EVAL_START + COVERAGE_SLACK and last >= (
        config.EVAL_END - COVERAGE_SLACK
    )
    same_stress = stress.bars.index.equals(base.bars.index) and np.array_equal(
        stress.bars["position"].to_numpy(), base.bars["position"].to_numpy()
    )
    same_bars = bars.index.equals(base.bars.index)
    return [
        _row(sym, "covers 2020-01..2026-06", float(covers), f"{first}..{last}", covers),
        _row(
            sym,
            "costs are 5 / 10 bps per side",
            base.cost_bps_per_side,
            "5 & 10",
            base.cost_bps_per_side == config.BASE_COST_BPS and stress.cost_bps_per_side == config.STRESS_COST_BPS,
        ),
        _row(sym, "stress run holds base positions", float(same_stress), "same index & positions", same_stress),
        _row(sym, "benchmark bars match backtest bars", float(same_bars), "bars.index == base index", same_bars),
    ]


def _benchmark_rows(sym: str, base: BacktestResult, bars: pd.DataFrame, m: dict) -> list[dict]:
    """Buy-and-hold benchmarks on the base backtest's own window (never a wider frame)."""
    first, last = base.bars.index[0], base.bars.index[-1]
    window = bars[(bars.index >= first) & (bars.index <= last)]
    if len(window) < 2:
        raise ValueError(f"Candidate.bars[{sym!r}] does not cover the base backtest window {first}..{last}")
    bh = summarize(buy_and_hold(window, config.BASE_COST_BPS))
    vm, w = vol_matched_buy_and_hold(window, base.daily, config.BASE_COST_BPS)
    return [
        _row(sym, "Sharpe vs buy-and-hold", m["sharpe"] - bh["sharpe"], "> 0", m["sharpe"] > bh["sharpe"]),
        _row(
            sym,
            f"Sharpe vs vol-matched B&H (w={w:.2f})",
            m["sharpe"] - sharpe(vm.daily),
            "> 0",
            m["sharpe"] > sharpe(vm.daily),
        ),
    ]


def _symbol_rows(c: Candidate, sym: str, trials_dir: Path | None, n_perm: int, seed: int) -> list[dict]:
    base, stress, bars = c.base[sym], c.stress[sym], c.bars[sym]
    m = summarize(base)
    perm = permutation_test(base, n_perm=n_perm, seed=seed)
    t_best = max(m["t_net_bps"], m["t_daily"])
    rows = [
        _row(sym, "net Sharpe @5bps", m["sharpe"], ">= 1.0", m["sharpe"] >= 1.0),
        _row(sym, "net Sharpe @10bps", sharpe(stress.daily), "> 0", sharpe(stress.daily) > 0),
        _row(sym, "OOS trades", m["n_trades"], ">= 500", m["n_trades"] >= 500),
        _row(sym, "t-stat (net bps/trade or daily)", t_best, ">= 3", t_best >= 3),
        _row(sym, "profit factor", m["profit_factor"], ">= 1.2", m["profit_factor"] >= 1.2),
        _row(sym, "max drawdown", m["max_drawdown"], ">= -25%", m["max_drawdown"] >= -0.25),
        _row(sym, "positive months", m["pct_positive_months"], ">= 60%", m["pct_positive_months"] >= 0.60),
        _row(sym, "worst calendar year", m["worst_year"], ">= -10%", m["worst_year"] >= -0.10),
        _row(sym, "best month share of P&L", m["best_month_share"], "< 30%", m["best_month_share"] < 0.30),
        _row(sym, "random-timing permutation p", perm["p_value"], "< 0.05", perm["p_value"] < 0.05),
    ]
    return (
        _consistency_rows(sym, base, stress, bars)
        + rows
        + _benchmark_rows(sym, base, bars, m)
        + _registry_rows(c, sym, base, m, trials_dir)
    )


def evaluate(candidate: Candidate, trials_dir: Path | None = None, n_perm: int = 500, seed: int = 0) -> pd.DataFrame:
    """PASS/FAIL table (one row per symbol x criterion). Use :func:`verdict` for the overall call."""
    rows = []
    for sym in config.SYMBOLS:
        present = sym in candidate.base and sym in candidate.stress and sym in candidate.bars
        rows.append(_row(sym, "symbol evaluated", float(present), "BTCUSDT and ETHUSDT", present))
        if present:
            assert_pre_holdout(candidate.base[sym].daily.index, "promotion.evaluate")
            rows += _symbol_rows(candidate, sym, trials_dir, n_perm, seed)
    table = pd.DataFrame(rows)
    table["result"] = np.where(table["pass"], "PASS", "FAIL")
    return table


def verdict(table: pd.DataFrame) -> str:
    return "PASS" if bool(table["pass"].all()) else "FAIL"


def holdout_check(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Holdout gate (net P&L > 0 and Sharpe >= 0.5 at base cost). Only inside ``open_holdout``.

    The verdict is appended to ``config.HOLDOUT_AUDIT_LOG``; the path is not a parameter.
    """
    if not holdout_is_open():
        raise HoldoutViolation("holdout_check must run inside `with open_holdout(reason=...)`")
    rows = []
    for sym in config.SYMBOLS:
        res = results.get(sym)
        ok = (
            res is not None
            and res.daily.index[0] >= config.HOLDOUT_START
            and res.cost_bps_per_side == config.BASE_COST_BPS
        )
        rows.append(_row(sym, "holdout-only data at 5bps", float(ok), f">= {config.HOLDOUT_START.date()}", ok))
        if res is not None:
            pnl, sr = float(np.prod(1 + res.daily.to_numpy()) - 1), sharpe(res.daily)
            rows += [
                _row(sym, "holdout net P&L", pnl, "> 0", pnl > 0),
                _row(sym, "holdout net Sharpe", sr, ">= 0.5", sr >= 0.5),
            ]
    table = pd.DataFrame(rows)
    table["result"] = np.where(table["pass"], "PASS", "FAIL")
    path = Path(config.HOLDOUT_AUDIT_LOG)
    with path.open("a", encoding="utf-8") as fh:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "holdout_check",
            "verdict": verdict(table),
            "rows": table.drop(columns="result").to_dict("records"),
        }
        fh.write(json.dumps(record, default=str) + "\n")
    return table
