"""Rule baselines for the intraday harness (family ``rule_baselines``, kind ``baseline``).

    .venv/bin/python experiments/intraday/rule_baselines.py --smoke   # 2018-2019 only, logs nothing
    .venv/bin/python experiments/intraday/rule_baselines.py           # the one logged OOS run

The pre-registered grid (24 configs per symbol) is ``GRID`` below and is mirrored at the top of
``research/intraday/results/rule_baselines.md``. Every config is backtested once on the stitched
OOS window [EVAL_START, EVAL_END) at 5 and 10 bps/side and logged with ``trials.log_trial``.

Rebuild one config for ``promotion.evaluate`` (from the repo root; nothing is logged)::

    import sys; sys.path.insert(0, "experiments/intraday")
    from rule_baselines import build_candidate
    cand = build_candidate("ema_px_4h_180")   # promotion.Candidate with base, stress, bars
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from profit_engine.intraday import backtest, config, data, features, labels, metrics, promotion, splits, trials
from profit_engine.intraday.provenance import data_hash
from profit_engine.intraday.resample import resample

FAMILY = "rule_baselines"
KIND = "baseline"
SEED = 0  # the rules are deterministic; logged for the registry
RESULTS_MD = config.RESEARCH_DIR / "results" / "rule_baselines.md"
BARS_PER_YEAR = {"15m": 4 * 24 * 365, "1h": 24 * 365, "4h": 6 * 365}


def _cfg(name: str, rule: str, tf: str, **p: object) -> dict:
    return {"name": name, "rule": rule, "tf": tf, "params": {"rule": rule, **p}}


GRID: list[dict] = [
    _cfg("tsmom_1h_24", "tsmom", "1h", n=24),
    _cfg("tsmom_1h_168", "tsmom", "1h", n=168),
    _cfg("tsmom_4h_42", "tsmom", "4h", n=42),
    _cfg("tsmom_4h_180", "tsmom", "4h", n=180),
    _cfg("ema_px_1h_168", "ema_px", "1h", n=168),
    _cfg("ema_px_4h_180", "ema_px", "4h", n=180),
    _cfg("ema_x_1h_24_168", "ema_x", "1h", fast=24, slow=168),
    _cfg("ema_x_4h_42_180", "ema_x", "4h", fast=42, slow=180),
    _cfg("mr_15m_k2", "mean_rev", "15m", h=4, window=672, k=2.0, max_hold=16),
    _cfg("mr_15m_k3", "mean_rev", "15m", h=4, window=672, k=3.0, max_hold=16),
    _cfg("mr_1h_k2", "mean_rev", "1h", h=4, window=720, k=2.0, max_hold=12),
    _cfg("mr_1h_k3", "mean_rev", "1h", h=4, window=720, k=3.0, max_hold=12),
    _cfg("donch_1h_24_24", "donchian", "1h", n=24, hold=24),
    _cfg("donch_1h_168_48", "donchian", "1h", n=168, hold=48),
    _cfg("donch_4h_42_12", "donchian", "4h", n=42, hold=12),
    _cfg("donch_4h_180_42", "donchian", "4h", n=180, hold=42),
    _cfg("voltgt_4h_s30_w42", "vol_target", "4h", sigma=0.30, window=42, band=0.2),
    _cfg("voltgt_4h_s30_w180", "vol_target", "4h", sigma=0.30, window=180, band=0.2),
    _cfg("voltgt_4h_s50_w42", "vol_target", "4h", sigma=0.50, window=42, band=0.2),
    _cfg("voltgt_4h_s50_w180", "vol_target", "4h", sigma=0.50, window=180, band=0.2),
    _cfg("tod_1h_k4_exp", "time_of_day", "1h", k=4, train="expanding"),
    _cfg("tod_1h_k4_roll12", "time_of_day", "1h", k=4, train="rolling", train_months=12),
    _cfg("tod_1h_k8_exp", "time_of_day", "1h", k=8, train="expanding"),
    _cfg("tod_1h_k8_roll12", "time_of_day", "1h", k=8, train="rolling", train_months=12),
]
GRID_BY_NAME = {c["name"]: c for c in GRID}


# --------------------------------------------------------------------------- rules (causal)


def tsmom(df: pd.DataFrame, n: int) -> pd.Series:
    """Long if the N-bar log close return is positive (known at bar t's close)."""
    r = features.log_returns(df, horizons=(n,))[f"ret_{n}"]
    return (r > 0).astype(float).where(r.notna())


def ema_px(df: pd.DataFrame, n: int) -> pd.Series:
    """Long if close > EMA(n) of close."""
    ema = df["close"].ewm(span=n, adjust=False, min_periods=n).mean()
    return (df["close"] > ema).astype(float).where(ema.notna())


def ema_x(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """Long if EMA(fast) > EMA(slow)."""
    f = df["close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
    s = df["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
    return (f > s).astype(float).where(s.notna())


def mean_rev(df: pd.DataFrame, h: int, window: int, k: float, max_hold: int) -> pd.Series:
    """Enter when the h-bar return z-score < -k; exit when z >= 0 or after ``max_hold`` bars."""
    z = features.return_zscore(df, horizon=h, window=window).iloc[:, 0].to_numpy()
    out = np.zeros(len(z))
    held, age = False, 0
    for i, zi in enumerate(z):
        if held:
            age += 1
            if not np.isfinite(zi) or zi >= 0 or age >= max_hold:
                held, age = False, 0
        elif np.isfinite(zi) and zi < -k:
            held, age = True, 0
        out[i] = 1.0 if held else 0.0
    return pd.Series(out, index=df.index)


def donchian(df: pd.DataFrame, n: int, hold: int) -> pd.Series:
    """Breakout: close_t > max(high over the previous n bars). Long for ``hold`` bars after the
    latest breakout (bar t included); a new breakout restarts the clock."""
    upper = df["high"].shift(1).rolling(n, min_periods=n).max()
    brk = (df["close"] > upper).astype(float)
    return brk.rolling(hold, min_periods=1).max()


def vol_target(df: pd.DataFrame, sigma: float, window: int, band: float, tf: str = "4h") -> pd.Series:
    """min(1, sigma / annualised realized vol), moved only when it differs from the current
    target by more than ``band`` (cost control). rv == 0 or NaN keeps the current target."""
    rv = features.realized_vol(df, windows=(window,))[f"rv_{window}"].to_numpy() * np.sqrt(BARS_PER_YEAR[tf])
    out = np.zeros(len(rv))
    cur = np.nan
    for i, v in enumerate(rv):
        if np.isfinite(v) and v > 0:
            raw = min(1.0, sigma / v)
            if not np.isfinite(cur) or abs(raw - cur) > band:
                cur = round(raw, 4)
        out[i] = cur if np.isfinite(cur) else 0.0
    return pd.Series(out, index=df.index)


def time_of_day(
    df: pd.DataFrame,
    k: int,
    train: str,
    train_months: int | None = None,
    eval_start: pd.Timestamp = config.EVAL_START,
    eval_end: pd.Timestamp = config.EVAL_END,
) -> pd.Series:
    """Walk-forward: each month, pick the k decision hours with the best mean next-bar return on
    the fold's purged training rows only; long at the close of those hours in the test month."""
    lab = labels.forward_return(df, 1)  # ret = log(open[t+2]/open[t+1]): what target[t] earns
    has_y = lab["ret"].notna().to_numpy()
    hours = df.index.hour.to_numpy()
    window = pd.DateOffset(months=train_months) if train == "rolling" else None
    folds = splits.walk_forward_splits(
        df.index, lab["t1"], train=train, train_window=window, eval_start=eval_start, eval_end=eval_end
    )
    target = pd.Series(np.nan, index=df.index)
    for fold in folds:
        tr = fold.train & has_y
        if (lab["t1"].to_numpy()[tr] >= fold.test_start).any():
            raise RuntimeError(f"training label reaches test block {fold.test_start}")
        means = pd.Series(lab["ret"].to_numpy()[tr]).groupby(hours[tr]).mean()
        best = means.sort_values(ascending=False).index[:k]
        target[fold.test] = np.isin(hours[fold.test], best).astype(float)
    return target


RULES = {
    "tsmom": tsmom,
    "ema_px": ema_px,
    "ema_x": ema_x,
    "mean_rev": mean_rev,
    "donchian": donchian,
    "vol_target": vol_target,
    "time_of_day": time_of_day,
}


def target_for(cfg: dict, df: pd.DataFrame, lo=config.EVAL_START, hi=config.EVAL_END) -> pd.Series:
    kw = {k: v for k, v in cfg["params"].items() if k != "rule"}
    if cfg["rule"] == "vol_target":
        kw["tf"] = cfg["tf"]
    if cfg["rule"] == "time_of_day":  # walk-forward months = the evaluated window
        kw.update(eval_start=lo, eval_end=hi)
    return RULES[cfg["rule"]](df, **kw)


# --------------------------------------------------------------------------- data + runs


def load_bars() -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, str]]:
    """{tf: {sym: bars}} for every timeframe in the grid, and the raw 5m data hash per symbol."""
    raw = {s: data.load_klines(s) for s in config.SYMBOLS}
    tfs = sorted({c["tf"] for c in GRID})
    bars = {tf: {s: resample(raw[s], tf, incomplete="drop") for s in config.SYMBOLS} for tf in tfs}
    for tf in tfs:
        for s in config.SYMBOLS:
            n_full = len(resample(raw[s], tf, incomplete="flag"))
            print(f"[bars] {s} {tf}: {len(bars[tf][s])} bars, dropped {n_full - len(bars[tf][s])} incomplete")
    return bars, {s: data_hash(raw[s]) for s in config.SYMBOLS}


def oos_slice(df: pd.DataFrame, target: pd.Series, lo: pd.Timestamp, hi: pd.Timestamp) -> tuple:
    keep = (df.index >= lo) & (df.index < hi)
    return df.loc[keep], target.loc[df.index[keep]]


def run_config(cfg: dict, df: pd.DataFrame, lo=config.EVAL_START, hi=config.EVAL_END) -> tuple:
    """Base and stress backtests of the stitched target over [lo, hi), and the bars used."""
    target = target_for(cfg, df, lo, hi)
    b, t = oos_slice(df, target, lo, hi)
    base = backtest.run_backtest(b, t, config.BASE_COST_BPS)
    stress = backtest.run_backtest(b, t, config.STRESS_COST_BPS)
    return base, stress, b


def build_candidate(name: str) -> promotion.Candidate:
    """Rebuild one grid config on both symbols as a ``promotion.Candidate``."""
    cfg = GRID_BY_NAME[name]
    raw = {s: data.load_klines(s) for s in config.SYMBOLS}
    out = {s: run_config(cfg, resample(raw[s], cfg["tf"], incomplete="drop")) for s in config.SYMBOLS}
    return promotion.Candidate(
        FAMILY,
        cfg["params"],
        cfg["tf"],
        base={s: o[0] for s, o in out.items()},
        stress={s: o[1] for s, o in out.items()},
        bars={s: o[2] for s, o in out.items()},
    )


def _row(cfg: dict, sym: str, base, stress) -> dict:
    mb, ms = metrics.summarize(base), metrics.summarize(stress)
    return {
        "name": cfg["name"],
        "rule": cfg["rule"],
        "tf": cfg["tf"],
        "symbol": sym,
        "sharpe_base": mb["sharpe"],
        "sharpe_stress": ms["sharpe"],
        "cagr_base": mb["cagr"],
        "cagr_stress": ms["cagr"],
        "maxdd_base": mb["max_drawdown"],
        "n_trades": mb["n_trades"],
        "trades_per_day": mb["trades_per_day"],
        "net_bps_base": mb["mean_net_bps"],
        "net_bps_stress": ms["mean_net_bps"],
        "pf_base": mb["profit_factor"],
        "pos_months": mb["pct_positive_months"],
        "worst_year": mb["worst_year"],
        "exposure": mb["exposure"],
        "fills": int((base.bars["position"].diff().fillna(base.bars["position"]) != 0).sum()),
        "t_daily": mb["t_daily"],
    }, (mb, ms)


def run_all(log: bool) -> pd.DataFrame:
    bars, hashes = load_bars()
    lo, hi = (config.EVAL_START, config.EVAL_END) if log else (pd.Timestamp("2018-01-01", tz="UTC"), config.EVAL_START)
    rows = []
    for cfg in GRID:
        for sym in config.SYMBOLS:
            base, stress, _ = run_config(cfg, bars[cfg["tf"]][sym], lo, hi)
            row, (mb, ms) = _row(cfg, sym, base, stress)
            if log:
                for res, m in ((base, mb), (stress, ms)):
                    trials.log_trial(
                        FAMILY, cfg["params"], cfg["tf"], sym, res.cost_bps_per_side, m, res.daily,
                        hashes[sym], SEED, kind=KIND, notes=cfg["name"],
                    )  # fmt: skip
            rows.append(row)
            print(
                f"{'OOS' if log else 'SMOKE'} {cfg['name']:<20} {sym} SR {row['sharpe_base']:+.2f}/"
                f"{row['sharpe_stress']:+.2f} trades {row['n_trades']} fills {row['fills']} exp {row['exposure']:.0%}",
                flush=True,
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report


def _fmt_row(r: pd.Series) -> str:
    return (
        f"| {r['name']} | {r['tf']} | {r['symbol'][:3]} | {r['sharpe_base']:.2f} | {r['sharpe_stress']:.2f} | "
        f"{r['cagr_base']:.1%} | {r['maxdd_base']:.1%} | {r['n_trades']} | {r['trades_per_day']:.2f} | "
        f"{r['net_bps_base']:.1f} / {r['net_bps_stress']:.1f} | {r['pf_base']:.2f} | {r['pos_months']:.0%} | "
        f"{r['worst_year']:.1%} | {r['exposure']:.0%} | {r['fills']} |"
    )


def render(df: pd.DataFrame) -> str:
    head = (
        "| config | tf | sym | Sharpe @5 | Sharpe @10 | CAGR @5 | MaxDD @5 | trades | trades/day | "
        "net bps/trade @5 / @10 | PF @5 | +months | worst year @5 | exposure | fills |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    body = "\n".join(_fmt_row(r) for _, r in df.sort_values("sharpe_base", ascending=False).iterrows())
    pivot = df.pivot(index="name", columns="symbol", values="sharpe_base")
    pivot["min"] = pivot.min(axis=1)
    pivot = pivot.sort_values("min", ascending=False)
    both = "| config | BTC Sharpe @5 | ETH Sharpe @5 | min |\n|---|---|---|---|\n" + "\n".join(
        f"| {n} | {r['BTCUSDT']:.2f} | {r['ETHUSDT']:.2f} | {r['min']:.2f} |" for n, r in pivot.iterrows()
    )
    return (
        f"## Results (OOS 2020-01-01..2026-06-30, generated by the script, n_trials in registry after "
        f"this run = {trials.n_trials()})\n\nEvery run, sorted by base-cost Sharpe:\n\n{head}\n{body}\n\n"
        f"Both symbols side by side, sorted by the weaker symbol:\n\n{both}\n"
    )


def write_results(df: pd.DataFrame, path: Path = RESULTS_MD) -> None:
    text = path.read_text()
    block = f"<!-- RESULTS:BEGIN -->\n{render(df)}<!-- RESULTS:END -->"
    new = re.sub(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->", lambda _: block, text, flags=re.S)
    path.write_text(new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run on 2018-2019 training data; log nothing")
    args = ap.parse_args()
    df = run_all(log=not args.smoke)
    if args.smoke:
        print(df[["name", "symbol", "sharpe_base", "n_trades", "exposure"]].to_string(index=False))
        return
    df.to_csv(sys.stdout, index=False)
    write_results(df)
    print("wrote", RESULTS_MD, "| registry N =", trials.n_trials())


if __name__ == "__main__":
    main()
