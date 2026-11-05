"""LightGBM triple-barrier classifier (family ``lgbm``, kind ``model``).

    .venv/bin/python experiments/intraday/lgbm.py --smoke [--only NAME]   # test year 2019, logs nothing
    .venv/bin/python experiments/intraday/lgbm.py                         # the one logged OOS run

The pre-registered grid (12 configs per symbol) is ``GRID`` below and is mirrored at the top of
``research/intraday/results/lgbm.md``. Every config is walked forward monthly, its stitched OOS
target is backtested once over [EVAL_START, EVAL_END) at 5 and 10 bps/side and logged.

Rebuild one config for ``promotion.evaluate`` (from the repo root; nothing is logged)::

    import sys; sys.path.insert(0, "experiments/intraday")
    from lgbm import build_candidate
    cand = build_candidate("lgbm_4h_h30_pt2_sl1")   # promotion.Candidate with base, stress, bars
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from profit_engine.intraday import backtest, config, data, features, labels, metrics, promotion, splits, trials
from profit_engine.intraday.provenance import data_hash
from profit_engine.intraday.resample import resample

FAMILY = "lgbm"
SEED = 7
RESULTS_MD = config.RESEARCH_DIR / "results" / "lgbm.md"
BARS_PER_DAY = {"15m": 96, "1h": 24, "4h": 6}
LABEL_MIN_RET = 2 * config.BASE_COST_BPS / 1e4  # y = 1 iff the barrier return beats one round trip
INNER_SPLITS = 4
RETRAIN_MONTHS = {"15m": 3, "1h": 1, "4h": 1}  # 15m quarterly: monthly took > 20 min per run on the shared box
QUANTILES = np.round(np.arange(0.20, 0.951, 0.05), 2)
MODEL_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 500,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 10.0,
    "reg_alpha": 1.0,
    "min_split_gain": 0.0,
}


def _cfg(tf: str, h: int, pt: float, sl: float) -> dict:
    name = f"lgbm_{tf}_h{h}_pt{pt:g}_sl{sl:g}"
    params = {
        "model": "lgbm_binary",
        "h": h,
        "pt": pt,
        "sl": sl,
        "vol_window_days": 7,
        "label_min_ret": LABEL_MIN_RET,
        "weights": "uniqueness",
        "retrain": "quarterly" if RETRAIN_MONTHS[tf] == 3 else "monthly",
        "threshold": f"inner_purged_kfold{INNER_SPLITS}_net_pnl_quantiles",
        **MODEL_PARAMS,
    }
    return {"name": name, "tf": tf, "h": h, "pt": pt, "sl": sl, "params": params}


GRID: list[dict] = [
    _cfg(tf, h, pt, 1.0)
    for tf, hs in (("4h", (6, 30)), ("1h", (6, 24)), ("15m", (16, 96)))  # cheapest first
    for h in hs
    for pt in (1.0, 2.0)
]
GRID_BY_NAME = {c["name"]: c for c in GRID}


# --------------------------------------------------------------------------- features + labels


def feature_params(tf: str) -> tuple[dict, tuple[int, ...]]:
    d = BARS_PER_DAY[tf]
    q = max(1, d // 6)
    params = {
        "log_returns": {"horizons": tuple(sorted({1, q, d, 7 * d, 30 * d}))},
        "realized_vol": {"windows": (d, 7 * d)},
        "return_zscore": {"horizon": q, "window": 7 * d},
        "taker_imbalance": {"windows": (1, d)},
        "vwap_deviation": {"windows": (d, 7 * d)},
        "volume_ratio": {"short": max(2, q), "long": 7 * d},
    }
    return params, (1, d, 7 * d)


def build_xy(df: pd.DataFrame, other: pd.DataFrame, cfg: dict) -> dict:
    """Features, triple-barrier labels, weights' inputs and the 1-bar return on one common index."""
    d = BARS_PER_DAY[cfg["tf"]]
    fparams, cross = feature_params(cfg["tf"])
    x = features.make_feature_frame(df, other, params=fparams, cross_horizons=cross)
    df = df.loc[x.index]
    close = df["close"]
    for n in (7 * d, 30 * d):  # causal: adjust=False EWM uses bars <= t only
        x[f"emadist_{n}"] = np.log(close / close.ewm(span=n, adjust=False, min_periods=n).mean())
    x["rvratio"] = np.log(x[f"rv_{d}"] / x[f"rv_{7 * d}"].where(x[f"rv_{7 * d}"] > 0))
    x = x.replace([np.inf, -np.inf], np.nan)
    tb = labels.triple_barrier(df, cfg["h"], cfg["pt"], cfg["sl"], vol_window=7 * d)
    r1 = labels.forward_return(df, 1)["ret"].to_numpy()
    end_pos = df.index.get_indexer(tb["t1"])  # -1 where unlabelled
    return {
        "df": df,
        "x": x,
        "y": (tb["ret"] > LABEL_MIN_RET).astype(int).to_numpy(),
        "t1": tb["t1"],
        "end_pos": end_pos,
        "r1": np.nan_to_num(r1),
        "has_x": x.notna().all(axis=1).to_numpy(),
        "has_y": tb["label"].notna().to_numpy() & (end_pos >= 0),
    }


def uniqueness_weights(rows: np.ndarray, end_pos: np.ndarray, n: int) -> np.ndarray:
    """Average uniqueness of each training label, from training labels only.

    Label i (bar position ``rows[k]``) is exposed on bars ``i+1 .. end_pos-1``. Concurrency at a
    bar = number of training labels exposed on it; uniqueness = mean of 1/concurrency over the
    label's span. Every training label ends before the test block, so this is causal.
    """
    start, stop = rows + 1, np.maximum(end_pos[rows], rows + 2)  # stop is exclusive
    conc = np.zeros(n + 1)
    np.add.at(conc, start, 1.0)
    np.add.at(conc, np.minimum(stop, n), -1.0)
    conc = np.cumsum(conc)[:n]
    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1.0), 0.0)
    cum = np.concatenate([[0.0], np.cumsum(inv)])
    stop = np.minimum(stop, n)
    w = (cum[stop] - cum[start]) / np.maximum(stop - start, 1)
    return w / w.mean()


# --------------------------------------------------------------------------- model + threshold


def make_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        **MODEL_PARAMS,
        objective="binary",
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        n_jobs=4,
        verbose=-1,
    )


def block_pnl(p: np.ndarray, r1: np.ndarray, block: np.ndarray, thr: float, cost: float) -> float:
    """Summed net log P&L of ``p > thr`` held bar by bar; every block starts flat."""
    pos = (p > thr).astype(float)
    turn = np.abs(np.diff(pos, prepend=0.0))
    starts = np.flatnonzero(np.diff(block, prepend=block[0] - 1) != 0)
    turn[starts] = pos[starts]
    return float((pos * r1).sum() - cost * turn.sum())


def tune_threshold(oof: np.ndarray, r1: np.ndarray, block: np.ndarray) -> tuple[float, float]:
    """Best absolute threshold among OOF-probability quantiles; inf (flat) if none makes money."""
    cost = config.BASE_COST_BPS / 1e4
    cands = np.quantile(oof, QUANTILES)
    pnl = np.array([block_pnl(oof, r1, block, t, cost) for t in cands])
    k = int(np.argmax(pnl))
    if pnl[k] <= 0:
        return np.inf, float("nan")
    return float(cands[k]), float(QUANTILES[k])


def fit_fold(xy: dict, train: np.ndarray, fold: splits.Fold) -> tuple[lgb.LGBMClassifier, float, float]:
    """Inner purged CV on the fold's training rows picks the threshold; then refit on all of them."""
    rows = np.flatnonzero(train)
    idx = xy["df"].index[rows]
    t1 = pd.Series(xy["t1"].to_numpy()[rows], index=idx)
    x_tr, y_tr, r1_tr = xy["x"].iloc[rows], xy["y"][rows], xy["r1"][rows]
    w = uniqueness_weights(rows, xy["end_pos"], len(xy["df"]))
    oof, block = np.full(len(rows), np.nan), np.full(len(rows), -1)
    for k, (itr, iva) in enumerate(splits.purged_kfold(idx, t1, n_splits=INNER_SPLITS, within=fold)):
        m = make_model().fit(x_tr.iloc[itr], y_tr[itr], sample_weight=w[itr])
        oof[iva], block[iva] = m.predict_proba(x_tr.iloc[iva])[:, 1], k
    thr, q = tune_threshold(oof, r1_tr, block)
    final = make_model().fit(x_tr, y_tr, sample_weight=w)
    return final, thr, q


def oos_positions(df: pd.DataFrame, other: pd.DataFrame, cfg: dict, lo, hi) -> tuple[pd.Series, dict]:
    """Monthly walk-forward over [lo, hi): refit on purged past rows, predict the next month only."""
    xy = build_xy(df, other, cfg)
    target = pd.Series(0.0, index=xy["df"].index)
    folds = splits.walk_forward_splits(
        xy["df"].index, xy["t1"], test_months=RETRAIN_MONTHS[cfg["tf"]], eval_start=lo, eval_end=hi, min_train_rows=2000
    )
    info = {"thr": [], "q": [], "importance": []}
    for fold in folds:
        train, test = fold.train & xy["has_x"] & xy["has_y"], fold.test & xy["has_x"]
        if (xy["t1"].to_numpy()[train] >= fold.test_start).any():
            raise RuntimeError(f"training label reaches test block {fold.test_start}")
        model, thr, q = fit_fold(xy, train, fold)
        if test.any() and np.isfinite(thr):
            p = model.predict_proba(xy["x"].iloc[np.flatnonzero(test)])[:, 1]
            target.iloc[np.flatnonzero(test)] = (p > thr).astype(float)
        info["thr"].append(thr)
        info["q"].append(q)
        gain = model.booster_.feature_importance(importance_type="gain")
        info["importance"].append(pd.Series(gain / max(gain.sum(), 1e-12), index=xy["x"].columns))
        if fold.number % 12 == 0:
            print(f"  {cfg['name']} fold {fold.number}/{len(folds)} {fold.test_start.date()}", flush=True)
    info["n_folds"] = len(folds)
    return target, info


# --------------------------------------------------------------------------- data + runs


def load_bars(tfs) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, str]]:
    raw = {s: data.load_klines(s) for s in config.SYMBOLS}
    bars = {tf: {s: resample(raw[s], tf, incomplete="drop") for s in config.SYMBOLS} for tf in tfs}
    return bars, {s: data_hash(raw[s]) for s in config.SYMBOLS}


def _other(sym: str) -> str:
    return "ETHUSDT" if sym == "BTCUSDT" else "BTCUSDT"


def run_config(cfg: dict, bars: dict[str, pd.DataFrame], sym: str, lo=config.EVAL_START, hi=config.EVAL_END):
    """Base and stress backtests of the stitched OOS target over [lo, hi), the bars used, fold info."""
    target, info = oos_positions(bars[sym], bars[_other(sym)], cfg, lo, hi)
    keep = (target.index >= lo) & (target.index < hi)
    b, t = bars[sym].loc[target.index[keep]], target[keep]
    base = backtest.run_backtest(b, t, config.BASE_COST_BPS)
    stress = backtest.run_backtest(b, t, config.STRESS_COST_BPS)
    return base, stress, b, info


def build_candidate(name: str) -> promotion.Candidate:
    """Rebuild one grid config on both symbols as a ``promotion.Candidate`` (nothing is logged)."""
    cfg = GRID_BY_NAME[name]
    bars, _ = load_bars([cfg["tf"]])
    out = {s: run_config(cfg, bars[cfg["tf"]], s) for s in config.SYMBOLS}
    return promotion.Candidate(
        FAMILY,
        cfg["params"],
        cfg["tf"],
        base={s: o[0] for s, o in out.items()},
        stress={s: o[1] for s, o in out.items()},
        bars={s: o[2] for s, o in out.items()},
    )


def _row(cfg: dict, sym: str, mb: dict, ms: dict, info: dict) -> dict:
    thr = np.array(info["thr"])
    last = 2 if RETRAIN_MONTHS[cfg["tf"]] == 3 else 6  # final ~6 months of models
    imp = pd.concat(info["importance"][-last:], axis=1).mean(axis=1).sort_values(ascending=False)
    return {
        "name": cfg["name"],
        "tf": cfg["tf"],
        "symbol": sym,
        "sharpe_base": mb["sharpe"],
        "sharpe_stress": ms["sharpe"],
        "cagr_base": mb["cagr"],
        "maxdd_base": mb["max_drawdown"],
        "n_trades": mb["n_trades"],
        "trades_per_day": mb["trades_per_day"],
        "net_bps_base": mb["mean_net_bps"],
        "net_bps_stress": ms["mean_net_bps"],
        "pf_base": mb["profit_factor"],
        "pos_months": mb["pct_positive_months"],
        "worst_year": mb["worst_year"],
        "yearly": mb["yearly_returns"],
        "exposure": mb["exposure"],
        "flat_folds": int((~np.isfinite(thr)).sum()),
        "n_folds": info["n_folds"],
        "med_q": float(np.nanmedian(info["q"])) if np.isfinite(info["q"]).any() else float("nan"),
        "t_daily": mb["t_daily"],
        "importance": imp,
    }


def run_all(log: bool, only: list[str] | None = None) -> pd.DataFrame:
    grid = [c for c in GRID if not only or c["name"] in only]
    bars, hashes = load_bars(sorted({c["tf"] for c in grid}))
    lo, hi = (config.EVAL_START, config.EVAL_END) if log else (pd.Timestamp("2019-01-01", tz="UTC"), config.EVAL_START)
    rows = []
    for cfg in grid:
        for sym in config.SYMBOLS:
            t0 = time.time()
            base, stress, _, info = run_config(cfg, bars[cfg["tf"]], sym, lo, hi)
            mb, ms = metrics.summarize(base), metrics.summarize(stress)
            if log:
                for res, m in ((base, mb), (stress, ms)):
                    trials.log_trial(
                        FAMILY, cfg["params"], cfg["tf"], sym, res.cost_bps_per_side, m, res.daily,
                        hashes[sym], SEED, kind="model", notes=cfg["name"],
                    )  # fmt: skip
            row = _row(cfg, sym, mb, ms, info)
            rows.append(row)
            print(
                f"{'OOS' if log else 'SMOKE'} {cfg['name']:<22} {sym} SR {row['sharpe_base']:+.2f}/"
                f"{row['sharpe_stress']:+.2f} trades {row['n_trades']} exp {row['exposure']:.0%} "
                f"flat {row['flat_folds']}/{row['n_folds']} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report


def _fmt_row(r: pd.Series) -> str:
    return (
        f"| {r['name']} | {r['tf']} | {r['symbol'][:3]} | {r['sharpe_base']:.2f} | {r['sharpe_stress']:.2f} | "
        f"{r['cagr_base']:.1%} | {r['maxdd_base']:.1%} | {r['n_trades']} | {r['trades_per_day']:.2f} | "
        f"{r['net_bps_base']:.1f} / {r['net_bps_stress']:.1f} | {r['pf_base']:.2f} | {r['pos_months']:.0%} | "
        f"{r['worst_year']:.1%} | {r['exposure']:.0%} | {r['flat_folds']}/{r['n_folds']} | {r['med_q']:.2f} |"
    )


def _years_table(df: pd.DataFrame) -> str:
    years = sorted({y for d in df["yearly"] for y in d})
    head = "| config | sym | " + " | ".join(str(y) for y in years) + " |\n|---|---|" + "---|" * len(years)
    lines = [
        f"| {r['name']} | {r['symbol'][:3]} | "
        + " | ".join(f"{r['yearly'].get(y, float('nan')):.1%}" for y in years)
        + " |"
        for _, r in df.sort_values("sharpe_base", ascending=False).iterrows()
    ]
    return head + "\n" + "\n".join(lines)


def _importance_block(df: pd.DataFrame) -> str:
    top = df.sort_values("sharpe_base", ascending=False).groupby("symbol").head(1)
    out = []
    for _, r in top.iterrows():
        feats = ", ".join(f"{k} {v:.0%}" for k, v in r["importance"].head(10).items())
        out.append(f"* best {r['symbol']} config `{r['name']}`: {feats}")
    fam = pd.concat(list(df["importance"]), axis=1).mean(axis=1).sort_values(ascending=False)
    out.append("* family mean over all 24 runs: " + ", ".join(f"{k} {v:.0%}" for k, v in fam.head(12).items()))
    return "\n".join(out)


def render(df: pd.DataFrame) -> str:
    head = (
        "| config | tf | sym | Sharpe @5 | Sharpe @10 | CAGR @5 | MaxDD @5 | trades | trades/day | "
        "net bps/trade @5 / @10 | PF @5 | +months | worst year @5 | exposure | flat folds | median q |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
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
        f"this run = {trials.n_trials()})\n\nEvery run, sorted by base-cost Sharpe (`flat folds` = test "
        f"months where no inner-CV threshold made money, so the model stayed flat; `median q` = median "
        f"chosen OOF-probability quantile):\n\n{head}\n{body}\n\nBoth symbols side by side, sorted by the "
        f"weaker symbol:\n\n{both}\n\nCalendar-year net returns @5 bps:\n\n{_years_table(df)}\n\n"
        "Feature importance (normalised gain, mean of the models trading 2026-01..2026-06: the final 6 monthly "
        f"or 2 quarterly fits):\n\n"
        f"{_importance_block(df)}\n"
    )


def write_results(df: pd.DataFrame, path: Path = RESULTS_MD) -> None:
    text = path.read_text()
    block = f"<!-- RESULTS:BEGIN -->\n{render(df)}<!-- RESULTS:END -->"
    path.write_text(re.sub(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->", lambda _: block, text, flags=re.S))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="test months 2019 only (training data); log nothing")
    ap.add_argument("--only", nargs="*", help="smoke only: restrict to these config names")
    args = ap.parse_args()
    if args.only and not args.smoke:
        sys.exit("--only is for --smoke: the logged run evaluates the whole pre-registered grid")
    df = run_all(log=not args.smoke, only=args.only)
    cols = ["name", "symbol", "sharpe_base", "sharpe_stress", "n_trades", "exposure", "flat_folds", "worst_year"]
    print(df[cols].to_string(index=False))
    if args.smoke:
        return
    write_results(df)
    print("wrote", RESULTS_MD, "| registry N =", trials.n_trials())


if __name__ == "__main__":
    main()
