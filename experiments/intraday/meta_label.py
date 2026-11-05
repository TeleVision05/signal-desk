"""Meta-labeling on the rule baselines' primaries (family ``meta_label``, kind ``model``).

    .venv/bin/python experiments/intraday/meta_label.py --smoke   # 2019 only (training data), logs nothing
    .venv/bin/python experiments/intraday/meta_label.py           # the one logged OOS run

A primary rule from ``rule_baselines`` decides WHEN a long trade could happen. A secondary model
(logistic regression or a small LightGBM) predicts at the signal bar whether that whole trade will
be net-profitable after a round trip at base cost; the trade is taken only if p > threshold. The
threshold is tuned per walk-forward fold on purged inner-CV out-of-fold probabilities of the fold's
training events only. The pre-registered grid (24 configs per symbol) is ``GRID`` and is mirrored at
the top of ``research/intraday/results/meta_label.md``.

Rebuild one config for ``promotion.evaluate`` (from the repo root; nothing is logged)::

    import sys; sys.path.insert(0, "experiments/intraday")
    from meta_label import build_candidate
    cand = build_candidate("meta_logreg_tsmom_4h_180")   # promotion.Candidate with base, stress, bars
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule_baselines as rb

from profit_engine.intraday import backtest, config, data, features, metrics, promotion, splits, trials
from profit_engine.intraday.provenance import data_hash
from profit_engine.intraday.resample import resample

FAMILY = "meta_label"
SEED = 7
RESULTS_MD = config.RESEARCH_DIR / "results" / "meta_label.md"
LABEL_COST_BPS = config.BASE_COST_BPS  # per side; the label is net of one round trip
THRESHOLDS = (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)  # 0 = take every primary trade
INNER_SPLITS = 5
MIN_TRAIN_EVENTS = 40
WF_START = pd.Timestamp("2019-10-01", tz="UTC")  # decisions for trades held into 2020-01-01
BARS_PER_DAY = {"15m": 96, "1h": 24, "4h": 6}
SMOKE = (pd.Timestamp("2019-01-01", tz="UTC"), config.EVAL_START)

PRIMARIES = (
    "tsmom_1h_24", "tsmom_1h_168", "tsmom_4h_42", "tsmom_4h_180",
    "donch_1h_24_24", "donch_1h_168_48", "donch_4h_42_12", "donch_4h_180_42",
    "mr_15m_k2", "mr_15m_k3", "mr_1h_k2", "mr_1h_k3",
)  # fmt: skip
MODEL_PARAMS = {
    "logreg": {"C": 0.1, "penalty": "l2", "max_iter": 1000, "impute": "median", "scale": True},
    "lgbm": {
        "n_estimators": 100, "learning_rate": 0.05, "num_leaves": 7, "max_depth": 3, "min_child_samples": 20,
        "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
    },
}  # fmt: skip


def _cfg(model: str, primary: str) -> dict:
    p = rb.GRID_BY_NAME[primary]
    params = {
        "primary": primary,
        "primary_params": p["params"],
        "model": model,
        "model_params": MODEL_PARAMS[model],
        "thresholds": list(THRESHOLDS),
        "objective": "oof_sum_net_taken",
        "label_cost_bps_per_side": LABEL_COST_BPS,
        "features": "meta_v1",
        "retrain": "monthly_expanding",
        "wf_start": str(WF_START.date()),
        "min_train_events": MIN_TRAIN_EVENTS,
    }
    return {"name": f"meta_{model}_{primary}", "primary": primary, "model": model, "tf": p["tf"], "params": params}


GRID: list[dict] = [_cfg(m, p) for m in MODEL_PARAMS for p in PRIMARIES]
GRID_BY_NAME = {c["name"]: c for c in GRID}


# --------------------------------------------------------------------------- events + labels


def extract_events(df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """One row per primary trade, indexed by its signal bar ``s``.

    ``s`` = first bar of a run of target 1, ``e`` = first bar back at 0 (may be len(df)). The
    simulator fills at open[s+1] and exits at open[e+1]; ``t1`` = open time of bar e+1.
    """
    pos = (target.reindex(df.index).fillna(0.0).to_numpy() > 0).astype(int)
    d = np.diff(np.concatenate([[0], pos, [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    n, o = len(df), df["open"].to_numpy(dtype="float64")
    ok = e + 1 < n
    gross = np.full(len(s), np.nan)
    gross[ok] = np.log(o[e[ok] + 1] / o[s[ok] + 1])
    t1 = pd.Series(pd.NaT, index=df.index[s], dtype=df.index.dtype)
    t1[ok] = df.index[e[ok] + 1]
    net = gross - 2 * LABEL_COST_BPS / 1e4
    ev = pd.DataFrame({"s": s, "e": e, "gross": gross, "net": net, "t1": t1}, index=df.index[s])
    ev["y"] = (ev["net"] > 0).astype(float).where(ev["net"].notna())
    return ev


def _signal_strength(cfg: dict, df: pd.DataFrame) -> pd.Series:
    p, close = cfg["params"], np.log(df["close"])
    r1 = close.diff()
    if cfg["rule"] == "tsmom":
        n = p["n"]
        return close.diff(n) / (r1.rolling(n, min_periods=n).std() * np.sqrt(n))
    if cfg["rule"] == "donchian":
        upper = df["high"].shift(1).rolling(p["n"], min_periods=p["n"]).max()
        rv = r1.rolling(BARS_PER_DAY[cfg["tf"]], min_periods=2).std()
        return np.log(df["close"] / upper) / rv
    return features.return_zscore(df, horizon=p["h"], window=p["window"]).iloc[:, 0]


def bar_features(df: pd.DataFrame, other: pd.DataFrame, tf: str, pcfg: dict) -> pd.DataFrame:
    """Causal per-bar features on ``df.index`` (bars missing in ``other`` get NaN cross features)."""
    d = BARS_PER_DAY[tf]
    w, m = 7 * d, 30 * d
    params = {
        "log_returns": {"horizons": (1, d, w, m)},
        "realized_vol": {"windows": (d, w, m)},
        "return_zscore": {"horizon": d, "window": m},
        "taker_imbalance": {"windows": (1, d)},
        "vwap_deviation": {"windows": (d, w)},
        "volume_ratio": {"short": d, "long": m},
    }
    x = features.make_feature_frame(df, other, params=params, cross_horizons=(1, d, w)).reindex(df.index)
    x["volreg"] = np.log(x[f"rv_{d}"] / x[f"rv_{m}"])
    x["trend_w"] = x[f"ret_{w}"] / (x[f"rv_{w}"] * np.sqrt(w))
    x["trend_m"] = x[f"ret_{m}"] / (x[f"rv_{m}"] * np.sqrt(m))
    x["signal_strength"] = _signal_strength(pcfg, df)
    return x.replace([np.inf, -np.inf], np.nan)


def event_features(ev: pd.DataFrame, bars_x: pd.DataFrame) -> pd.DataFrame:
    """Bar features at each signal bar plus the primary's own record of already-exited trades."""
    x = bars_x.loc[ev.index].copy()
    prev = ev["net"].shift(1)
    x["prev_net"] = prev
    x["prev10_mean"] = prev.rolling(10, min_periods=3).mean()
    x["prev10_win"] = (prev > 0).astype(float).where(prev.notna()).rolling(10, min_periods=3).mean()
    gap = ev["s"].to_numpy() - np.concatenate([[np.nan], ev["e"].to_numpy()[:-1]])
    x["log_gap"] = np.log1p(gap)
    return x


# --------------------------------------------------------------------------- secondary model


def make_model(kind: str):
    if kind == "logreg":
        lr = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), lr)
    return lgb.LGBMClassifier(**MODEL_PARAMS["lgbm"], random_state=SEED, n_jobs=1, verbose=-1)


def fit_predict(kind: str, xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray) -> np.ndarray:
    if len(ytr) == 0 or len(np.unique(ytr)) < 2:
        return np.full(len(xte), float(ytr.mean()) if len(ytr) else 0.5)
    return make_model(kind).fit(xtr, ytr).predict_proba(xte)[:, 1]


def choose_threshold(kind: str, ev: pd.DataFrame, x: np.ndarray, rows: np.ndarray, fold: splits.Fold) -> float:
    """Threshold maximising the sum of net returns of taken trades, out-of-fold, training rows only."""
    idx = ev.index[rows]
    inner = splits.purged_kfold(idx, ev["t1"][rows], n_splits=INNER_SPLITS, within=fold)
    xr, yr, net = x[rows], ev["y"].to_numpy()[rows], ev["net"].to_numpy()[rows]
    oof = np.full(len(idx), np.nan)
    for tr, va in inner:
        if tr.sum() >= MIN_TRAIN_EVENTS // 2:
            oof[va] = fit_predict(kind, xr[tr], yr[tr], xr[va])
    ok = np.isfinite(oof)
    if not ok.any():
        return 0.0
    scores = [net[ok].sum() if t == 0 else net[ok][oof[ok] > t].sum() for t in THRESHOLDS]
    return THRESHOLDS[int(np.argmax(scores))]


def meta_decisions(cfg: dict, ev: pd.DataFrame, xf: pd.DataFrame, hi: pd.Timestamp, wf_start: pd.Timestamp):
    """Walk-forward take/skip per event; returns (take, prob, threshold per event)."""
    x = xf.to_numpy(dtype="float64")
    take, prob, thr_of = np.zeros(len(ev), bool), np.full(len(ev), np.nan), np.full(len(ev), np.nan)
    has_y = ev["y"].notna().to_numpy()
    folds = splits.walk_forward_splits(ev.index, ev["t1"], eval_start=wf_start, eval_end=hi)
    for fold in folds:
        rows = fold.train & has_y
        if rows.sum() < MIN_TRAIN_EVENTS or len(np.unique(ev["y"].to_numpy()[rows])) < 2:
            take[fold.test] = True  # pre-registered fallback: the primary
            continue
        if (ev["t1"][rows] >= fold.test_start).any():
            raise RuntimeError(f"training label reaches test block {fold.test_start}")
        thr = choose_threshold(cfg["model"], ev, x, rows, fold)
        p = fit_predict(cfg["model"], x[rows], ev["y"].to_numpy()[rows], x[fold.test])
        prob[fold.test], thr_of[fold.test] = p, thr
        take[fold.test] = True if thr == 0 else p > thr
    return take, prob, thr_of


def target_from_events(df: pd.DataFrame, ev: pd.DataFrame, take: np.ndarray) -> pd.Series:
    out = np.zeros(len(df))
    for s, e in zip(ev["s"].to_numpy()[take], ev["e"].to_numpy()[take], strict=True):
        out[s:e] = 1.0
    return pd.Series(out, index=df.index)


# --------------------------------------------------------------------------- runs


def run_meta(cfg: dict, df: pd.DataFrame, other: pd.DataFrame, lo=config.EVAL_START, hi=config.EVAL_END):
    """(base, stress, bars, diagnostics) of the meta-labeled primary over [lo, hi)."""
    pcfg = rb.GRID_BY_NAME[cfg["primary"]]
    ev = extract_events(df, rb.target_for(pcfg, df))
    xf = event_features(ev, bar_features(df, other, cfg["tf"], pcfg))
    wf_start = WF_START if lo == config.EVAL_START else lo - pd.DateOffset(months=3)
    take, prob, thr = meta_decisions(cfg, ev, xf, hi, wf_start)
    b, t = rb.oos_slice(df, target_from_events(df, ev, take), lo, hi)
    base = backtest.run_backtest(b, t, config.BASE_COST_BPS)
    stress = backtest.run_backtest(b, t, config.STRESS_COST_BPS)
    return base, stress, b, _diagnostics(ev, take, prob, thr, lo, hi)


def _diagnostics(ev: pd.DataFrame, take, prob, thr, lo, hi) -> dict:
    """Where the edge comes from: OOS events taken vs skipped (event-level, labels at base cost)."""
    oos = (ev.index >= lo) & (ev.index < hi) & ev["net"].notna().to_numpy()
    net, y = ev["net"].to_numpy() * 1e4, ev["y"].to_numpy()
    tk, sk = oos & take, oos & ~take
    scored = oos & np.isfinite(prob)
    auc = roc_auc_score(y[scored], prob[scored]) if scored.sum() > 10 and len(np.unique(y[scored])) == 2 else np.nan
    in_oos = (ev.index >= lo) & (ev.index < hi)
    thr_o = thr[in_oos & np.isfinite(thr)]
    return {
        "n_events": int(oos.sum()),
        "take_rate": float(tk.sum() / max(oos.sum(), 1)),
        "net_bps_all": float(net[oos].mean()) if oos.any() else np.nan,
        "net_bps_taken": float(net[tk].mean()) if tk.any() else np.nan,
        "net_bps_skipped": float(net[sk].mean()) if sk.any() else np.nan,
        "win_taken": float(y[tk].mean()) if tk.any() else np.nan,
        "win_skipped": float(y[sk].mean()) if sk.any() else np.nan,
        "auc_oos": float(auc),
        "pct_events_filtering": float((thr_o > 0).mean()) if len(thr_o) else 0.0,
    }


def build_candidate(name: str) -> promotion.Candidate:
    """Rebuild one grid config on both symbols as a ``promotion.Candidate`` (logs nothing)."""
    cfg = GRID_BY_NAME[name]
    bars = {s: resample(data.load_klines(s), cfg["tf"], incomplete="drop") for s in config.SYMBOLS}
    out = {s: run_meta(cfg, bars[s], bars[_other(s)]) for s in config.SYMBOLS}
    return promotion.Candidate(
        FAMILY,
        cfg["params"],
        cfg["tf"],
        base={s: o[0] for s, o in out.items()},
        stress={s: o[1] for s, o in out.items()},
        bars={s: o[2] for s, o in out.items()},
    )


def _other(sym: str) -> str:
    return "ETHUSDT" if sym == "BTCUSDT" else "BTCUSDT"


def _metrics_row(prefix: str, base, stress) -> tuple[dict, dict, dict]:
    mb, ms = metrics.summarize(base), metrics.summarize(stress)
    row = {
        f"{prefix}sharpe_base": mb["sharpe"],
        f"{prefix}sharpe_stress": ms["sharpe"],
        f"{prefix}cagr_base": mb["cagr"],
        f"{prefix}maxdd_base": mb["max_drawdown"],
        f"{prefix}n_trades": mb["n_trades"],
        f"{prefix}trades_per_day": mb["trades_per_day"],
        f"{prefix}net_bps_base": mb["mean_net_bps"],
        f"{prefix}net_bps_stress": ms["mean_net_bps"],
        f"{prefix}pf_base": mb["profit_factor"],
        f"{prefix}pos_months": mb["pct_positive_months"],
        f"{prefix}worst_year": mb["worst_year"],
        f"{prefix}exposure": mb["exposure"],
        f"{prefix}t_daily": mb["t_daily"],
    }
    return row, mb, ms


def _primary_logged(base, stress) -> bool:
    log = trials.load_trials()
    log = log[log["family"] == rb.FAMILY]
    return all((log["returns_hash"] == data_hash(r.daily)).any() for r in (base, stress))


def _primary_cache(bars, lo, hi) -> dict:
    """Primary-alone runs on identical bars/window, keyed by (primary, symbol). Not re-logged."""
    out = {}
    for p in PRIMARIES:
        pcfg = rb.GRID_BY_NAME[p]
        for sym in config.SYMBOLS:
            base, stress, _ = rb.run_config(pcfg, bars[pcfg["tf"]][sym], lo, hi)
            row, _, _ = _metrics_row("prim_", base, stress)
            row["prim_logged"] = _primary_logged(base, stress) if lo == config.EVAL_START else False
            out[(p, sym)] = row
    return out


def run_all(log: bool) -> pd.DataFrame:
    bars, hashes = rb.load_bars()
    lo, hi = (config.EVAL_START, config.EVAL_END) if log else SMOKE
    prim = _primary_cache(bars, lo, hi)
    rows = []
    for cfg in GRID:
        for sym in config.SYMBOLS:
            t0 = time.time()
            base, stress, _, diag = run_meta(cfg, bars[cfg["tf"]][sym], bars[cfg["tf"]][_other(sym)], lo, hi)
            row, mb, ms = _metrics_row("", base, stress)
            if log:
                for res, m in ((base, mb), (stress, ms)):
                    trials.log_trial(
                        FAMILY, cfg["params"], cfg["tf"], sym, res.cost_bps_per_side, {**m, **diag}, res.daily,
                        hashes[sym], SEED, kind="model", notes=cfg["name"],
                    )  # fmt: skip
            row.update({"name": cfg["name"], "primary": cfg["primary"], "model": cfg["model"], "tf": cfg["tf"]})
            row.update({"symbol": sym, **diag, **prim[(cfg["primary"], sym)]})
            row["yearly_base"] = mb["yearly_returns"]
            rows.append(row)
            print(
                f"{'OOS' if log else 'SMOKE'} {cfg['name']:<26} {sym} SR {row['sharpe_base']:+.2f}/"
                f"{row['sharpe_stress']:+.2f} (primary {row['prim_sharpe_base']:+.2f}) trades {row['n_trades']} "
                f"take {diag['take_rate']:.0%} auc {diag['auc_oos']:.3f} [{time.time() - t0:.0f}s]",
                flush=True,
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report


def _fmt_row(r: pd.Series) -> str:
    return (
        f"| {r['name']} | {r['tf']} | {r['symbol'][:3]} | {r['sharpe_base']:.2f} | {r['sharpe_stress']:.2f} | "
        f"{r['prim_sharpe_base']:.2f} / {r['prim_sharpe_stress']:.2f} | {r['cagr_base']:.1%} | "
        f"{r['maxdd_base']:.1%} | {r['n_trades']} | {r['trades_per_day']:.2f} | "
        f"{r['net_bps_base']:.1f} / {r['net_bps_stress']:.1f} | {r['pf_base']:.2f} | {r['pos_months']:.0%} | "
        f"{r['worst_year']:.1%} | {r['take_rate']:.0%} | {r['net_bps_taken']:.1f} / {r['net_bps_skipped']:.1f} | "
        f"{r['auc_oos']:.3f} |"
    )


def _pivot(df: pd.DataFrame) -> tuple[str, list[str]]:
    pv = df.pivot(index="name", columns="symbol", values=["sharpe_base", "prim_sharpe_base"])
    pv["min"] = pv["sharpe_base"].min(axis=1)
    pv = pv.sort_values("min", ascending=False)
    lines = [
        "| config | BTC Sharpe @5 | ETH Sharpe @5 | min | primary BTC / ETH |",
        "|---|---|---|---|---|",
    ]
    for n, r in pv.iterrows():
        lines.append(
            f"| {n} | {r[('sharpe_base', 'BTCUSDT')]:.2f} | {r[('sharpe_base', 'ETHUSDT')]:.2f} | "
            f"{r['min'].item():.2f} | "
            f"{r[('prim_sharpe_base', 'BTCUSDT')]:.2f} / {r[('prim_sharpe_base', 'ETHUSDT')]:.2f} |"
        )
    return "\n".join(lines), list(pv.index[:4])


def _yearly(df: pd.DataFrame, names: list[str]) -> str:
    years = sorted({y for d in df["yearly_base"] for y in d})
    head = "| config | sym | " + " | ".join(str(y) for y in years) + " |\n|---|---|" + "---|" * len(years)
    body = [
        f"| {r['name']} | {r['symbol'][:3]} | "
        + " | ".join(f"{r['yearly_base'].get(y, np.nan):.1%}" for y in years)
        + " |"
        for _, r in df[df["name"].isin(names)].iterrows()
    ]
    return head + "\n" + "\n".join(body)


def render(df: pd.DataFrame) -> str:
    head = (
        "| config | tf | sym | Sharpe @5 | Sharpe @10 | primary alone @5 / @10 | CAGR @5 | MaxDD @5 | trades | "
        "trades/day | net bps/trade @5 / @10 | PF @5 | +months | worst year @5 | take rate | "
        "event net bps taken / skipped | AUC |\n" + "|---" * 17 + "|"
    )
    body = "\n".join(_fmt_row(r) for _, r in df.sort_values("sharpe_base", ascending=False).iterrows())
    both, top = _pivot(df)
    logged = bool(df["prim_logged"].all())
    return (
        f"## Results (OOS 2020-01-01..2026-06-30, generated by the script, n_trials in registry after "
        f"this run = {trials.n_trials()})\n\nEvery run, sorted by base-cost Sharpe. 'primary alone' is the rule "
        f"on identical bars and window (returns hash already in the `rule_baselines` registry: {logged}). "
        f"'event net bps' is the mean labelled trade return (round trip at 5 bps/side) of OOS primary trades "
        f"the filter took vs skipped. AUC is of the OOS probabilities vs the OOS labels (diagnostic only).\n\n"
        f"{head}\n{body}\n\nBoth symbols side by side, sorted by the weaker symbol:\n\n{both}\n\n"
        f"Calendar-year returns @5 of the top 4 by the weaker symbol:\n\n{_yearly(df, top)}\n"
    )


def write_results(df: pd.DataFrame, path: Path = RESULTS_MD) -> None:
    text = path.read_text()
    block = f"<!-- RESULTS:BEGIN -->\n{render(df)}<!-- RESULTS:END -->"
    path.write_text(re.sub(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->", lambda _: block, text, flags=re.S))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run on 2019 training data; log nothing")
    ap.add_argument("--only", default="", help="smoke only: comma-separated config names")
    ap.add_argument("--csv", default="", help="optional path for the per-run table (not part of the repo)")
    args = ap.parse_args()
    if args.only and not args.smoke:
        raise SystemExit("--only is for --smoke: the logged run is the whole pre-registered grid")
    if args.only:
        GRID[:] = [GRID_BY_NAME[n] for n in args.only.split(",")]
    df = run_all(log=not args.smoke)
    if args.csv:
        df.to_pickle(args.csv)
    if not args.smoke:
        write_results(df)
    print("done | registry N =", trials.n_trials())


if __name__ == "__main__":
    main()
