"""L2 logistic regression family (``logreg``, kind ``model``) for the intraday harness.

    .venv/bin/python experiments/intraday/logreg.py --smoke            # 2019 only, logs nothing
    .venv/bin/python experiments/intraday/logreg.py                    # the one logged OOS run

The pre-registered grid (24 configs per symbol) is ``GRID`` below and is mirrored at the top of
``research/intraday/results/logreg.md``. Per monthly fold (rolling 24-month train), ``C`` and the
entry threshold are chosen by purged inner K-fold on the fold's training rows (``within=fold``),
maximizing pooled net bps per would-be trade at 5 bps/side. The stitched OOS target is backtested
once over [EVAL_START, EVAL_END) at 5 and 10 bps/side and both runs are logged.

Rebuild one config for ``promotion.evaluate`` (from the repo root; nothing is logged)::

    import sys; sys.path.insert(0, "experiments/intraday")
    from logreg import build_candidate
    cand = build_candidate("lr_4h_tb_h42_hi")   # promotion.Candidate with base, stress, bars
"""

from __future__ import annotations

import argparse
import re
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from profit_engine.intraday import backtest, config, data, features, labels, metrics, promotion, splits, trials
from profit_engine.intraday.provenance import data_hash
from profit_engine.intraday.resample import resample

FAMILY = "logreg"
SEED = 7
RESULTS_MD = config.RESEARCH_DIR / "results" / "logreg.md"
BPD = {"15m": 96, "1h": 24, "4h": 6}  # bars per day
HORIZONS = {"15m": (4, 16), "1h": (4, 24), "4h": (6, 42)}
C_GRID = (0.0005, 0.005, 0.05)
THR_GRID = (0.50, 0.51, 0.52, 0.54, 0.56, 0.58, 0.60)
INNER_SPLITS = 4
MIN_TRADES = 20
TRAIN_MONTHS = 24
COST_CTRL = {
    "lo": {"band": 0.02, "hold_div": 4, "max_rt": {"15m": 6, "1h": 4, "4h": 3}},
    "hi": {"band": 0.05, "hold_div": 1, "max_rt": {"15m": 2, "1h": 2, "4h": 1}},
}
OTHER = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT"}


def _cfg(tf: str, h: int, label: str, cc: str) -> dict:
    ctl = COST_CTRL[cc]
    params = {
        "label": label,
        "h": h,
        "band": ctl["band"],
        "min_hold": max(1, h // ctl["hold_div"]),
        "max_rt": ctl["max_rt"][tf],
        "train": f"rolling{TRAIN_MONTHS}m",
        "C_grid": list(C_GRID),
        "thr_grid": list(THR_GRID),
        "inner": f"purged_kfold{INNER_SPLITS}",
        "objective": "net_bps_per_trade@5",
        "min_trades": MIN_TRADES,
    }
    return {"name": f"lr_{tf}_{label}_h{h}_{cc}", "tf": tf, "params": params}


GRID: list[dict] = [
    _cfg(tf, h, lab, cc) for tf in BPD for h in HORIZONS[tf] for lab in ("fwd", "tb") for cc in ("lo", "hi")
]
GRID_BY_NAME = {c["name"]: c for c in GRID}


# --------------------------------------------------------------------------- features + labels


def feature_params(tf: str) -> dict:
    d = BPD[tf]
    return {
        "log_returns": {"horizons": (1, 4, d, 7 * d)},
        "realized_vol": {"windows": (d, 7 * d)},
        "return_zscore": {"horizon": 4, "window": 30 * d},
        "taker_imbalance": {"windows": (1, d)},
        "vwap_deviation": {"windows": (d, 7 * d)},
        "volume_ratio": {"short": max(2, d // 4), "long": 7 * d},
    }


def trend_features(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Causal trend/state features (value at t uses bars <= t only)."""
    d, c = BPD[tf], df["close"]
    lc = np.log(c)
    out = {f"pxema_{k}d": lc - np.log(c.ewm(span=k * d, adjust=False, min_periods=k * d).mean()) for k in (1, 7, 30)}
    out["ret_30d"] = lc.diff(30 * d)
    out["dd_30d"] = lc - np.log(c.rolling(30 * d, min_periods=30 * d).max())
    r1 = lc.diff()
    out["rv_ratio"] = np.log(r1.rolling(d, min_periods=d).std() / r1.rolling(30 * d, min_periods=30 * d).std())
    return pd.DataFrame(out, index=df.index)


def build_xy(bars: pd.DataFrame, other: pd.DataFrame, cfg: dict) -> tuple:
    """Aligned bars, features, binary label, label end ``t1`` and usable-row masks."""
    tf, prm = cfg["tf"], cfg["params"]
    d = BPD[tf]
    p, o = features.align(bars, other)
    x = pd.concat(
        [
            features.make_feature_frame(p, o, params=feature_params(tf), cross_horizons=(1, d, 7 * d)),
            trend_features(p, tf),
        ],
        axis=1,
    )
    if prm["label"] == "fwd":
        lab = labels.forward_return(p, prm["h"])
    else:
        lab = labels.triple_barrier(p, prm["h"], 1.0, 1.0, vol_window=7 * d)
    y = (lab["ret"] > 0).astype(int)
    has_x = np.isfinite(x.to_numpy(dtype="float64")).all(axis=1)
    has_y = (lab["ret"].notna() & lab["t1"].notna()).to_numpy()
    return p, x, y, lab["t1"], has_x, has_y


# --------------------------------------------------------------------------- model + selection


def _model(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=500, random_state=SEED))


def hysteresis(p: np.ndarray, enter: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    """Long when p > enter; stay long while p >= exit; NaN p is flat."""
    out = np.zeros(len(p))
    held = False
    for i in range(len(p)):
        held = bool(p[i] >= exit_[i]) if held else bool(p[i] > enter[i])
        out[i] = 1.0 if held else 0.0
    return out


def _score(vbars: pd.DataFrame, prob: np.ndarray, thr: float, prm: dict) -> tuple[float, int]:
    """Sum of net bps and trade count of the would-be trades on one inner validation block."""
    n = len(prob)
    tgt = hysteresis(prob, np.full(n, thr), np.full(n, thr - prm["band"]))
    if not tgt.any():
        return 0.0, 0
    res = backtest.run_backtest(
        vbars,
        pd.Series(tgt, index=vbars.index),
        config.BASE_COST_BPS,
        max_round_trips_per_day=prm["max_rt"],
        min_hold_bars=prm["min_hold"],
    )
    return float(res.trades["net_bps"].sum()), len(res.trades)


def select(p: pd.DataFrame, xa: np.ndarray, ya: np.ndarray, t1: pd.Series, train: np.ndarray, fold, prm: dict):
    """(net bps/trade, C, thr) maximizing pooled inner-validation net bps per trade, or None (flat)."""
    tr_pos = np.flatnonzero(train)
    inner = splits.purged_kfold(p.index[tr_pos], t1.iloc[tr_pos], n_splits=INNER_SPLITS, within=fold)
    stats = {(c, thr): [0.0, 0] for c in C_GRID for thr in THR_GRID}
    for itr, iva in inner:
        a, b = tr_pos[itr], tr_pos[iva]
        vbars = p.iloc[b]
        for c in C_GRID:
            prob = _model(c).fit(xa[a], ya[a]).predict_proba(xa[b])[:, 1]
            for thr in THR_GRID:
                s, n = _score(vbars, prob, thr, prm)
                stats[(c, thr)][0] += s
                stats[(c, thr)][1] += n
    best = None
    for (c, thr), (s, n) in stats.items():
        if n >= MIN_TRADES and s / n > 0 and (best is None or s / n > best[0]):
            best = (s / n, c, thr)
    return best


def run_folds(p, x, y, t1, has_x, has_y, prm: dict, lo: pd.Timestamp, hi: pd.Timestamp) -> dict:
    """Walk-forward OOS probabilities and per-bar entry/exit thresholds (inf = flat month)."""
    folds = splits.walk_forward_splits(
        p.index, t1, train="rolling", train_window=pd.DateOffset(months=TRAIN_MONTHS), eval_start=lo, eval_end=hi
    )
    xa, ya = x.to_numpy(dtype="float64"), y.to_numpy()
    n = len(p)
    prob, enter, exit_ = np.full(n, np.nan), np.full(n, np.inf), np.full(n, np.inf)
    picks, coefs = [], []
    for fold in folds:
        train, test = fold.train & has_x & has_y, fold.test & has_x
        if (t1.to_numpy()[train] >= fold.test_start).any():
            raise RuntimeError(f"training label reaches test block {fold.test_start}")
        best = select(p, xa, ya, t1, train, fold, prm)
        picks.append({"month": fold.test_start, "pick": best})
        if best is None:
            continue
        _, c, thr = best
        model = _model(c).fit(xa[train], ya[train])
        prob[test] = model.predict_proba(xa[test])[:, 1]
        enter[fold.test], exit_[fold.test] = thr, thr - prm["band"]
        coefs.append(model[-1].coef_[0])
    return {"prob": prob, "enter": enter, "exit": exit_, "picks": picks, "coefs": coefs, "n_folds": len(folds)}


# --------------------------------------------------------------------------- one run


def _diagnostics(cfg: dict, fit: dict, cols: list[str], base, p: pd.DataFrame) -> dict:
    picks = [q["pick"] for q in fit["picks"]]
    chosen = [q for q in picks if q is not None]
    coef = np.mean(fit["coefs"], axis=0) if fit["coefs"] else np.zeros(len(cols))
    top = [f"{cols[i]}{'+' if coef[i] > 0 else '-'}" for i in np.argsort(-np.abs(coef))[:3]]
    span = 30 * BPD[cfg["tf"]]
    ema = p["close"].ewm(span=span, adjust=False, min_periods=span).mean()
    trend = (p["close"] > ema).shift(1, fill_value=False).reindex(base.bars.index)  # state when decided
    in_trend = trend.to_numpy(dtype=bool)
    gross = base.bars["gross_ret"].to_numpy()
    tot = gross.sum()
    return {
        "flat_months": 1 - len(chosen) / max(len(picks), 1),
        "med_thr": float(np.median([q[2] for q in chosen])) if chosen else float("nan"),
        "mode_C": float(pd.Series([q[1] for q in chosen]).mode().iloc[0]) if chosen else float("nan"),
        "top_coef": " ".join(top),
        "trend_pnl_share": float(gross[in_trend].sum() / tot) if tot != 0 else float("nan"),
        "in_trend_exposure": float(base.bars["position"].to_numpy()[in_trend].mean()),
        "gross_sharpe": metrics.sharpe(base.daily_gross),
    }


def run_one(cfg: dict, sym: str, bars_tf: dict, lo: pd.Timestamp, hi: pd.Timestamp, full: bool = False):
    """Fit, stitch, backtest at both costs. Returns metrics (or full results with ``full``)."""
    t0, prm = time.time(), cfg["params"]
    p, x, y, t1, has_x, has_y = build_xy(bars_tf[sym], bars_tf[OTHER[sym]], cfg)
    fit = run_folds(p, x, y, t1, has_x, has_y, prm, lo, hi)
    target = pd.Series(hysteresis(fit["prob"], fit["enter"], fit["exit"]), index=p.index)
    keep = (p.index >= lo) & (p.index < hi)
    ob, ot = p.loc[keep], target.loc[keep]
    kw = {"max_round_trips_per_day": prm["max_rt"], "min_hold_bars": prm["min_hold"]}
    base = backtest.run_backtest(ob, ot, config.BASE_COST_BPS, **kw)
    stress = backtest.run_backtest(ob, ot, config.STRESS_COST_BPS, **kw)
    if full:
        return base, stress, ob
    diag = _diagnostics(cfg, fit, list(x.columns), base, p)
    diag["minutes"] = (time.time() - t0) / 60
    return {
        "name": cfg["name"],
        "sym": sym,
        "mb": metrics.summarize(base),
        "ms": metrics.summarize(stress),
        "daily_b": base.daily,
        "daily_s": stress.daily,
        "diag": diag,
    }


def load_bars(tfs: set[str]) -> tuple[dict, dict]:
    raw = {s: data.load_klines(s) for s in config.SYMBOLS}
    bars = {tf: {s: resample(raw[s], tf, incomplete="drop") for s in config.SYMBOLS} for tf in sorted(tfs)}
    return bars, {s: data_hash(raw[s]) for s in config.SYMBOLS}


def build_candidate(name: str) -> promotion.Candidate:
    """Rebuild one grid config on both symbols as a ``promotion.Candidate`` (nothing logged)."""
    cfg = GRID_BY_NAME[name]
    bars, _ = load_bars({cfg["tf"]})
    out = {s: run_one(cfg, s, bars[cfg["tf"]], config.EVAL_START, config.EVAL_END, full=True) for s in config.SYMBOLS}
    return promotion.Candidate(
        FAMILY,
        cfg["params"],
        cfg["tf"],
        base={s: o[0] for s, o in out.items()},
        stress={s: o[1] for s, o in out.items()},
        bars={s: o[2] for s, o in out.items()},
    )


# --------------------------------------------------------------------------- driver + report


def _row(r: dict) -> dict:
    mb, ms, dg = r["mb"], r["ms"], r["diag"]
    cfg = GRID_BY_NAME[r["name"]]
    return {
        "name": r["name"],
        "tf": cfg["tf"],
        "symbol": r["sym"],
        "sharpe_base": metrics.sharpe(r["daily_b"]),
        "sharpe_stress": metrics.sharpe(r["daily_s"]),
        "cagr_base": mb["cagr"],
        "maxdd_base": mb["max_drawdown"],
        "n_trades": mb["n_trades"],
        "trades_per_day": mb["trades_per_day"],
        "net_bps_base": mb["mean_net_bps"],
        "net_bps_stress": ms["mean_net_bps"],
        "pf_base": mb["profit_factor"],
        "pos_months": mb["pct_positive_months"],
        "worst_year": mb["worst_year"],
        "exposure": mb["exposure"],
        "t_daily": mb["t_daily"],
        "yearly": mb["yearly_returns"],
        **dg,
    }


def run_all(log: bool, lo: pd.Timestamp, hi: pd.Timestamp, names: list[str] | None, n_jobs: int) -> pd.DataFrame:
    grid = [c for c in GRID if names is None or c["name"] in names]
    bars, hashes = load_bars({c["tf"] for c in grid})
    jobs = [(cfg, sym) for cfg in grid for sym in config.SYMBOLS]
    out = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_one)(cfg, sym, bars[cfg["tf"]], lo, hi) for cfg, sym in jobs
    )  # fmt: skip
    rows = []
    for (cfg, sym), r in zip(jobs, out, strict=True):  # log in GRID order, from one process
        if log:
            for daily, m, cost in (
                (r["daily_b"], r["mb"], config.BASE_COST_BPS),
                (r["daily_s"], r["ms"], config.STRESS_COST_BPS),
            ):
                trials.log_trial(
                    FAMILY, cfg["params"], cfg["tf"], sym, cost, m, daily, hashes[sym], SEED, notes=cfg["name"]
                )
        row = _row(r)
        rows.append(row)
        print(
            f"{'OOS' if log else 'SMOKE'} {cfg['name']:<20} {sym} "
            f"SR {row['sharpe_base']:+.2f}/{row['sharpe_stress']:+.2f} "
            f"trades {row['n_trades']} exp {row['exposure']:.0%} flat {row['flat_months']:.0%} "
            f"{row['minutes']:.1f} min",
            flush=True,
        )
    return pd.DataFrame(rows)


def _fmt_row(r: pd.Series) -> str:
    return (
        f"| {r['name']} | {r['symbol'][:3]} | {r['sharpe_base']:.2f} | {r['sharpe_stress']:.2f} | "
        f"{r['gross_sharpe']:.2f} | {r['cagr_base']:.1%} | {r['maxdd_base']:.1%} | {r['n_trades']} | "
        f"{r['trades_per_day']:.2f} | {r['net_bps_base']:.1f} / {r['net_bps_stress']:.1f} | {r['pf_base']:.2f} | "
        f"{r['pos_months']:.0%} | {r['worst_year']:.1%} | {r['exposure']:.0%} | {r['flat_months']:.0%} | "
        f"{r['med_thr']:.2f} | {r['trend_pnl_share']:.0%} | {r['top_coef']} |"
    )


def render(df: pd.DataFrame) -> str:
    head = (
        "| config | sym | Sharpe @5 | Sharpe @10 | gross Sharpe | CAGR @5 | MaxDD @5 | trades | trades/day | "
        "net bps/trade @5 / @10 | PF @5 | +months | worst year @5 | exposure | flat months | median thr | "
        "gross P&L in trend | top |coef| |\n" + "|---" * 18 + "|"
    )
    body = "\n".join(_fmt_row(r) for _, r in df.sort_values("sharpe_base", ascending=False).iterrows())
    pv = df.pivot(index="name", columns="symbol", values="sharpe_base")
    pv["min"] = pv.min(axis=1)
    pv = pv.sort_values("min", ascending=False)
    both = "| config | BTC Sharpe @5 | ETH Sharpe @5 | min |\n|---|---|---|---|\n" + "\n".join(
        f"| {n} | {r['BTCUSDT']:.2f} | {r['ETHUSDT']:.2f} | {r['min']:.2f} |" for n, r in pv.iterrows()
    )
    top = df[df["name"].isin(pv.index[:4])].sort_values(["name", "symbol"])
    years = sorted({y for d in top["yearly"] for y in d})
    yr = (
        "| config | sym | " + " | ".join(map(str, years)) + " |\n" + "|---" * (len(years) + 2) + "|\n"
        + "\n".join(
            f"| {r['name']} | {r['symbol'][:3]} | " + " | ".join(f"{r['yearly'].get(y, 0):.1%}" for y in years) + " |"
            for _, r in top.iterrows()
        )
    )  # fmt: skip
    return (
        f"## Results (OOS 2020-01-01..2026-06-30, generated by the script, n_trials in registry after "
        f"this run = {trials.n_trials()})\n\nEvery run, sorted by base-cost Sharpe. `gross Sharpe` is the same "
        "run before costs; `flat months` = share of OOS months where inner CV found no pair with positive net "
        "bps/trade; `gross P&L in trend` = share of gross P&L earned while close > EMA(30 days) at decision "
        f"time; `top |coef|` = largest mean standardized coefficients across folds.\n\n{head}\n{body}\n\n"
        f"Both symbols side by side, sorted by the weaker symbol:\n\n{both}\n\n"
        f"Calendar-year net returns @5 bps of the four best configs (by weaker symbol):\n\n{yr}\n"
    )


def write_results(df: pd.DataFrame) -> None:
    text = RESULTS_MD.read_text()
    block = f"<!-- RESULTS:BEGIN -->\n{render(df)}<!-- RESULTS:END -->"
    RESULTS_MD.write_text(re.sub(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->", lambda _: block, text, flags=re.S))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2019-01..2019-04 on training-era data; log nothing")
    ap.add_argument("--only", nargs="*", default=None, help="smoke only: subset of config names")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if args.smoke:
        lo, hi = pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2019-04-01", tz="UTC")
        df = run_all(False, lo, hi, args.only, args.jobs)
        print(df[["name", "symbol", "sharpe_base", "n_trades", "exposure", "flat_months", "minutes"]].to_string())
        return
    df = run_all(True, config.EVAL_START, config.EVAL_END, None, args.jobs)
    print(df.drop(columns=["yearly"]).to_csv(index=False))
    write_results(df)
    print("wrote", RESULTS_MD, "| registry N =", trials.n_trials())


if __name__ == "__main__":
    main()
