"""Regime-gated exposure (family ``regime_gated``, kind ``model``).

    .venv/bin/python experiments/intraday/regime_gated.py --smoke   # 2019 as eval window, logs nothing
    .venv/bin/python experiments/intraday/regime_gated.py           # the one logged OOS run

Two stages, refit every month inside ``walk_forward_splits`` training rows only:
(a) a HAR model forecasts next-24h realized volatility (and, for ``logit``, a logistic model the
    sign of the next H-bar return); (b) each bar is mapped to a discrete regime cell, and the book
    is long only while that cell's TRAINING mean net forward return is positive with t_adj >= 1,
    holding at least ``min_hold`` bars per entry.

The pre-registered grid (18 configs per symbol) is ``GRID`` and is mirrored at the top of
``research/intraday/results/regime_gated.md``.

Rebuild one config for ``promotion.evaluate`` (from the repo root; nothing is logged)::

    import sys; sys.path.insert(0, "experiments/intraday")
    from regime_gated import build_candidate
    cand = build_candidate("trend_4h_h6_exp")   # promotion.Candidate with base, stress, bars
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from profit_engine.intraday import backtest, config, data, features, labels, metrics, promotion, splits, trials
from profit_engine.intraday.provenance import data_hash
from profit_engine.intraday.resample import resample

FAMILY = "regime_gated"
KIND = "model"
SEED = 7
RESULTS_MD = config.RESEARCH_DIR / "results" / "regime_gated.md"
BARS_PER_DAY = {"1h": 24, "4h": 6}
HURDLE = 2 * config.BASE_COST_BPS / 1e4  # one round trip at base cost per label
T_MIN, MIN_EFF_N = 1.0, 30
VOL_Q, FLOW_Q, LOGIT_C = (0.4, 0.8), (0.1, 0.9), 0.1
SMOKE = (pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2020-01-01", tz="UTC"))
LOGIT_X = ["ret_d", "ret_w", "ret_m", "ema_dev", "vf", "rv_ratio", "tib_4h", "tib_d", "volratio", "vwapdev"]
N_CELLS = {"har_trend": 12, "har_flow": 18, "logit": 15}
VOL_NAMES, FLOW_NAMES = ("lo", "mid", "hi"), ("sell", "norm", "buy")


def _cfg(model: str, tf: str, h: int, hold: int, train: str) -> dict:
    tag = {"har_trend": "trend", "har_flow": "flow", "logit": "logit"}[model]
    name = f"{tag}_{tf}_h{h}_{'exp' if train == 'expanding' else 'roll24'}"
    params = {
        "model": model, "H": h, "min_hold": hold, "train": train, "train_months": 24 if train == "rolling" else None,
        "t_min": T_MIN, "min_eff_n": MIN_EFF_N, "hurdle_bps": HURDLE * 1e4, "vol_q": list(VOL_Q),
        "flow_q": list(FLOW_Q), "C": LOGIT_C if model == "logit" else None, "refit": "monthly",
    }  # fmt: skip
    return {"name": name, "model": model, "tf": tf, "params": params}


_HZ = {("1h", "short"): (4, 8), ("1h", "long"): (24, 24), ("4h", "short"): (1, 2), ("4h", "long"): (6, 6)}
GRID: list[dict] = [
    _cfg(m, tf, *_HZ[(tf, hz)], "expanding")
    for m in ("har_trend", "har_flow", "logit")
    for tf in ("1h", "4h")
    for hz in ("short", "long")
] + [_cfg(m, tf, *_HZ[(tf, "long")], "rolling") for m in ("har_trend", "har_flow", "logit") for tf in ("1h", "4h")]
GRID_BY_NAME = {c["name"]: c for c in GRID}


# --------------------------------------------------------------------------- features (causal)


def bar_rv2(raw: pd.DataFrame, tf: str, index: pd.DatetimeIndex) -> pd.Series:
    """Sum of squared 5m log close returns inside each bar (bar t uses 5m closes <= close_t)."""
    r2 = np.log(raw["close"]).diff().pow(2)
    rv2 = r2.groupby(raw.index.floor(tf)).sum(min_count=1)
    return rv2.reindex(index)


def build_frame(df: pd.DataFrame, rv2: pd.Series, tf: str, h: int) -> pd.DataFrame:
    """Every input known at bar t's close, plus the two labels and their combined end time t1."""
    d = BARS_PER_DAY[tf]
    w, m = 7 * d, 30 * d
    f = pd.DataFrame(index=df.index)

    def lvol(s: pd.Series) -> pd.Series:
        return np.log(np.sqrt(s.where(s > 0)))

    for k, n in (("har_d", d), ("har_w", w), ("har_m", m)):
        f[k] = lvol(rv2.rolling(n, min_periods=n).mean())
    f["y_vol"] = lvol(rv2.rolling(d, min_periods=d).mean().shift(-d))  # bars t+1..t+d
    lc = np.log(df["close"])
    ema = df["close"].ewm(span=m, adjust=False, min_periods=m).mean()
    f["ema_dev"] = np.log(df["close"] / ema)
    f["ret_d"], f["ret_w"], f["ret_m"] = lc.diff(d), lc.diff(w), lc.diff(m)
    f["trend_l"] = (f["ema_dev"] > 0).astype(float)
    f["trend_s"] = (f["ret_d"] > 0).astype(float)
    tib = features.taker_imbalance(df, windows=(d // 6, d))
    f["tib_4h"], f["tib_d"] = tib[f"tib_{d // 6}"], tib[f"tib_{d}"]
    f["rv_ratio"] = f["har_d"] - f["har_m"]
    f["volratio"] = features.volume_ratio(df, short=d, long=m).iloc[:, 0]
    f["vwapdev"] = features.vwap_deviation(df, windows=(d,)).iloc[:, 0]
    lab_r, lab_v = labels.forward_return(df, h), labels.forward_return(df, d)
    f["ret"] = lab_r["ret"]
    t1 = pd.concat([lab_r["t1"], lab_v["t1"]], axis=1)
    f["t1"] = t1.max(axis=1).where(t1.notna().all(axis=1))
    f["vf"] = np.nan  # filled per fold by the HAR fit
    return f


# --------------------------------------------------------------------------- stage (a): models


def fit_har(f: pd.DataFrame, tr: np.ndarray) -> np.ndarray:
    """OLS of next-24h log RV on log RV over the last day / week / month; forecast for every row."""
    x = np.column_stack([np.ones(len(f)), f[["har_d", "har_w", "har_m"]].to_numpy()])
    ok = tr & np.isfinite(f["y_vol"].to_numpy())
    beta, *_ = np.linalg.lstsq(x[ok], f["y_vol"].to_numpy()[ok], rcond=None)
    return x @ beta


def _bucket(v: np.ndarray, ref: np.ndarray, qs: tuple[float, ...]) -> np.ndarray:
    """0..len(qs) by the training rows' quantiles ``qs`` of ``ref``."""
    return np.digitize(v, np.quantile(ref, qs)).astype(int)


def _logit() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=LOGIT_C, max_iter=1000, random_state=SEED))


def logit_quintiles(f: pd.DataFrame, tr: np.ndarray, te: np.ndarray, fold: splits.Fold) -> np.ndarray:
    """Probability quintile per row (-1 outside train/test). Edges come from inner purged K-fold
    OOS probabilities on the training rows; the test rows use a model fit on all training rows."""
    x, y = f[LOGIT_X].to_numpy(), (f["ret"].to_numpy() > 0).astype(int)
    tr_idx = f.index[tr]
    inner = splits.purged_kfold(tr_idx, f["t1"][tr_idx], n_splits=5, within=fold)
    xt, yt = x[tr], y[tr]
    p_in = np.full(len(tr_idx), np.nan)
    for itr, iva in inner:
        p_in[iva] = _logit().fit(xt[itr], yt[itr]).predict_proba(xt[iva])[:, 1]
    edges = np.quantile(p_in[np.isfinite(p_in)], (0.2, 0.4, 0.6, 0.8))
    out = np.full(len(f), -1)
    out[tr] = np.where(np.isfinite(p_in), np.digitize(p_in, edges), -1)
    if te.any():
        out[te] = np.digitize(_logit().fit(xt, yt).predict_proba(x[te])[:, 1], edges)
    return out


def regime_codes(model: str, f: pd.DataFrame, tr: np.ndarray, te: np.ndarray, fold: splits.Fold) -> np.ndarray:
    """Cell code per row (-1 = no cell). Every edge is estimated on training rows only."""
    vf = f["vf"].to_numpy()
    vol = _bucket(vf, vf[tr], VOL_Q)
    tl = f["trend_l"].to_numpy().astype(int)
    if model == "har_trend":
        return tl * 6 + f["trend_s"].to_numpy().astype(int) * 3 + vol
    if model == "har_flow":
        tib = f["tib_4h"].to_numpy()
        return tl * 9 + _bucket(tib, tib[tr], FLOW_Q) * 3 + vol
    pq = logit_quintiles(f, tr, te, fold)
    return np.where(pq >= 0, pq * 3 + vol, -1)


def cell_label(model: str, code: int) -> str:
    v = VOL_NAMES[code % 3]
    if model == "har_trend":
        return f"L{'up' if code // 6 else 'dn'}_S{'up' if (code // 3) % 2 else 'dn'}_V{v}"
    if model == "har_flow":
        return f"L{'up' if code // 9 else 'dn'}_F{FLOW_NAMES[(code // 3) % 3]}_V{v}"
    return f"P{code // 3 + 1}_V{v}"


# --------------------------------------------------------------------------- stage (b): gate


def cell_gate(codes: np.ndarray, net: np.ndarray, h: int, n_cells: int) -> np.ndarray:
    """ON cells: n/H >= MIN_EFF_N, mean net > 0 and overlap-adjusted t >= T_MIN (training rows)."""
    ok = codes >= 0
    c, r = codes[ok], net[ok]
    n = np.bincount(c, minlength=n_cells).astype(float)
    s, ss = np.bincount(c, r, n_cells), np.bincount(c, r * r, n_cells)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s / n
        var = (ss - n * mean**2) / (n - 1)
        t = mean / np.sqrt(var / (n / h))
    return (n / h >= MIN_EFF_N) & (mean > 0) & (t >= T_MIN)


def hold(gate: np.ndarray, min_hold: int) -> np.ndarray:
    """Enter on the first ON bar; stay at least ``min_hold`` bars; then exit on the first OFF bar."""
    out = np.zeros(len(gate))
    held, age = False, 0
    for i, g in enumerate(gate):
        if held:
            age += 1
            if age >= min_hold and not g:
                held = False
        elif g:
            held, age = True, 0
        out[i] = 1.0 if held else 0.0
    return out


def oos_target(cfg: dict, f: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """Stitched walk-forward target over the frame (0 outside test blocks) and the test cell codes."""
    p = cfg["params"]
    window = pd.DateOffset(months=p["train_months"]) if p["train"] == "rolling" else None
    folds = splits.walk_forward_splits(
        f.index, f["t1"], train=p["train"], train_window=window, eval_start=lo, eval_end=hi
    )
    base_cols = [c for c in LOGIT_X if c != "vf"] + ["har_d", "har_w", "har_m"]
    has_x = f[base_cols].notna().all(axis=1).to_numpy()
    has_y = (f["ret"].notna() & f["y_vol"].notna() & f["t1"].notna()).to_numpy()
    net = f["ret"].to_numpy() - HURDLE
    gate, codes = np.zeros(len(f), dtype=bool), np.full(len(f), -1)
    for fold in folds:
        tr, te = fold.train & has_x & has_y, fold.test & has_x
        if (f["t1"].to_numpy()[tr] >= fold.test_start).any():
            raise RuntimeError(f"training label reaches test block {fold.test_start}")
        f["vf"] = fit_har(f, tr)
        c = regime_codes(cfg["model"], f, tr, te, fold)
        on = cell_gate(c[tr], net[tr], p["H"], N_CELLS[cfg["model"]])
        gate[te] = on[c[te]] & (c[te] >= 0)
        codes[te] = c[te]
    target = pd.Series(hold(gate, p["min_hold"]), index=f.index)
    return target, pd.Series(codes, index=f.index)


# --------------------------------------------------------------------------- data + runs


def load_inputs() -> tuple[dict, dict, dict]:
    """{tf: {sym: bars}}, {tf: {sym: rv2}}, and the raw 5m data hash per symbol."""
    raw = {s: data.load_klines(s) for s in config.SYMBOLS}
    bars, rv2 = {}, {}
    for tf in ("1h", "4h"):
        bars[tf] = {s: resample(raw[s], tf, incomplete="drop") for s in config.SYMBOLS}
        rv2[tf] = {s: bar_rv2(raw[s], tf, bars[tf][s].index) for s in config.SYMBOLS}
        for s in config.SYMBOLS:
            n_full = len(resample(raw[s], tf, incomplete="flag"))
            print(f"[bars] {s} {tf}: {len(bars[tf][s])} bars, dropped {n_full - len(bars[tf][s])} incomplete")
    return bars, rv2, {s: data_hash(raw[s]) for s in config.SYMBOLS}


def run_config(cfg: dict, df: pd.DataFrame, rv2: pd.Series, lo=config.EVAL_START, hi=config.EVAL_END) -> tuple:
    """Base and stress backtests of the stitched target over [lo, hi), the bars, codes and frame."""
    f = build_frame(df, rv2, cfg["tf"], cfg["params"]["H"])
    target, codes = oos_target(cfg, f, lo, hi)
    keep = (df.index >= lo) & (df.index < hi)
    b, t = df.loc[keep], target.loc[df.index[keep]]
    base = backtest.run_backtest(b, t, config.BASE_COST_BPS)
    stress = backtest.run_backtest(b, t, config.STRESS_COST_BPS)
    return base, stress, b, codes.loc[b.index], f.loc[b.index]


def build_candidate(name: str) -> promotion.Candidate:
    """Rebuild one grid config on both symbols as a ``promotion.Candidate``."""
    cfg = GRID_BY_NAME[name]
    raw = {s: data.load_klines(s) for s in config.SYMBOLS}
    out = {}
    for s in config.SYMBOLS:
        df = resample(raw[s], cfg["tf"], incomplete="drop")
        out[s] = run_config(cfg, df, bar_rv2(raw[s], cfg["tf"], df.index))
    return promotion.Candidate(
        FAMILY,
        cfg["params"],
        cfg["tf"],
        base={s: o[0] for s, o in out.items()},
        stress={s: o[1] for s, o in out.items()},
        bars={s: o[2] for s, o in out.items()},
    )


def attribution(model: str, base, codes: pd.Series) -> dict[str, float]:
    """Net log P&L of held bars, attributed to the cell of the bar that decided the position."""
    b = base.bars
    held = b["position"].to_numpy() > 0
    dec = codes.shift(1).fillna(-1).astype(int).to_numpy()
    pnl = np.log1p(b["net_ret"].to_numpy())
    s = pd.Series(pnl[held]).groupby(dec[held]).sum()
    return {("pre" if k < 0 else cell_label(model, int(k))): float(v) for k, v in s.items()}


def _row(cfg: dict, sym: str, base, stress, codes: pd.Series, f: pd.DataFrame) -> dict:
    mb, ms = metrics.summarize(base), metrics.summarize(stress)
    held = base.bars["position"].to_numpy() > 0
    trend_up = f["trend_l"].shift(1).to_numpy()[held]
    bh = np.log(base.bars["ret"] + 1).groupby(base.bars.index.floor("D")).sum()
    return {
        "name": cfg["name"], "model": cfg["model"], "tf": cfg["tf"], "symbol": sym,
        "sharpe_base": mb["sharpe"], "sharpe_stress": ms["sharpe"], "cagr_base": mb["cagr"],
        "maxdd_base": mb["max_drawdown"], "n_trades": mb["n_trades"], "trades_per_day": mb["trades_per_day"],
        "net_bps_base": mb["mean_net_bps"], "net_bps_stress": ms["mean_net_bps"], "pf_base": mb["profit_factor"],
        "pos_months": mb["pct_positive_months"], "worst_year": mb["worst_year"], "exposure": mb["exposure"],
        "t_daily": mb["t_daily"], "long_in_uptrend": float(np.nanmean(trend_up)) if held.any() else np.nan,
        "corr_bh": float(np.corrcoef(np.log1p(base.daily), bh.reindex(base.daily.index).fillna(0))[0, 1]),
        "years": mb["yearly_returns"], "attr": attribution(cfg["model"], base, codes),
    }, (mb, ms)  # fmt: skip


def run_all(log: bool, only: list[str] | None = None) -> pd.DataFrame:
    bars, rv2, hashes = load_inputs()
    lo, hi = (config.EVAL_START, config.EVAL_END) if log else SMOKE
    rows = []
    for cfg in [c for c in GRID if not only or c["name"] in only]:
        for sym in config.SYMBOLS:
            tf = cfg["tf"]
            base, stress, _, codes, f = run_config(cfg, bars[tf][sym], rv2[tf][sym], lo, hi)
            row, (mb, ms) = _row(cfg, sym, base, stress, codes, f)
            if log:
                for res, m in ((base, mb), (stress, ms)):
                    trials.log_trial(
                        FAMILY, cfg["params"], tf, sym, res.cost_bps_per_side, m, res.daily,
                        hashes[sym], SEED, kind=KIND, notes=cfg["name"],
                    )  # fmt: skip
            rows.append(row)
            print(
                f"{'OOS' if log else 'SMOKE'} {cfg['name']:<22} {sym} SR {row['sharpe_base']:+.2f}/"
                f"{row['sharpe_stress']:+.2f} trades {row['n_trades']} ({row['trades_per_day']:.2f}/d) "
                f"exp {row['exposure']:.0%} dd {row['maxdd_base']:.0%} wy {row['worst_year']:.0%}",
                flush=True,
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report


def _fmt_row(r: pd.Series) -> str:
    return (
        f"| {r['name']} | {r['tf']} | {r['symbol'][:3]} | {r['sharpe_base']:.2f} | {r['sharpe_stress']:.2f} | "
        f"{r['cagr_base']:.1%} | {r['maxdd_base']:.1%} | {r['n_trades']} | {r['trades_per_day']:.2f} | "
        f"{r['net_bps_base']:.1f} / {r['net_bps_stress']:.1f} | {r['pf_base']:.2f} | {r['pos_months']:.0%} | "
        f"{r['worst_year']:.1%} | {r['exposure']:.0%} | {r['long_in_uptrend']:.0%} | {r['corr_bh']:.2f} |"
    )


def _years_table(df: pd.DataFrame) -> str:
    yrs = sorted({y for d in df["years"] for y in d})
    head = "| config | sym | " + " | ".join(str(y) for y in yrs) + " |\n|---|---|" + "---|" * len(yrs)
    lines = [
        f"| {r['name']} | {r['symbol'][:3]} | " + " | ".join(f"{r['years'].get(y, np.nan):.0%}" for y in yrs) + " |"
        for _, r in df.sort_values("sharpe_base", ascending=False).iterrows()
    ]
    return head + "\n" + "\n".join(lines)


def render(df: pd.DataFrame) -> str:
    head = (
        "| config | tf | sym | Sharpe @5 | Sharpe @10 | CAGR @5 | MaxDD @5 | trades | trades/day | "
        "net bps/trade @5 / @10 | PF @5 | +months | worst year @5 | exposure | long bars in uptrend | corr B&H |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    body = "\n".join(_fmt_row(r) for _, r in df.sort_values("sharpe_base", ascending=False).iterrows())
    pivot = df.pivot(index="name", columns="symbol", values="sharpe_base")
    pivot["min"] = pivot.min(axis=1)
    pivot = pivot.sort_values("min", ascending=False)
    both = "| config | BTC Sharpe @5 | ETH Sharpe @5 | min |\n|---|---|---|---|\n" + "\n".join(
        f"| {n} | {r['BTCUSDT']:.2f} | {r['ETHUSDT']:.2f} | {r['min']:.2f} |" for n, r in pivot.iterrows()
    )
    attr = []
    for name in pivot.index[:3]:
        for _, r in df[df["name"] == name].iterrows():
            top = sorted(r["attr"].items(), key=lambda kv: -abs(kv[1]))[:6]
            attr.append(f"* {name} {r['symbol'][:3]}: " + ", ".join(f"{k} {v:+.2f}" for k, v in top))
    return (
        f"## Results (OOS 2020-01-01..2026-06-30, generated by the script, n_trials in registry after "
        f"this run = {trials.n_trials()})\n\nEvery run, sorted by base-cost Sharpe (`long bars in uptrend` = share "
        f"of held bars decided while close > EMA 30d; `corr B&H` = daily correlation with buy-and-hold):\n\n"
        f"{head}\n{body}\n\nBoth symbols side by side, sorted by the weaker symbol:\n\n{both}\n\n"
        f"Calendar-year net returns @5 bps:\n\n{_years_table(df)}\n\n"
        f"Net log P&L by deciding regime cell (top 6 by size) for the three best configs:\n\n"
        + "\n".join(attr) + "\n"
    )  # fmt: skip


def write_results(df: pd.DataFrame, path: Path = RESULTS_MD) -> None:
    text = path.read_text()
    block = f"<!-- RESULTS:BEGIN -->\n{render(df)}<!-- RESULTS:END -->"
    path.write_text(re.sub(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->", lambda _: block, text, flags=re.S))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="evaluate on 2019 (training data); log nothing")
    ap.add_argument("--only", nargs="*", help="restrict to these grid names (smoke debugging)")
    args = ap.parse_args()
    df = run_all(log=not args.smoke, only=args.only)
    if args.smoke:
        print(df[["name", "symbol", "sharpe_base", "n_trades", "trades_per_day", "exposure"]].to_string(index=False))
        return
    df.drop(columns=["years", "attr"]).to_csv(sys.stdout, index=False)
    write_results(df)
    print("wrote", RESULTS_MD, "| registry N =", trials.n_trials())


if __name__ == "__main__":
    main()
