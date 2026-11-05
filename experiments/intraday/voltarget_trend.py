"""Vol-targeted multi-lookback trend (family ``voltarget_trend``, kind ``model``).

    .venv/bin/python experiments/intraday/voltarget_trend.py --smoke   # 2018-07..2019 only, logs nothing
    .venv/bin/python experiments/intraday/voltarget_trend.py           # the one logged OOS run

The pre-registered grid (24 configs per symbol) is ``GRID`` below and is mirrored at the top of
``research/intraday/results/voltarget_trend.md``. Every config is backtested once on the stitched
OOS window [EVAL_START, EVAL_END) at 5 and 10 bps/side and logged with ``trials.log_trial``.

Rebuild one config for ``promotion.evaluate`` (from the repo root; nothing is logged)::

    import sys; sys.path.insert(0, "experiments/intraday")
    from voltarget_trend import build_candidate
    cand = build_candidate("vt_4h_slow4_vote_s50")   # promotion.Candidate with base, stress, bars
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from profit_engine.intraday import backtest, config, data, labels, metrics, promotion, splits, trials
from profit_engine.intraday.provenance import data_hash
from profit_engine.intraday.resample import resample

FAMILY = "voltarget_trend"
KIND = "model"
SEED = 0  # deterministic rules; logged for the registry
RESULTS_MD = config.RESEARCH_DIR / "results" / "voltarget_trend.md"
BARS_PER_DAY = {"1h": 24, "4h": 6}
LOOKBACKS = {"mix5": (1, 3, 7, 14, 30), "slow4": (7, 14, 30, 60)}
STEP = 0.05
SMOKE = (pd.Timestamp("2018-07-01", tz="UTC"), config.EVAL_START)


def _cfg(
    name: str,
    tf: str,
    lookbacks: str = "mix5",
    signal: str = "vote",
    sigma: float | None = 0.30,
    band: float = 0.2,
    hl: float = 3,
    regime: int | None = None,
) -> dict:
    params = {
        "rule": "voltarget_trend",
        "lookbacks_days": list(LOOKBACKS[lookbacks]),
        "signal": signal,
        "sigma_target": sigma,
        "band": band,
        "vol_halflife_days": hl,
        "regime_ema_days": regime,
        "step": STEP,
    }
    return {"name": name, "tf": tf, "params": params}


def _block_a() -> list[dict]:
    out = []
    for tf in ("1h", "4h"):
        for lb in ("mix5", "slow4"):
            for sig in ("vote", "z"):
                for s in (0.30, 0.50):
                    out.append(_cfg(f"vt_{tf}_{lb}_{sig}_s{int(s * 100)}", tf, lb, sig, s))
    return out


BLOCK_A = _block_a()
POOL_4H = [c["name"] for c in BLOCK_A if c["tf"] == "4h"]
ANCHOR = "vt_4h_mix5_vote_s30"
SELECT = {
    "name": "wf_4h_select",
    "tf": "4h",
    "params": {"rule": "wf_select", "pool": POOL_4H, "test_months": 3, "valid_frac": 0.5, "score_cost_bps": 5.0},
}
GRID: list[dict] = [
    *BLOCK_A,
    _cfg(f"{ANCHOR}_b10", "4h", band=0.1),
    _cfg(f"{ANCHOR}_b35", "4h", band=0.35),
    _cfg(f"{ANCHOR}_hl1", "4h", hl=1),
    _cfg(f"{ANCHOR}_hl10", "4h", hl=10),
    _cfg(f"{ANCHOR}_reg30", "4h", regime=30),
    _cfg(f"{ANCHOR}_reg90", "4h", regime=90),
    _cfg("vt_4h_mix5_vote_novol", "4h", sigma=None),
    SELECT,
]
GRID_BY_NAME = {c["name"]: c for c in GRID}
assert len(GRID) == 24 and len(GRID_BY_NAME) == 24


# --------------------------------------------------------------------------- rule (causal)


def ewma_sigma(df: pd.DataFrame, hl_bars: float) -> pd.Series:
    """Per-bar EWMA vol of 1-bar log close returns (zero mean), known at bar t's close."""
    r1 = np.log(df["close"]).diff()
    return np.sqrt((r1**2).ewm(halflife=hl_bars, min_periods=int(2 * hl_bars)).mean())


def trend_score(df: pd.DataFrame, bars: list[int], signal: str, sig: pd.Series) -> pd.Series:
    """Mean over lookbacks of a per-lookback [0, 1] signal; NaN until every lookback is warm."""
    lc = np.log(df["close"])
    parts = []
    for n in bars:
        r = lc.diff(n)
        if signal == "vote":
            s = (r > 0).astype(float).where(r.notna())
        elif signal == "z":
            s = (r / (sig.where(sig > 0) * np.sqrt(n))).clip(0.0, 1.0)
        else:
            raise ValueError(f"unknown signal {signal}")
        parts.append(s)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=False)


def hysteresis(raw: np.ndarray, band: float, step: float = STEP) -> np.ndarray:
    """Move the target to round(raw/step)*step only when |raw - target| > band; a raw that
    rounds to 0 exits at once; NaN keeps the current target (flat before the first value)."""
    out = np.zeros(len(raw))
    cur = 0.0
    for i, x in enumerate(raw):
        if np.isfinite(x):
            q = min(1.0, round(x / step) * step)
            if q <= 0.0:
                cur = 0.0
            elif abs(x - cur) > band:
                cur = q
        out[i] = cur
    return out


def raw_exposure(df: pd.DataFrame, tf: str, p: dict) -> pd.Series:
    bpd = BARS_PER_DAY[tf]
    sig = ewma_sigma(df, p["vol_halflife_days"] * bpd)
    score = trend_score(df, [d * bpd for d in p["lookbacks_days"]], p["signal"], sig)
    if p["sigma_target"] is None:
        scale = pd.Series(1.0, index=df.index).where(sig.notna())
    else:
        vol = sig.where(sig > 0) * np.sqrt(bpd * 365)
        scale = (p["sigma_target"] / vol).clip(upper=1.0)
    raw = score * scale
    if p["regime_ema_days"]:
        n = p["regime_ema_days"] * bpd
        ema = df["close"].ewm(span=n, adjust=False, min_periods=n).mean()
        raw = raw * (df["close"] > ema).astype(float).where(ema.notna())
    return raw


def vt_target(df: pd.DataFrame, tf: str, p: dict) -> pd.Series:
    raw = raw_exposure(df, tf, p).to_numpy(dtype="float64")
    return pd.Series(hysteresis(raw, p["band"]), index=df.index)


def wf_select(df: pd.DataFrame, p: dict, lo: pd.Timestamp, hi: pd.Timestamp) -> tuple[pd.Series, list]:
    """Quarterly walk-forward choice among ``p['pool']`` by validation-tail Sharpe of per-bar net
    returns on the fold's purged training rows only (``within=fold``)."""
    pool = {n: vt_target(df, "4h", GRID_BY_NAME[n]["params"]) for n in p["pool"]}
    full = df.loc[df.index < hi]
    net = {
        n: backtest.run_backtest(full, t.loc[full.index], p["score_cost_bps"]).bars["net_ret"] for n, t in pool.items()
    }
    t1 = labels.forward_return(df, 1)["t1"]
    folds = splits.walk_forward_splits(df.index, t1, test_months=p["test_months"], eval_start=lo, eval_end=hi)
    target, picks = pd.Series(np.nan, index=df.index), []
    for fold in folds:
        tr_idx = df.index[fold.train]
        _, valid = splits.purged_tail_split(tr_idx, t1.loc[tr_idx], valid_frac=p["valid_frac"], within=fold)
        rows = tr_idx[valid]
        score = {n: s.loc[rows].mean() / s.loc[rows].std() for n, s in net.items()}
        best = max(score, key=lambda n: (score[n], n))
        picks.append((str(fold.test_start.date()), best, rows[0], rows[-1]))
        target[fold.test] = pool[best][fold.test]
    return target, picks


def target_for(cfg: dict, df: pd.DataFrame, lo=config.EVAL_START, hi=config.EVAL_END) -> tuple[pd.Series, list]:
    if cfg["params"]["rule"] == "wf_select":
        return wf_select(df, cfg["params"], lo, hi)
    return vt_target(df, cfg["tf"], cfg["params"]), []


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


def run_config(cfg: dict, df: pd.DataFrame, lo=config.EVAL_START, hi=config.EVAL_END) -> tuple:
    """Base and stress backtests of the stitched target over [lo, hi), the bars used, and picks."""
    target, picks = target_for(cfg, df, lo, hi)
    keep = (df.index >= lo) & (df.index < hi)
    b, t = df.loc[keep], target.loc[df.index[keep]]
    base = backtest.run_backtest(b, t, config.BASE_COST_BPS)
    stress = backtest.run_backtest(b, t, config.STRESS_COST_BPS)
    return base, stress, b, picks


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


def _row(cfg: dict, sym: str, base, stress) -> tuple[dict, tuple]:
    mb, ms = metrics.summarize(base), metrics.summarize(stress)
    pos = base.bars["position"]
    row = {
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
        "exposure": mb["exposure"],
        "fills": int((pos.diff().fillna(pos) != 0).sum()),
        "cost_drag": float(base.bars["cost"].sum()),
        "t_daily": mb["t_daily"],
        "years": mb["yearly_returns"],
    }
    return row, (mb, ms)


def run_all(log: bool) -> tuple[pd.DataFrame, dict]:
    bars, hashes = load_bars()
    lo, hi = (config.EVAL_START, config.EVAL_END) if log else SMOKE
    rows, extra = [], {"picks": {}, "bench": {}}
    for cfg in GRID:
        for sym in config.SYMBOLS:
            base, stress, _b, picks = run_config(cfg, bars[cfg["tf"]][sym], lo, hi)
            row, (mb, ms) = _row(cfg, sym, base, stress)
            if log:
                for res, m in ((base, mb), (stress, ms)):
                    trials.log_trial(
                        FAMILY, cfg["params"], cfg["tf"], sym, res.cost_bps_per_side, m, res.daily,
                        hashes[sym], SEED, kind=KIND, notes=cfg["name"],
                    )  # fmt: skip
            if picks:
                extra["picks"][sym] = picks
            rows.append(row)
            print(
                f"{'OOS' if log else 'SMOKE'} {cfg['name']:<28} {sym} SR {row['sharpe_base']:+.2f}/"
                f"{row['sharpe_stress']:+.2f} DD {row['maxdd_base']:.0%} trades {row['n_trades']} "
                f"fills {row['fills']} exp {row['exposure']:.0%}",
                flush=True,
            )
    extra["bench"] = benchmarks(bars, lo, hi)
    return pd.DataFrame(rows), extra


def benchmarks(bars: dict, lo: pd.Timestamp, hi: pd.Timestamp) -> dict:
    """Buy-and-hold per symbol on the 4h and 1h OOS bars (context only; not logged)."""
    out = {}
    for tf in ("4h", "1h"):
        for sym in config.SYMBOLS:
            df = bars[tf][sym]
            b = df.loc[(df.index >= lo) & (df.index < hi)]
            out[(tf, sym)] = metrics.summarize(backtest.buy_and_hold(b, config.BASE_COST_BPS))
    return out


# --------------------------------------------------------------------------- report


def _fmt_row(r: pd.Series) -> str:
    return (
        f"| {r['name']} | {r['tf']} | {r['symbol'][:3]} | {r['sharpe_base']:.2f} | {r['sharpe_stress']:.2f} | "
        f"{r['cagr_base']:.1%} | {r['maxdd_base']:.1%} | {r['n_trades']} | {r['trades_per_day']:.2f} | "
        f"{r['net_bps_base']:.1f} / {r['net_bps_stress']:.1f} | {r['pf_base']:.2f} | {r['pos_months']:.0%} | "
        f"{r['worst_year']:.1%} | {r['exposure']:.0%} | {r['fills']} | {r['cost_drag']:.1%} |"
    )


def _years_table(df: pd.DataFrame, names: list[str]) -> str:
    yrs = sorted({y for d in df["years"] for y in d})
    head = "| config | sym | " + " | ".join(str(y) for y in yrs) + " |\n|---|---|" + "---|" * len(yrs)
    lines = []
    for n in names:
        for _, r in df[df["name"] == n].iterrows():
            lines.append(
                f"| {n} | {r['symbol'][:3]} | " + " | ".join(f"{r['years'].get(y, np.nan):+.1%}" for y in yrs) + " |"
            )
    return head + "\n" + "\n".join(lines)


def _bench_lines(bench: dict) -> str:
    return "\n".join(
        f"* buy-and-hold {tf} {sym[:3]} @5: Sharpe {m['sharpe']:.2f}, CAGR {m['cagr']:.1%}, MaxDD "
        f"{m['max_drawdown']:.1%}, worst year {m['worst_year']:.1%}, +months {m['pct_positive_months']:.0%}"
        for (tf, sym), m in bench.items()
    )


def _picks_lines(picks: dict) -> str:
    out = []
    for sym, ps in picks.items():
        counts = pd.Series([p[1] for p in ps]).value_counts()
        seq = ", ".join(f"{d}: {n.replace('vt_4h_', '')}" for d, n, _, _ in ps)
        tally = ", ".join(f"{n}: {int(c)}" for n, c in counts.items())
        out.append(f"* {sym[:3]} picks per quarter: {seq}\n  (counts: {tally})")
    return "\n".join(out)


def render(df: pd.DataFrame, extra: dict) -> str:
    head = (
        "| config | tf | sym | Sharpe @5 | Sharpe @10 | CAGR @5 | MaxDD @5 | trades | trades/day | "
        "net bps/trade @5 / @10 | PF @5 | +months | worst year @5 | exposure | fills | cost drag @5 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    body = "\n".join(_fmt_row(r) for _, r in df.sort_values("sharpe_base", ascending=False).iterrows())
    pv = df.pivot(index="name", columns="symbol", values=["sharpe_base", "sharpe_stress", "maxdd_base"])
    pv["min"] = pv["sharpe_base"].min(axis=1)
    pv = pv.sort_values("min", ascending=False)
    both = (
        "| config | BTC Sharpe @5 / @10 | ETH Sharpe @5 / @10 | BTC MaxDD | ETH MaxDD | min Sharpe @5 |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(
            f"| {n} | {r[('sharpe_base', 'BTCUSDT')]:.2f} / {r[('sharpe_stress', 'BTCUSDT')]:.2f} | "
            f"{r[('sharpe_base', 'ETHUSDT')]:.2f} / {r[('sharpe_stress', 'ETHUSDT')]:.2f} | "
            f"{r[('maxdd_base', 'BTCUSDT')]:.1%} | {r[('maxdd_base', 'ETHUSDT')]:.1%} | {r['min'].iloc[0]:.2f} |"
            for n, r in pv.iterrows()
        )
    )
    top = list(pv.index[:4]) + [n for n in (ANCHOR, "vt_4h_mix5_vote_novol") if n not in pv.index[:4]]
    return (
        f"## Results (OOS 2020-01-01..2026-06-30, generated by the script, n_trials in registry after "
        f"this run = {trials.n_trials()})\n\nEvery run, sorted by base-cost Sharpe (cost drag = sum of "
        f"per-fill costs as a fraction of equity):\n\n{head}\n{body}\n\n"
        f"Both symbols side by side, sorted by the weaker symbol:\n\n{both}\n\n"
        f"Calendar-year net returns @5 bps (top 4 by min Sharpe, plus the anchor and the no-vol ablation):\n\n"
        f"{_years_table(df, top)}\n\nBenchmarks (context, not logged):\n\n{_bench_lines(extra['bench'])}\n\n"
        f"`wf_4h_select` choices:\n\n{_picks_lines(extra['picks'])}\n"
    )


def write_results(df: pd.DataFrame, extra: dict, path: Path = RESULTS_MD) -> None:
    text = path.read_text()
    block = f"<!-- RESULTS:BEGIN -->\n{render(df, extra)}<!-- RESULTS:END -->"
    path.write_text(re.sub(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->", lambda _: block, text, flags=re.S))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run on 2018-07..2019 training data; log nothing")
    args = ap.parse_args()
    df, extra = run_all(log=not args.smoke)
    if args.smoke:
        print(
            df[["name", "symbol", "sharpe_base", "maxdd_base", "n_trades", "fills", "exposure"]].to_string(index=False)
        )
        print(_picks_lines(extra["picks"]))
        return
    df.drop(columns="years").to_csv(sys.stdout, index=False)
    write_results(df, extra)
    print("wrote", RESULTS_MD, "| registry N =", trials.n_trials())


if __name__ == "__main__":
    main()
