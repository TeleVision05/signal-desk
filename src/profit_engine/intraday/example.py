"""End-to-end example: load -> features -> labels -> walk-forward fit -> positions -> backtest
-> metrics -> log trial -> promotion table.

    python -m profit_engine.intraday.example --trials-dir /tmp/demo_trials

``--trials-dir`` defaults to a throwaway directory so the demo never pollutes the real
registry: the demo points ``config.TRIALS_DIR`` at it, because ``promotion.evaluate`` always
reads ``config.TRIALS_DIR``. A real experiment never changes it (writes
research/intraday/trials/) and logs EVERY configuration it tries, including the ones that lose.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from profit_engine.intraday import backtest, config, data, features, labels, metrics, promotion, splits, trials
from profit_engine.intraday.provenance import data_hash
from profit_engine.intraday.resample import resample

SEED = 7
TIMEFRAME = "1h"
FAMILY = "logreg_demo"
PARAMS = {"C": 0.1, "horizon": 4, "threshold": 0.55}
FEATURE_PARAMS = {  # windows are in bars of TIMEFRAME: 24 = 1 day at 1h
    "log_returns": {"horizons": (1, 4, 24)},
    "realized_vol": {"windows": (24, 168)},
    "return_zscore": {"horizon": 4, "window": 168},
    "taker_imbalance": {"windows": (1, 24)},
    "vwap_deviation": {"windows": (24, 168)},
    "volume_ratio": {"short": 4, "long": 168},
}


def oos_positions(bars: pd.DataFrame, other: pd.DataFrame) -> pd.Series:
    """Walk-forward: refit monthly on purged past data, predict the next month only."""
    x = features.make_feature_frame(bars, other, params=FEATURE_PARAMS, cross_horizons=(1, 24))
    bars = bars.loc[x.index]
    lab = labels.forward_return(bars, PARAMS["horizon"])
    y = (lab["ret"] > 0).astype(int)
    has_x = x.notna().all(axis=1).to_numpy()
    has_y = lab["ret"].notna().to_numpy()
    target = pd.Series(0.0, index=bars.index)
    for fold in splits.walk_forward_splits(bars.index, lab["t1"]):
        train, test = fold.train & has_x & has_y, fold.test & has_x
        model = make_pipeline(StandardScaler(), LogisticRegression(C=PARAMS["C"], max_iter=1000, random_state=SEED))
        model.fit(x[train], y[train])
        target[test] = (model.predict_proba(x[test])[:, 1] > PARAMS["threshold"]).astype(float)
    return target


def run_symbol(sym: str, bars: dict[str, pd.DataFrame], raw_hash: str) -> tuple:
    other = bars["ETHUSDT" if sym == "BTCUSDT" else "BTCUSDT"]
    target = oos_positions(bars[sym], other)
    in_oos = (target.index >= config.EVAL_START) & (target.index < config.EVAL_END)
    oos_bars, oos_target = bars[sym].loc[target.index[in_oos]], target[in_oos]
    base = backtest.run_backtest(oos_bars, oos_target, config.BASE_COST_BPS)
    stress = backtest.run_backtest(oos_bars, oos_target, config.STRESS_COST_BPS)
    for res in (base, stress):
        m = metrics.summarize(res)
        trials.log_trial(FAMILY, PARAMS, TIMEFRAME, sym, res.cost_bps_per_side, m, res.daily, raw_hash, SEED)
    print(metrics.one_line(f"{sym} {FAMILY} @5bps", metrics.summarize(base)))
    return base, stress, oos_bars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials-dir", type=Path, default=None)
    args = ap.parse_args()
    trials_dir = args.trials_dir or Path(tempfile.mkdtemp(prefix="intraday_demo_"))
    config.TRIALS_DIR = trials_dir  # demo only: keep the demo's runs out of the real registry
    raw = {s: data.load_klines(s) for s in config.SYMBOLS}  # [TRAIN_START, HOLDOUT_START) only
    bars = {s: resample(df, TIMEFRAME, incomplete="drop") for s, df in raw.items()}
    out = {s: run_symbol(s, bars, data_hash(raw[s])) for s in config.SYMBOLS}
    cand = promotion.Candidate(
        FAMILY,
        PARAMS,
        TIMEFRAME,
        base={s: o[0] for s, o in out.items()},
        stress={s: o[1] for s, o in out.items()},
        bars={s: o[2] for s, o in out.items()},
    )
    table = promotion.evaluate(cand, n_perm=200, seed=SEED)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(table.to_string(index=False))
    print("VERDICT:", promotion.verdict(table), "| registry:", trials_dir)


if __name__ == "__main__":
    main()
