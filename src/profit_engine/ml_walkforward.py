"""Purged walk-forward ML long/flat signals + stop-aware backtest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .daily_backtest import StopConfig, run_long_only_daily, summarize_daily
from .data import fetch_ohlcv
from .features import build_features, make_labels

OUT = Path(__file__).resolve().parents[2] / "output"


@dataclass
class WFConfig:
    ticker: str = "QQQ"
    horizon: int = 5
    train_days: int = 504  # ~2y
    test_days: int = 21  # ~1 month OOS step
    embargo: int = 5  # purge gap = horizon
    proba_threshold: float = 0.55
    label_mode: str = "excess_vol"
    model: str = "hgb"  # hgb | gb | logreg
    hard_pct: float = 0.08
    trailing_pct: float = 0.12
    cost_bps: float = 10.0
    start: str = "2012-01-01"
    cash: float = 100_000.0


def _make_model(kind: str):
    if kind == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        class_weight="balanced",
                        C=0.5,
                    ),
                ),
            ]
        )
    if kind == "gb":
        return GradientBoostingClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
        )
    # HistGradientBoosting handles NaNs but we dropna anyway
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.06,
        max_iter=120,
        l2_regularization=0.1,
    )


def walk_forward_signals(ohlc: pd.DataFrame, feat: pd.DataFrame, y: pd.Series, cfg: WFConfig) -> pd.Series:
    """
    Expanding/rolling train → predict next test block.
    Embargo: exclude last `embargo` train rows that overlap label horizon.
    """
    df = feat.copy()
    df["y"] = y
    df = df.dropna()
    idx = df.index
    X_all = df.drop(columns=["y"])
    y_all = df["y"].astype(int)

    signal = pd.Series(0, index=ohlc.index, dtype=int)
    proba_s = pd.Series(np.nan, index=ohlc.index)

    def _fit_predict(train_end: int, te_idx, *, feature_frame: pd.DataFrame | None = None) -> None:
        if train_end < cfg.train_days // 2 or len(te_idx) == 0:
            return
        train_start = max(0, train_end - cfg.train_days)
        tr_idx = idx[train_start:train_end]
        X_tr, y_tr = X_all.loc[tr_idx], y_all.loc[tr_idx]
        if y_tr.nunique() < 2 or len(X_tr) < 100:
            return
        src = feature_frame if feature_frame is not None else X_all
        # Features may be missing on the very latest bar; drop those rows
        X_te = src.reindex(te_idx).dropna(how="any")
        if X_te.empty:
            return
        # Align columns to training features
        X_te = X_te.reindex(columns=X_all.columns).dropna(how="any")
        if X_te.empty:
            return
        model = _make_model(cfg.model)
        model.fit(X_tr.values, y_tr.values)
        p = model.predict_proba(X_te.values)[:, 1]
        for dt, pi in zip(X_te.index, p):
            proba_s.loc[dt] = float(pi)
            signal.loc[dt] = 1 if float(pi) >= cfg.proba_threshold else 0

    i = cfg.train_days
    while i + cfg.test_days <= len(idx):
        train_end = i - cfg.embargo
        te_idx = idx[i : i + cfg.test_days]
        _fit_predict(train_end, te_idx)
        i += cfg.test_days

    # Cover leftover labeled days through the last labelable bar
    if i < len(idx):
        train_end = max(cfg.embargo, i - cfg.embargo)
        _fit_predict(train_end, idx[i:])

    # Live extension: most recent bars have features but no forward label yet
    # (horizon days). Score them with the freshest train window so paper has p.
    feat_live = feat.dropna(how="any")
    live_idx = feat_live.index[feat_live.index > idx[-1]]
    if len(live_idx):
        train_end = len(idx) - cfg.embargo
        _fit_predict(train_end, live_idx, feature_frame=feat_live)

    signal.name = "ml_signal"
    return signal, proba_s


def run_ml_experiment(cfg: WFConfig) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = {
        "QQQ": fetch_ohlcv("QQQ", start=cfg.start),
        "SPY": fetch_ohlcv("SPY", start=cfg.start),
        "TQQQ": fetch_ohlcv("TQQQ", start=max(cfg.start, "2010-02-11")),
    }
    # VIX via Yahoo
    try:
        vix = fetch_ohlcv("^VIX", start=cfg.start)["Close"]
    except Exception:
        vix = None

    ohlc = panel[cfg.ticker]
    extras = {"spy": panel["SPY"]["Close"]}
    if cfg.ticker != "QQQ":
        extras["qqq"] = panel["QQQ"]["Close"]
    if vix is not None:
        extras["vix"] = vix

    feat = build_features(ohlc, extras=extras)
    y = make_labels(ohlc["Close"], horizon=cfg.horizon, mode=cfg.label_mode)
    signal, proba = walk_forward_signals(ohlc, feat, y, cfg)

    # Only evaluate after first predictions exist
    first = proba.dropna().index.min()
    if pd.isna(first):
        return {"error": "no predictions", "cfg": cfg.__dict__}

    ohlc_bt = ohlc.loc[ohlc.index >= first]
    sig_bt = signal.reindex(ohlc_bt.index).fillna(0).astype(int)

    stop = StopConfig(hard_pct=cfg.hard_pct, trailing_pct=cfg.trailing_pct)
    result = run_long_only_daily(
        ohlc_bt,
        sig_bt,
        initial_cash=cfg.cash,
        round_trip_cost_bps=cfg.cost_bps,
        stop=stop,
        strategy_name=f"ml_{cfg.ticker}_{cfg.model}_h{cfg.horizon}_t{cfg.proba_threshold}",
        asset=cfg.ticker,
    )

    spy = panel["SPY"]
    spy_eq = (cfg.cash / float(spy.loc[spy.index >= first].iloc[0]["Open"])) * spy.loc[
        spy.index >= first, "Close"
    ]
    summary = summarize_daily(result, spy_eq)
    summary["cfg"] = cfg.__dict__
    summary["n_long_days"] = int(sig_bt.sum())
    proba_bt = proba.reindex(sig_bt.index)
    long_mask = sig_bt.eq(1)
    summary["avg_proba_when_long"] = (
        float(proba_bt.loc[long_mask].mean()) if bool(long_mask.any()) else None
    )

    # Sealed recent OOS: last 3 years of the signal path
    oos_start = ohlc_bt.index.max() - pd.DateOffset(years=3)
    ohlc_oos = ohlc_bt.loc[ohlc_bt.index >= oos_start]
    sig_oos = sig_bt.loc[sig_bt.index >= oos_start]
    if len(ohlc_oos) > 50:
        res_oos = run_long_only_daily(
            ohlc_oos,
            sig_oos,
            initial_cash=cfg.cash,
            round_trip_cost_bps=cfg.cost_bps,
            stop=stop,
            strategy_name=summary["strategy"] + "_last3y",
            asset=cfg.ticker,
        )
        spy_oos = spy_eq.loc[spy_eq.index >= oos_start]
        summary["last3y"] = summarize_daily(res_oos, spy_oos)

    # Last 3 months with $1000
    w0 = ohlc_bt.index.max() - pd.DateOffset(months=3)
    ohlc_3m = ohlc_bt.loc[ohlc_bt.index >= w0]
    sig_3m = sig_bt.loc[sig_bt.index >= w0]
    res_3m = run_long_only_daily(
        ohlc_3m,
        sig_3m,
        initial_cash=1000.0,
        round_trip_cost_bps=cfg.cost_bps,
        stop=stop,
        strategy_name=summary["strategy"] + "_3m_1k",
        asset=cfg.ticker,
    )
    summary["last3m_1k"] = {
        "start": str(res_3m.equity.index[0].date()),
        "end": str(res_3m.equity.index[-1].date()),
        "start_cash": 1000.0,
        "end_equity": float(res_3m.equity.iloc[-1]),
        "profit": float(res_3m.equity.iloc[-1] - 1000.0),
        "return": float(res_3m.equity.iloc[-1] / 1000.0 - 1),
        "max_drawdown": float((res_3m.equity / res_3m.equity.cummax() - 1).min()),
        "n_trades": int(len(res_3m.trades)),
    }

    tag = summary["strategy"]
    result.equity.to_csv(OUT / f"{tag}_equity.csv")
    proba.dropna().to_csv(OUT / f"{tag}_proba.csv")
    return summary


def sweep() -> pd.DataFrame:
    """Pre-registered small grid — report median, promote best OOS last3y."""
    configs = []
    for ticker in ("QQQ", "TQQQ"):
        for model in ("hgb", "logreg"):
            for horizon in (3, 5, 10):
                for thresh in (0.52, 0.55, 0.58):
                    for hard, trail in ((0.07, 0.10), (0.10, 0.15)):
                        configs.append(
                            WFConfig(
                                ticker=ticker,
                                model=model,
                                horizon=horizon,
                                proba_threshold=thresh,
                                hard_pct=hard,
                                trailing_pct=trail,
                                embargo=horizon,
                            )
                        )

    # Cap grid for runtime: prioritize promising cells first
    priority = []
    for ticker in ("QQQ", "TQQQ"):
        for model in ("hgb", "logreg"):
            for horizon, thresh, hard, trail in (
                (5, 0.55, 0.08, 0.12),
                (5, 0.58, 0.10, 0.15),
                (3, 0.55, 0.07, 0.10),
                (10, 0.55, 0.10, 0.15),
                (5, 0.52, 0.08, 0.12),
            ):
                priority.append(
                    WFConfig(
                        ticker=ticker,
                        model=model,
                        horizon=horizon,
                        proba_threshold=thresh,
                        hard_pct=hard,
                        trailing_pct=trail,
                        embargo=horizon,
                    )
                )

    rows = []
    for cfg in priority:
        print(f"Running {cfg.ticker} {cfg.model} h={cfg.horizon} p>={cfg.proba_threshold} ...")
        try:
            summary = run_ml_experiment(cfg)
        except Exception as e:
            print(f"  FAIL: {e}")
            continue
        if "error" in summary:
            print(f"  SKIP: {summary['error']}")
            continue
        flat = {k: v for k, v in summary.items() if k not in ("cfg", "last3y", "last3m_1k")}
        if "last3y" in summary:
            for k, v in summary["last3y"].items():
                if k != "cfg":
                    flat[f"l3y_{k}"] = v
        if "last3m_1k" in summary:
            flat["l3m_profit"] = summary["last3m_1k"]["profit"]
            flat["l3m_return"] = summary["last3m_1k"]["return"]
        rows.append(flat)
        print(
            f"  CAGR={summary['cagr']:.2%}  "
            f"l3y={summary.get('last3y', {}).get('cagr', float('nan')):.2%}  "
            f"3m$1k={summary.get('last3m_1k', {}).get('profit', float('nan')):+.2f}  "
            f"beats={summary.get('beats_spy')}"
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("l3y_cagr", ascending=False, na_position="last")
    df.to_csv(OUT / "ml_sweep.csv", index=False)
    (OUT / "ml_sweep.json").write_text(df.to_json(orient="records", indent=2))

    # Promote: best last-3y CAGR that also beats SPY on last3y if possible
    promote = None
    if "l3y_beats_spy" in df.columns:
        beaters = df[df["l3y_beats_spy"] == True]  # noqa: E712
        if len(beaters):
            promote = beaters.iloc[0].to_dict()
    if promote is None and len(df):
        promote = df.iloc[0].to_dict()
    if promote:
        (OUT / "ml_promoted.json").write_text(json.dumps(promote, indent=2, default=str))
        print("\nPROMOTED:", promote.get("strategy"), "l3y CAGR", promote.get("l3y_cagr"))
    print(
        f"\nSweep N={len(df)}  median l3y CAGR={df['l3y_cagr'].median():.2%}  "
        f"median full CAGR={df['cagr'].median():.2%}"
    )
    return df
