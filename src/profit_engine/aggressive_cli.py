"""Bake off aggressive daily/weekly strategies with stop-losses on real ETF data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .aggressive import breakout, dual_sma, rsi_pullback, sma_trend, weekly_momentum
from .daily_backtest import StopConfig, run_long_only_daily, summarize_daily
from .data import fetch_ohlcv

OUT = Path(__file__).resolve().parents[2] / "output"


def _spy_equity(start: str, end: str | None, cash: float) -> pd.Series:
    spy = fetch_ohlcv("SPY", start=start, end=end)
    shares = cash / float(spy.iloc[0]["Open"])
    return (shares * spy["Close"]).rename("spy")


def run_bakeoff(
    *,
    start: str = "2015-01-01",
    cash: float = 100_000.0,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    tickers = ["TQQQ", "QQQ", "SPY", "SOXL"]
    panels = {t: fetch_ohlcv(t, start=start) for t in tickers}
    spy_eq = _spy_equity(start, None, cash)

    # (name, ticker, signal_fn, stop)
    jobs = []
    for ticker in tickers:
        close = panels[ticker]["Close"]
        jobs.extend(
            [
                (
                    f"{ticker}_sma50_hard8_trail12",
                    ticker,
                    sma_trend(close, 50),
                    StopConfig(hard_pct=0.08, trailing_pct=0.12),
                ),
                (
                    f"{ticker}_sma20_50_hard10_trail15",
                    ticker,
                    dual_sma(close, 20, 50),
                    StopConfig(hard_pct=0.10, trailing_pct=0.15),
                ),
                (
                    f"{ticker}_mom20_hard8_trail10",
                    ticker,
                    weekly_momentum(close, 20, 0.0),
                    StopConfig(hard_pct=0.08, trailing_pct=0.10),
                ),
                (
                    f"{ticker}_breakout20_hard7_trail12",
                    ticker,
                    breakout(close, 20),
                    StopConfig(hard_pct=0.07, trailing_pct=0.12),
                ),
                (
                    f"{ticker}_rsi35_55_hard8_trail12",
                    ticker,
                    rsi_pullback(close, 14, 35, 55),
                    StopConfig(hard_pct=0.08, trailing_pct=0.12),
                ),
            ]
        )
        # Tighter / looser stop variants on best-looking structures for leveraged
        if ticker in ("TQQQ", "SOXL"):
            jobs.extend(
                [
                    (
                        f"{ticker}_sma50_hard12_trail20",
                        ticker,
                        sma_trend(close, 50),
                        StopConfig(hard_pct=0.12, trailing_pct=0.20),
                    ),
                    (
                        f"{ticker}_mom10_hard10_trail15",
                        ticker,
                        weekly_momentum(close, 10, 0.0),
                        StopConfig(hard_pct=0.10, trailing_pct=0.15),
                    ),
                    (
                        f"{ticker}_dualsma_hard15_trail25",
                        ticker,
                        dual_sma(close, 10, 30),
                        StopConfig(hard_pct=0.15, trailing_pct=0.25),
                    ),
                ]
            )

    rows = []
    best = None
    best_score = float("-inf")
    for name, ticker, sig, stop in jobs:
        result = run_long_only_daily(
            panels[ticker],
            sig,
            initial_cash=cash,
            round_trip_cost_bps=cost_bps,
            stop=stop,
            strategy_name=name,
            asset=ticker,
        )
        # Need enough history
        if len(result.equity) < 200:
            continue
        summary = summarize_daily(result, spy_eq)
        rows.append(summary)
        # Score: reward CAGR and weekly avg, penalize drawdown severity
        dd = abs(summary["max_drawdown"])
        score = summary["cagr"] + 0.5 * summary.get("avg_weekly_return", 0) * 52 - 0.35 * dd
        if summary.get("beats_spy") and score > best_score:
            best_score = score
            best = (name, result, summary, stop, ticker, sig)

        result.equity.to_csv(OUT / f"agg_{name}_equity.csv")

    df = pd.DataFrame(rows).sort_values("cagr", ascending=False)
    df.to_csv(OUT / "aggressive_bakeoff.csv", index=False)
    (OUT / "aggressive_bakeoff.json").write_text(df.to_json(orient="records", indent=2))

    # Sealed OOS on top candidates by full-sample CAGR (top 5) + report median
    oos_start = pd.Timestamp("2022-01-01")
    oos_rows = []
    top_names = set(df.head(8)["strategy"].tolist()) if len(df) else set()
    for name, ticker, sig, stop in jobs:
        if name not in top_names:
            continue
        # Restrict to OOS
        ohlc = panels[ticker].loc[panels[ticker].index >= oos_start]
        sig_oos = sig.loc[sig.index >= oos_start]
        if len(ohlc) < 100:
            continue
        result = run_long_only_daily(
            ohlc,
            sig_oos,
            initial_cash=cash,
            round_trip_cost_bps=cost_bps,
            stop=stop,
            strategy_name=name + "_oos2022",
            asset=ticker,
        )
        spy_oos = spy_eq.loc[spy_eq.index >= oos_start]
        summary = summarize_daily(result, spy_oos)
        oos_rows.append(summary)

    oos_df = pd.DataFrame(oos_rows).sort_values("cagr", ascending=False) if oos_rows else pd.DataFrame()
    if len(oos_df):
        oos_df.to_csv(OUT / "aggressive_oos_2022.csv", index=False)

    # Pick promote: best OOS that still beats SPY if any, else best full-sample risk-adjusted
    promote = None
    if len(oos_df):
        beaters = oos_df[oos_df.get("beats_spy", False) == True]  # noqa: E712
        if len(beaters):
            promote = beaters.iloc[0].to_dict()
        else:
            promote = oos_df.iloc[0].to_dict()
    elif best:
        promote = best[2]

    if promote:
        (OUT / "aggressive_promoted.json").write_text(json.dumps(promote, indent=2))

    return df, oos_df, promote


def print_bakeoff(df: pd.DataFrame, oos_df: pd.DataFrame, promote: dict | None) -> None:
    print("\n=== FULL-SAMPLE TOP 12 BY CAGR ===")
    cols = [
        "strategy",
        "cagr",
        "excess_cagr_vs_spy",
        "max_drawdown",
        "avg_weekly_return",
        "positive_week_pct",
        "avg_daily_return",
        "n_stop_exits",
        "beats_spy",
    ]
    show = [c for c in cols if c in df.columns]
    top = df.head(12)[show].copy()
    for c in ("cagr", "excess_cagr_vs_spy", "max_drawdown", "avg_weekly_return", "positive_week_pct", "avg_daily_return"):
        if c in top.columns:
            top[c] = top[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    print(top.to_string(index=False))

    if len(oos_df):
        print("\n=== OOS 2022+ (top full-sample names) ===")
        show2 = [c for c in cols if c in oos_df.columns]
        t2 = oos_df[show2].copy()
        for c in ("cagr", "excess_cagr_vs_spy", "max_drawdown", "avg_weekly_return", "positive_week_pct", "avg_daily_return"):
            if c in t2.columns:
                t2[c] = t2[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        print(t2.to_string(index=False))
        print(
            f"\nOOS median CAGR: {oos_df['cagr'].median():.2%}  "
            f"median weekly: {oos_df['avg_weekly_return'].median():.2%}  "
            f"fractions beat SPY: {oos_df['beats_spy'].mean():.0%}"
        )

    if promote:
        print("\n=== PROMOTED (paper candidate) ===")
        print(json.dumps(promote, indent=2, default=str))
