"""CLI entrypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .backtest import run_monthly_strategy
from .data import fetch_cpi_yoy, fetch_panel, month_end_closes
from .metrics import build_report, format_report, walk_forward_slices
from .paper import sync_gem_paper
from .strategies import buy_and_hold, faber_sma, gem_dual_momentum

OUT = Path(__file__).resolve().parents[2] / "output"


def cmd_aggressive(args: argparse.Namespace) -> None:
    from .aggressive_cli import print_bakeoff, run_bakeoff

    df, oos_df, promote = run_bakeoff(
        start=args.start, cash=args.cash, cost_bps=args.cost_bps
    )
    print_bakeoff(df, oos_df, promote)


def cmd_ml(args: argparse.Namespace) -> None:
    from .ml_walkforward import WFConfig, run_ml_experiment, sweep

    if args.sweep:
        df = sweep()
        if len(df):
            cols = [
                c
                for c in (
                    "strategy",
                    "cagr",
                    "l3y_cagr",
                    "l3y_excess_cagr_vs_spy",
                    "l3y_max_drawdown",
                    "l3m_profit",
                    "l3m_return",
                    "beats_spy",
                )
                if c in df.columns
            ]
            print(df.head(15)[cols].to_string(index=False))
        return
    cfg = WFConfig(
        ticker=args.ticker,
        model=args.model,
        horizon=args.horizon,
        proba_threshold=args.threshold,
        hard_pct=args.hard,
        trailing_pct=args.trail,
        cash=args.cash,
    )
    summary = run_ml_experiment(cfg)
    print(json.dumps(summary, indent=2, default=str))


def cmd_fetch(_: argparse.Namespace) -> None:
    tickers = ["SPY", "VEU", "AGG", "BIL"]
    panel = fetch_panel(tickers)
    for t, df in panel.items():
        print(f"{t}: {len(df)} rows  {df.index[0].date()} → {df.index[-1].date()}")
    cpi = fetch_cpi_yoy()
    print(f"CPI YoY: {len(cpi)} months  last={float(cpi.iloc[-1]):.2%}")


def cmd_backtest(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = fetch_panel(["SPY", "VEU", "AGG", "BIL", "QQQ"], start=args.start)
    gem_closes = month_end_closes({k: panel[k] for k in ("SPY", "VEU", "AGG")})
    gem_qqq_closes = month_end_closes({k: panel[k] for k in ("QQQ", "VEU", "AGG")})
    faber_closes = month_end_closes({k: panel[k] for k in ("SPY", "BIL")})
    cpi = fetch_cpi_yoy(start=args.start)

    strategies = {
        "gem_dual_momentum": (gem_dual_momentum(gem_closes), panel),
        "gem_qqq_veu": (
            gem_dual_momentum(gem_qqq_closes, risk_on=("QQQ", "VEU"), risk_off="AGG"),
            panel,
        ),
        "faber_sma_spy": (faber_sma(faber_closes, asset="SPY", cash="BIL", sma_months=10), panel),
        "buy_hold_spy": (buy_and_hold(gem_closes, "SPY"), panel),
        "buy_hold_qqq": (buy_and_hold(gem_qqq_closes, "QQQ"), panel),
    }

    reports = {}
    for name, (sig, p) in strategies.items():
        # Optional sealed OOS window
        if args.oos_start:
            sig = sig.loc[sig.index >= pd.Timestamp(args.oos_start)]
        result = run_monthly_strategy(
            p,
            sig,
            initial_cash=args.cash,
            round_trip_cost_bps=args.cost_bps,
            strategy_name=name,
        )
        report = build_report(result, p, cpi)
        reports[name] = report.to_dict()
        print("\n" + "=" * 60)
        print(format_report(report))
        result.equity.to_csv(OUT / f"{name}_equity.csv")
        result.trades.to_csv(OUT / f"{name}_trades.csv", index=False)

    (OUT / "backtest_reports.json").write_text(json.dumps(reports, indent=2))
    print(f"\nWrote reports to {OUT}")


def cmd_walkforward(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = fetch_panel(["SPY", "VEU", "AGG", "QQQ"], start=args.start)
    variants = {
        "gem_spy_veu": (
            month_end_closes({k: panel[k] for k in ("SPY", "VEU", "AGG")}),
            ("SPY", "VEU"),
            "AGG",
        ),
        "gem_qqq_veu": (
            month_end_closes({k: panel[k] for k in ("QQQ", "VEU", "AGG")}),
            ("QQQ", "VEU"),
            "AGG",
        ),
    }
    cpi = fetch_cpi_yoy(start=args.start)
    for vname, (month_closes, risk_on, risk_off) in variants.items():
        print(f"\n=== Walk-forward {vname} ===")
        signals = gem_dual_momentum(month_closes, risk_on=risk_on, risk_off=risk_off)
        slices = walk_forward_slices(month_closes.index, train_years=8, test_years=2)
        rows = []
        for train_start, train_end, test_start, test_end in slices:
            sig = signals.loc[(signals.index > test_start) & (signals.index <= test_end)]
            if len(sig) < 6:
                continue
            result = run_monthly_strategy(
                panel,
                sig,
                initial_cash=args.cash,
                round_trip_cost_bps=args.cost_bps,
                strategy_name=f"{vname}_wf",
            )
            report = build_report(result, panel, cpi)
            rows.append(
                {
                    "variant": vname,
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                    **report.to_dict(),
                }
            )
            print(
                f"OOS {test_start.date()}→{test_end.date()}  "
                f"CAGR {report.cagr:.2%} vs SPY {report.spy_cagr:.2%}  "
                f"MaxDD {report.max_drawdown:.2%}  beats_spy={report.beats_spy}"
            )
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"walkforward_{vname}.csv", index=False)
        if not df.empty:
            print(f"\nWalk-forward summary ({vname}):")
            print(f"  mean excess vs SPY: {df['excess_cagr_vs_spy'].mean():+.2%}")
            print(f"  fractions beats SPY: {(df['beats_spy']).mean():.1%}")
            print(f"  mean MaxDD: {df['max_drawdown'].mean():.2%}")
            if df["beats_inflation"].notna().any():
                print(f"  fractions beats CPI: {df['beats_inflation'].mean():.1%}")


def cmd_paper(args: argparse.Namespace) -> None:
    if getattr(args, "ml", False):
        from .paper_ml import sync_ml_paper

        info = sync_ml_paper(initial_cash=getattr(args, "cash", 1000.0))
    elif getattr(args, "aggressive", False):
        from .paper_aggressive import sync_aggressive_paper

        info = sync_aggressive_paper()
    else:
        info = sync_gem_paper()
    print(json.dumps(info, indent=2))


def cmd_iterate(args: argparse.Namespace) -> None:
    """
    Honest limited grid: few pre-registered variants, report MEDIAN not best cell.
    Does not promote the best-looking lookback.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    panel = fetch_panel(["SPY", "VEU", "AGG", "BIL"], start=args.start)
    gem_closes = month_end_closes({k: panel[k] for k in ("SPY", "VEU", "AGG")})
    faber_closes = month_end_closes({k: panel[k] for k in ("SPY", "BIL")})
    cpi = fetch_cpi_yoy(start=args.start)
    oos = pd.Timestamp(args.oos_start)
    variants = []
    for lookback in (6, 9, 12):
        for cutoff in (0.0,):
            name = f"gem_lb{lookback}_cut{cutoff}"
            sig = gem_dual_momentum(gem_closes, lookback=lookback, absolute_cutoff=cutoff)
            sig = sig.loc[sig.index >= oos]
            result = run_monthly_strategy(
                panel, sig, initial_cash=args.cash, round_trip_cost_bps=args.cost_bps, strategy_name=name
            )
            report = build_report(result, panel, cpi)
            variants.append(report.to_dict())
            print(f"{name}: CAGR {report.cagr:.2%} excess {report.excess_cagr_vs_spy:+.2%} MaxDD {report.max_drawdown:.2%}")

    # Faber control
    sig = faber_sma(faber_closes)
    sig = sig.loc[sig.index >= oos]
    result = run_monthly_strategy(
        panel, sig, initial_cash=args.cash, round_trip_cost_bps=args.cost_bps, strategy_name="faber_oos"
    )
    report = build_report(result, panel, cpi)
    variants.append(report.to_dict())
    print(f"faber_oos: CAGR {report.cagr:.2%} excess {report.excess_cagr_vs_spy:+.2%} MaxDD {report.max_drawdown:.2%}")

    df = pd.DataFrame(variants)
    (OUT / "iteration_grid.json").write_text(json.dumps(variants, indent=2))
    gem_only = df[df["strategy"].str.startswith("gem_")]
    print("\nMEDIAN GEM cell (not best):")
    print(f"  median CAGR: {gem_only['cagr'].median():.2%}")
    print(f"  median excess vs SPY: {gem_only['excess_cagr_vs_spy'].median():+.2%}")
    print(f"  median MaxDD: {gem_only['max_drawdown'].median():.2%}")
    print(f"  trial count N={len(gem_only)} (feed into Deflated Sharpe if promoting)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="profit-engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="Download real market + CPI data")
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("backtest", help="Run frozen strategies")
    b.add_argument("--start", default="2007-01-01")
    b.add_argument("--oos-start", default=None)
    b.add_argument("--cash", type=float, default=100_000.0)
    b.add_argument("--cost-bps", type=float, default=7.5)
    b.set_defaults(func=cmd_backtest)

    w = sub.add_parser("walkforward", help="Walk-forward OOS for GEM")
    w.add_argument("--start", default="2007-01-01")
    w.add_argument("--cash", type=float, default=100_000.0)
    w.add_argument("--cost-bps", type=float, default=7.5)
    w.set_defaults(func=cmd_walkforward)

    i = sub.add_parser("iterate", help="Small pre-registered grid; report median")
    i.add_argument("--start", default="2007-01-01")
    i.add_argument("--oos-start", default="2019-01-01")
    i.add_argument("--cash", type=float, default=100_000.0)
    i.add_argument("--cost-bps", type=float, default=7.5)
    i.set_defaults(func=cmd_iterate)

    pap = sub.add_parser("paper", help="Sync paper wallet (GEM by default)")
    pap.add_argument(
        "--aggressive",
        action="store_true",
        help="Use promoted TQQQ mom10 + stop-loss paper book",
    )
    pap.add_argument(
        "--ml",
        action="store_true",
        help="Use promoted walk-forward ML TQQQ paper book",
    )
    pap.add_argument("--cash", type=float, default=1000.0, help="Initial cash for new paper runs")
    pap.set_defaults(func=cmd_paper)

    ag = sub.add_parser(
        "aggressive",
        help="Bake off riskier daily/weekly strategies with hard+trailing stops",
    )
    ag.add_argument("--start", default="2015-01-01")
    ag.add_argument("--cash", type=float, default=100_000.0)
    ag.add_argument("--cost-bps", type=float, default=10.0)
    ag.set_defaults(func=cmd_aggressive)

    ml = sub.add_parser("ml", help="Walk-forward ML long/flat with stops")
    ml.add_argument("--sweep", action="store_true", help="Run pre-registered grid")
    ml.add_argument("--ticker", default="QQQ")
    ml.add_argument("--model", default="hgb", choices=["hgb", "gb", "logreg"])
    ml.add_argument("--horizon", type=int, default=5)
    ml.add_argument("--threshold", type=float, default=0.55)
    ml.add_argument("--hard", type=float, default=0.08)
    ml.add_argument("--trail", type=float, default=0.12)
    ml.add_argument("--cash", type=float, default=100_000.0)
    ml.set_defaults(func=cmd_ml)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()