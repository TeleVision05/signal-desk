"""Performance reporting vs SPY and inflation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .backtest import BacktestResult, align_monthly_returns, buy_and_hold_equity, cagr, max_drawdown, sharpe


@dataclass
class Report:
    strategy: str
    start: str
    end: str
    cagr: float
    total_return: float
    sharpe: float
    max_drawdown: float
    positive_month_pct: float
    positive_week_pct: float
    n_trades: int
    spy_cagr: float
    spy_total_return: float
    spy_max_drawdown: float
    excess_cagr_vs_spy: float
    avg_cpi_yoy: float | None
    real_cagr_approx: float | None
    beats_spy: bool
    beats_inflation: bool | None

    def to_dict(self) -> dict:
        return asdict(self)


def weekly_hit_rate(equity: pd.Series) -> float:
    w = equity.resample("W-FRI").last().dropna().pct_change().dropna()
    if w.empty:
        return float("nan")
    return float((w > 0).mean())


def build_report(
    result: BacktestResult,
    panel: dict[str, pd.DataFrame],
    cpi_yoy: pd.Series | None = None,
    *,
    rf_annual: float = 0.02,
) -> Report:
    eq = result.equity
    start = eq.index[0]
    spy = buy_and_hold_equity(panel, "SPY", start=start)
    # Align end
    end = min(eq.index[-1], spy.index[-1])
    eq = eq.loc[:end]
    spy = spy.loc[start:end]
    # Rebase spy to same initial
    spy = spy / spy.iloc[0] * eq.iloc[0]

    mrets = align_monthly_returns(eq)
    strat_cagr = cagr(eq)
    spy_cagr = cagr(spy)
    avg_cpi = None
    real = None
    beats_inf = None
    if cpi_yoy is not None and not cpi_yoy.empty:
        overlap = cpi_yoy.loc[(cpi_yoy.index >= start) & (cpi_yoy.index <= end)]
        if not overlap.empty:
            avg_cpi = float(overlap.mean())
            real = strat_cagr - avg_cpi
            beats_inf = bool(real > 0)

    return Report(
        strategy=result.strategy,
        start=str(start.date()),
        end=str(end.date()),
        cagr=strat_cagr,
        total_return=float(eq.iloc[-1] / eq.iloc[0] - 1),
        sharpe=sharpe(mrets, rf_annual=rf_annual),
        max_drawdown=max_drawdown(eq),
        positive_month_pct=float((mrets > 0).mean()) if len(mrets) else float("nan"),
        positive_week_pct=weekly_hit_rate(eq),
        n_trades=int(len(result.trades)),
        spy_cagr=spy_cagr,
        spy_total_return=float(spy.iloc[-1] / spy.iloc[0] - 1),
        spy_max_drawdown=max_drawdown(spy),
        excess_cagr_vs_spy=strat_cagr - spy_cagr,
        avg_cpi_yoy=avg_cpi,
        real_cagr_approx=real,
        beats_spy=bool(strat_cagr > spy_cagr),
        beats_inflation=beats_inf,
    )


def walk_forward_slices(
    index: pd.DatetimeIndex,
    *,
    train_years: int = 8,
    test_years: int = 2,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Expanding-style contiguous train/test windows on monthly index."""
    if len(index) < (train_years + test_years) * 12:
        return []
    slices = []
    start = index[0]
    cursor = start + pd.DateOffset(years=train_years)
    while True:
        train_end = cursor
        test_end = cursor + pd.DateOffset(years=test_years)
        if test_end > index[-1]:
            break
        slices.append((start, train_end, train_end, test_end))
        cursor = cursor + pd.DateOffset(years=test_years)
    return slices


def format_report(r: Report) -> str:
    lines = [
        f"Strategy: {r.strategy}",
        f"Period:   {r.start} → {r.end}",
        f"CAGR:     {r.cagr:.2%}  |  SPY BH: {r.spy_cagr:.2%}  |  excess: {r.excess_cagr_vs_spy:+.2%}",
        f"Total:    {r.total_return:.2%}  |  SPY: {r.spy_total_return:.2%}",
        f"Sharpe:   {r.sharpe:.2f}",
        f"MaxDD:    {r.max_drawdown:.2%}  |  SPY MaxDD: {r.spy_max_drawdown:.2%}",
        f"Hit rate: months {r.positive_month_pct:.1%}  weeks {r.positive_week_pct:.1%}",
        f"Trades:   {r.n_trades}",
    ]
    if r.avg_cpi_yoy is not None:
        lines.append(
            f"Inflation avg YoY: {r.avg_cpi_yoy:.2%}  |  approx real CAGR: {r.real_cagr_approx:.2%}  |  beats CPI: {r.beats_inflation}"
        )
    lines.append(f"Beats SPY (CAGR): {r.beats_spy}")
    return "\n".join(lines)