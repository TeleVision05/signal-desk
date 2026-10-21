"""Event-style monthly strategy backtester with next-open fills and costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import month_end_closes, next_open_prices


@dataclass
class BacktestResult:
    equity: pd.Series
    holdings: pd.Series
    trades: pd.DataFrame
    returns: pd.Series
    strategy: str


def run_monthly_strategy(
    panel: dict[str, pd.DataFrame],
    signals: pd.Series,
    *,
    initial_cash: float = 100_000.0,
    round_trip_cost_bps: float = 7.5,
    strategy_name: str = "strategy",
) -> BacktestResult:
    """
    At each month-end signal, rebalance at the next session open.
    Round-trip cost is charged on the notional that turns over.
    """
    month_closes = month_end_closes(panel)
    # Align signals to months that exist in closes
    signals = signals.reindex(month_closes.index).dropna()
    fills = next_open_prices(panel, signals.index)
    # Drop last signal if no fill yet
    common = signals.index.intersection(fills.index)
    signals = signals.loc[common]
    fills = fills.loc[common]

    cash = initial_cash
    shares = 0.0
    asset: str | None = None
    equity_rows = []
    trade_rows = []
    hold_rows = []

    # Mark-to-market path: daily closes after first fill for equity curve density
    # For monthly strategies we build equity at each fill + final mark.
    for i, dt in enumerate(signals.index):
        target = str(signals.loc[dt])
        fill_dt = fills.loc[dt, "fill_date"]
        price = float(fills.loc[dt, target])

        # Liquidate current if switching
        if asset is not None and asset != target and shares != 0:
            exit_px = float(fills.loc[dt, asset])
            gross = shares * exit_px
            cost = gross * (round_trip_cost_bps / 10_000.0) / 2.0  # one side
            cash += gross - cost
            trade_rows.append(
                {
                    "signal_date": dt,
                    "fill_date": fill_dt,
                    "side": "SELL",
                    "asset": asset,
                    "shares": shares,
                    "price": exit_px,
                    "cost": cost,
                }
            )
            shares = 0.0

        if asset != target or shares == 0:
            # Deploy all cash into target (half RT cost on entry)
            cost = cash * (round_trip_cost_bps / 10_000.0) / 2.0
            investable = cash - cost
            shares = investable / price
            cash = 0.0
            trade_rows.append(
                {
                    "signal_date": dt,
                    "fill_date": fill_dt,
                    "side": "BUY",
                    "asset": target,
                    "shares": shares,
                    "price": price,
                    "cost": cost,
                }
            )
            asset = target

        # Equity at fill using target close that day if available else open
        mark_px = float(panel[asset].loc[fill_dt, "Close"]) if fill_dt in panel[asset].index else price
        equity = cash + shares * mark_px
        equity_rows.append({"date": fill_dt, "equity": equity})
        hold_rows.append({"date": fill_dt, "asset": asset})

    # Final mark on last available close of held asset
    if asset is not None and shares > 0:
        last_px = float(panel[asset]["Close"].iloc[-1])
        last_dt = panel[asset].index[-1]
        equity_rows.append({"date": last_dt, "equity": cash + shares * last_px})
        hold_rows.append({"date": last_dt, "asset": asset})

    eq = pd.DataFrame(equity_rows).drop_duplicates("date").set_index("date")["equity"].sort_index()
    holds = pd.DataFrame(hold_rows).drop_duplicates("date").set_index("date")["asset"].sort_index()
    trades = pd.DataFrame(trade_rows)
    rets = eq.pct_change().dropna()
    return BacktestResult(
        equity=eq,
        holdings=holds,
        trades=trades,
        returns=rets,
        strategy=strategy_name,
    )


def buy_and_hold_equity(
    panel: dict[str, pd.DataFrame],
    asset: str = "SPY",
    *,
    initial_cash: float = 100_000.0,
    start: pd.Timestamp | None = None,
) -> pd.Series:
    df = panel[asset]
    if start is not None:
        df = df.loc[df.index >= start]
    if df.empty:
        raise ValueError("empty buy-and-hold series")
    shares = initial_cash / float(df.iloc[0]["Open"])
    return (shares * df["Close"]).rename("spy_bh")


def align_monthly_returns(equity: pd.Series) -> pd.Series:
    """Month-end returns from an equity curve."""
    me = equity.resample("ME").last().dropna()
    return me.pct_change().dropna()


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def sharpe(monthly_returns: pd.Series, rf_annual: float = 0.02) -> float:
    if monthly_returns.empty or monthly_returns.std() == 0:
        return float("nan")
    rf_m = (1 + rf_annual) ** (1 / 12) - 1
    excess = monthly_returns - rf_m
    return float(np.sqrt(12) * excess.mean() / excess.std())