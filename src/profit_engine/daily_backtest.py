"""Daily bar backtester with hard and trailing stop-losses (next-open entries)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import BacktestResult, cagr, max_drawdown, sharpe


@dataclass
class StopConfig:
    hard_pct: float | None = 0.08  # exit if close or low breaches entry*(1-hard)
    trailing_pct: float | None = 0.12  # trail from peak since entry
    use_intrabar_low: bool = True  # if Low <= stop, fill at stop


def _stop_hit(
    bar: pd.Series,
    entry: float,
    peak: float,
    cfg: StopConfig,
) -> tuple[bool, float | None]:
    """Return (hit, exit_price). Prefer trailing then hard; fill at stop level."""
    stops = []
    if cfg.hard_pct is not None:
        stops.append(entry * (1.0 - cfg.hard_pct))
    if cfg.trailing_pct is not None:
        stops.append(peak * (1.0 - cfg.trailing_pct))
    if not stops:
        return False, None
    stop_px = max(stops)  # highest floor among stops = most protective for long
    # For long: we exit when price falls TO the stop. The binding stop is the
    # higher of hard and trailing floors.
    low = float(bar["Low"]) if cfg.use_intrabar_low else float(bar["Close"])
    close = float(bar["Close"])
    if low <= stop_px:
        # Conservative: fill at stop (assume we get out at stop, not worse)
        return True, float(stop_px)
    if close <= stop_px:
        return True, close
    return False, None


def run_long_only_daily(
    ohlc: pd.DataFrame,
    signals: pd.Series,
    *,
    initial_cash: float = 100_000.0,
    round_trip_cost_bps: float = 10.0,
    stop: StopConfig | None = None,
    strategy_name: str = "daily",
    asset: str = "ASSET",
) -> BacktestResult:
    """
    `signals` is a daily Series of {1=want long, 0=flat}, known at close.
    Entries/exits (non-stop) execute at the next session Open.
    Stop exits can fire intrabar using Low.
    """
    stop = stop or StopConfig()
    df = ohlc.sort_index().dropna(subset=["Open", "High", "Low", "Close"]).copy()
    sig = signals.reindex(df.index).ffill().fillna(0).astype(int)

    cash = float(initial_cash)
    shares = 0.0
    entry_px = 0.0
    peak = 0.0
    pending: str | None = None  # 'buy' | 'sell'
    equity_rows = []
    trade_rows = []
    hold_rows = []
    half = round_trip_cost_bps / 10_000.0 / 2.0

    for i, (dt, bar) in enumerate(df.iterrows()):
        open_px = float(bar["Open"])
        close_px = float(bar["Close"])
        high_px = float(bar["High"])

        # 1) Execute pending orders at open
        if pending == "sell" and shares > 0:
            cost = shares * open_px * half
            cash += shares * open_px - cost
            trade_rows.append(
                {
                    "date": dt,
                    "side": "SELL",
                    "reason": "signal",
                    "shares": shares,
                    "price": open_px,
                    "cost": cost,
                }
            )
            shares = 0.0
            entry_px = 0.0
            peak = 0.0
            pending = None
        elif pending == "buy" and shares == 0 and cash > 0:
            cost = cash * half
            investable = cash - cost
            shares = investable / open_px
            cash = 0.0
            entry_px = open_px
            peak = open_px
            trade_rows.append(
                {
                    "date": dt,
                    "side": "BUY",
                    "reason": "signal",
                    "shares": shares,
                    "price": open_px,
                    "cost": cost,
                }
            )
            pending = None

        # 2) Stop check while long (after open fill)
        if shares > 0:
            peak = max(peak, high_px)
            hit, exit_px = _stop_hit(bar, entry_px, peak, stop)
            if hit and exit_px is not None:
                cost = shares * exit_px * half
                cash += shares * exit_px - cost
                trade_rows.append(
                    {
                        "date": dt,
                        "side": "SELL",
                        "reason": "stop",
                        "shares": shares,
                        "price": exit_px,
                        "cost": cost,
                    }
                )
                shares = 0.0
                entry_px = 0.0
                peak = 0.0
                pending = None  # cancel any pending

        # 3) New signal at close → queue next open (if not stopped to flat same day)
        want = int(sig.loc[dt])
        if shares > 0 and want == 0:
            pending = "sell"
        elif shares == 0 and want == 1:
            pending = "buy"

        equity = cash + shares * close_px
        equity_rows.append({"date": dt, "equity": equity})
        hold_rows.append({"date": dt, "asset": asset if shares > 0 else "CASH"})

    eq = pd.DataFrame(equity_rows).set_index("date")["equity"]
    holds = pd.DataFrame(hold_rows).set_index("date")["asset"]
    trades = pd.DataFrame(trade_rows)
    return BacktestResult(
        equity=eq,
        holdings=holds,
        trades=trades,
        returns=eq.pct_change().dropna(),
        strategy=strategy_name,
    )


def summarize_daily(result: BacktestResult, spy_equity: pd.Series | None = None) -> dict:
    eq = result.equity
    weekly = eq.resample("W-FRI").last().dropna().pct_change().dropna()
    daily = eq.pct_change().dropna()
    stops = 0
    if not result.trades.empty and "reason" in result.trades.columns:
        stops = int((result.trades["reason"] == "stop").sum())
    out = {
        "strategy": result.strategy,
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "cagr": cagr(eq),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "max_drawdown": max_drawdown(eq),
        "sharpe": sharpe(eq.resample("ME").last().pct_change().dropna()),
        "positive_week_pct": float((weekly > 0).mean()) if len(weekly) else float("nan"),
        "avg_weekly_return": float(weekly.mean()) if len(weekly) else float("nan"),
        "median_weekly_return": float(weekly.median()) if len(weekly) else float("nan"),
        "best_week": float(weekly.max()) if len(weekly) else float("nan"),
        "worst_week": float(weekly.min()) if len(weekly) else float("nan"),
        "positive_day_pct": float((daily > 0).mean()) if len(daily) else float("nan"),
        "avg_daily_return": float(daily.mean()) if len(daily) else float("nan"),
        "n_trades": int(len(result.trades)),
        "n_stop_exits": stops,
        "time_in_market": float((result.holdings != "CASH").mean()),
    }
    if spy_equity is not None and len(spy_equity):
        aligned = spy_equity.reindex(eq.index).ffill().dropna()
        common = eq.index.intersection(aligned.index)
        if len(common) > 2:
            s = aligned.loc[common]
            s = s / s.iloc[0] * eq.loc[common].iloc[0]
            out["spy_cagr"] = cagr(s)
            out["excess_cagr_vs_spy"] = out["cagr"] - out["spy_cagr"]
            out["beats_spy"] = bool(out["cagr"] > out["spy_cagr"])
    return out
