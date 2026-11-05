"""Performance metrics on a :class:`BacktestResult` (daily P&L based, 365-day years).

Overfitting statistics (DSR, PBO, permutation test) live in :mod:`.overfitting`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from profit_engine.intraday import config
from profit_engine.intraday.backtest import BacktestResult
from profit_engine.intraday.holdout import assert_pre_holdout

ANN = np.sqrt(config.PERIODS_PER_YEAR)


def _guard(daily: pd.Series, what: str) -> None:
    """Every public helper refuses post-holdout dates (a result built inside open_holdout
    and analysed after it closed included)."""
    assert_pre_holdout(daily.index, f"metrics.{what}")


def sharpe(daily: pd.Series, annualize: bool = True) -> float:
    _guard(daily, "sharpe")
    sd = daily.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    sr = daily.mean() / sd
    return float(sr * ANN if annualize else sr)


def sortino(daily: pd.Series) -> float:
    _guard(daily, "sortino")
    downside = np.sqrt(np.mean(np.minimum(daily.to_numpy(), 0.0) ** 2))
    return float(daily.mean() / downside * ANN) if downside > 0 else 0.0


def cagr(daily: pd.Series) -> float:
    _guard(daily, "cagr")
    growth = float(np.prod(1 + daily.to_numpy()))
    years = len(daily) / config.PERIODS_PER_YEAR
    return growth ** (1 / years) - 1 if years > 0 and growth > 0 else -1.0


def max_drawdown(daily: pd.Series) -> float:
    _guard(daily, "max_drawdown")
    eq = np.cumprod(1 + daily.to_numpy())
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float((eq / peak - 1).min()) if len(eq) else 0.0


def t_stat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype="float64")
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def monthly_returns(daily: pd.Series) -> pd.Series:
    _guard(daily, "monthly_returns")
    naive = daily.copy()
    naive.index = naive.index.tz_localize(None)
    return (1 + naive).groupby(naive.index.to_period("M")).prod() - 1


def yearly_returns(daily: pd.Series) -> dict[int, float]:
    _guard(daily, "yearly_returns")
    yearly = (1 + daily).groupby(daily.index.year).prod() - 1
    return {int(y): float(r) for y, r in yearly.items()}


def best_month_share(daily: pd.Series) -> float:
    """Best month's log P&L / total log P&L. inf when total P&L <= 0 (always fails a < cap)."""
    _guard(daily, "best_month_share")
    lm = np.log1p(monthly_returns(daily))
    total = lm.sum()
    return float(lm.max() / total) if total > 0 else float("inf")


def trade_stats(trades: pd.DataFrame) -> dict:
    if "exit_time" in trades and len(trades):
        assert_pre_holdout(pd.DatetimeIndex(trades["exit_time"]), "metrics.trade_stats")
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "mean_net_bps": 0.0,
            "t_net_bps": 0.0,
            "mean_gross_bps": 0.0,
        }
    net = trades["net_ret"].to_numpy()
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    return {
        "n_trades": len(trades),
        "win_rate": float((net > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "mean_net_bps": float(trades["net_bps"].mean()),
        "t_net_bps": t_stat(trades["net_bps"].to_numpy()),
        "mean_gross_bps": float(trades["gross_bps"].mean()),
    }


def summarize(result: BacktestResult) -> dict:
    """All headline metrics for one backtest. Raises on post-holdout data outside open_holdout."""
    daily = result.daily
    assert_pre_holdout(daily.index, "metrics.summarize")
    months = monthly_returns(daily)
    pos = result.bars["position"]
    out = {
        "start": str(daily.index[0].date()),
        "end": str(daily.index[-1].date()),
        "n_days": len(daily),
        "cost_bps_per_side": result.cost_bps_per_side,
        "total_return": float(np.prod(1 + daily.to_numpy()) - 1),
        "cagr": cagr(daily),
        "sharpe": sharpe(daily),
        "sharpe_daily": sharpe(daily, annualize=False),
        "sortino": sortino(daily),
        "max_drawdown": max_drawdown(daily),
        "t_daily": t_stat(daily.to_numpy()),
        "skew": float(daily.skew()),
        "kurtosis": float(daily.kurt() + 3.0),  # non-excess, as DSR expects
        "exposure": float(pos.mean()),
        "time_in_market": float((pos > 0).mean()),
        "pct_positive_months": float((months > 0).mean()) if len(months) else 0.0,
        "worst_year": min(yearly_returns(daily).values()),
        "yearly_returns": yearly_returns(daily),
        "best_month_share": best_month_share(daily),
    }
    out.update(trade_stats(result.trades))
    out["trades_per_day"] = out["n_trades"] / max(len(daily), 1)
    return out


def one_line(name: str, m: dict) -> str:
    return (
        f"{name}: Sharpe {m['sharpe']:.2f} | CAGR {m['cagr']:.1%} | MaxDD {m['max_drawdown']:.1%} | "
        f"Sortino {m['sortino']:.2f} | trades {m['n_trades']} | exposure {m['exposure']:.0%} | "
        f"+months {m['pct_positive_months']:.0%} | worst yr {m['worst_year']:.1%}"
    )
