"""Paper-book fill arithmetic, identical to the harness simulator.

``intraday.backtest`` holds SHARES between fills, so a fractional weight drifts with the
price. A fill happens only when the target changes; it re-sets the weight to the new target
and costs ``cost_bps / 1e4 * |target - drifted weight| * equity``. Expressed with cash and
units instead of compounded returns, that is exactly what :func:`plan_fill` computes, so a
replayed bar stream reproduces ``run_backtest`` positions, fills and equity.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FillPlan:
    price: float
    target: float
    prev_target: float
    weight_before: float
    units_before: float
    units_after: float
    cash_before: float
    cash_after: float
    equity_before: float
    equity_after: float
    cost: float

    @property
    def side(self) -> str:
        return "buy" if self.units_after > self.units_before else "sell"

    @property
    def units_delta(self) -> float:
        return self.units_after - self.units_before

    def as_dict(self) -> dict:
        return {**asdict(self), "side": self.side, "units_delta": self.units_delta}


def equity(cash: float, units: float, price: float) -> float:
    return cash + units * price


def check_price(price: float) -> float:
    price = float(price)
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"fill price must be finite and positive, got {price!r}")
    return price


def plan_fill(
    cash: float, units: float, prev_target: float, target: float, price: float, cost_bps: float
) -> FillPlan | None:
    """The fill that moves the book from ``prev_target`` to ``target`` at ``price``, or None.

    None when the target did not change (the harness fills only on a change; a fractional
    weight is left to drift in between).
    """
    if target == prev_target:
        return None
    price = check_price(price)
    eq = equity(cash, units, price)
    if eq <= 0:
        raise ValueError("book equity is not positive; refusing to trade")
    weight = units * price / eq
    cost = cost_bps / 1e4 * abs(target - weight) * eq
    eq_after = eq - cost
    units_after = target * eq_after / price
    return FillPlan(
        price=price,
        target=float(target),
        prev_target=float(prev_target),
        weight_before=weight,
        units_before=units,
        units_after=units_after,
        cash_before=cash,
        cash_after=eq_after - units_after * price,
        equity_before=eq,
        equity_after=eq_after,
        cost=cost,
    )
