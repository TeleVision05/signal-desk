"""Strategies the books can name with ``"strategy": "<name>"`` in books.json (:data:`BUILTINS`)."""

from __future__ import annotations

from profit_engine.live.strategies.lgbm import LgbmStrategy
from profit_engine.live.strategies.voltarget import VolTargetTrend
from profit_engine.live.strategy import SmaTrendDemo

BUILTINS = {SmaTrendDemo.name: SmaTrendDemo, VolTargetTrend.name: VolTargetTrend, LgbmStrategy.name: LgbmStrategy}
