"""Real cached bars and the frozen research modules, for the live-vs-research parity tests.

The research experiments are not a package; they are imported from ``experiments/intraday``
for COMPARISON ONLY (the live code never imports them). Without the harness cache
(``data/cache/intraday/<SYMBOL>_5m.parquet``) the parity tests skip and say how to build it.
"""

from __future__ import annotations

import importlib
import sys
from functools import cache

import pandas as pd
import pytest

from profit_engine.intraday import config, data
from profit_engine.intraday.resample import resample

EXPERIMENTS = config.REPO_ROOT / "experiments" / "intraday"


def research(name: str):
    """``experiments/intraday/<name>.py`` as a module (comparison only)."""
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    return importlib.import_module(name)


def require_cache() -> None:
    for symbol in config.SYMBOLS:
        path = data.cache_path(symbol)
        if not path.exists():
            pytest.skip(
                f"parity test needs the research cache {path}; build it with "
                f"`.venv/bin/python -c \"from profit_engine.intraday import data; data.sync_cache('{symbol}')\"`"
            )


@cache
def raw_5m(symbol: str) -> pd.DataFrame:
    """Closed 5m bars [TRAIN_START, HOLDOUT_START), exactly what the research loaded."""
    require_cache()
    return data.load_klines(symbol)


@cache
def bars(symbol: str, tf: str) -> pd.DataFrame:
    return resample(raw_5m(symbol), tf, incomplete="drop")
