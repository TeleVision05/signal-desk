"""Sync the cache and print a data summary + buy-and-hold metrics per symbol and timeframe.

python -m profit_engine.intraday.sanity            # sync from network, then summarise
python -m profit_engine.intraday.sanity --offline  # summarise the existing cache only
"""

from __future__ import annotations

import argparse
import json

from profit_engine.intraday import config, data
from profit_engine.intraday.backtest import buy_and_hold
from profit_engine.intraday.metrics import one_line, summarize
from profit_engine.intraday.resample import resample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--offline", action="store_true", help="do not hit the network")
    args = ap.parse_args()
    for sym in config.SYMBOLS:
        if args.offline:
            rep = data.DataReport(**json.loads(data.cache_path(sym).with_suffix(".json").read_text()))
        else:
            rep = data.sync_cache(sym)
        print(rep.line())
        print(f"  source={rep.source}; largest gaps: {rep.largest_gaps[:3]}")
    window = f"{config.EVAL_START.date()}..{(config.EVAL_END - config.BAR).date()}"
    print(f"\nBuy-and-hold, walk-forward range {window}, {config.BASE_COST_BPS:g} bps/side:")
    for sym in config.SYMBOLS:
        raw = data.load_klines(sym)
        for tf in config.TIMEFRAMES:
            bars = raw if tf == "5m" else resample(raw, tf, incomplete="drop")
            dropped = "" if tf == "5m" else f" (incomplete {tf} bars dropped: {_n_incomplete(raw, tf)})"
            oos = bars[(bars.index >= config.EVAL_START) & (bars.index < config.EVAL_END)]
            m = summarize(buy_and_hold(oos))
            print(f"  {one_line(f'{sym} {tf:>3} B&H', m)}{dropped}")


def _n_incomplete(raw, tf: str) -> int:
    flagged = resample(raw, tf, incomplete="flag")
    return int((~flagged["complete"]).sum())


if __name__ == "__main__":
    main()
