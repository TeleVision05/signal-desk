"""One-time, audited holdout read (2026-07-01 .. latest closed bar).

No candidate PASSed promotion, so this is NOT a promotion step. It is an informational
forward check of the three strongest walk-forward configs and buy-and-hold, run once by the
orchestrator so the user can decide whether to forward-test any of them as a paper book.
Every open is appended to research/intraday/holdout_access.jsonl.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rule_baselines as rb
import voltarget_trend as vt

from profit_engine.intraday import backtest, config, data, metrics
from profit_engine.intraday.holdout import open_holdout
from profit_engine.intraday.resample import resample

REASON = (
    "Informational forward read after all 6 families FAILed promotion: vt_1h_slow4_z_s30, "
    "ema_px_4h_180, tsmom_4h_180 and buy-and-hold, to decide what (if anything) to forward-test."
)
CANDIDATES = [
    ("voltarget_trend", "vt_1h_slow4_z_s30"),
    ("rule_baselines", "ema_px_4h_180"),
    ("rule_baselines", "tsmom_4h_180"),
]


def _target(family: str, name: str, df: pd.DataFrame) -> tuple[str, pd.Series]:
    if family == "voltarget_trend":
        cfg = vt.GRID_BY_NAME[name]
        return cfg["tf"], vt.vt_target(df, cfg["tf"], cfg["params"])
    cfg = rb.GRID_BY_NAME[name]
    return cfg["tf"], rb.target_for(cfg, df)


def _line(label: str, res: backtest.BacktestResult) -> str:
    m = metrics.summarize(res)
    pnl = float((1 + res.daily).prod() - 1)
    return (
        f"  {label:<26} return {pnl:+7.1%}  Sharpe {m['sharpe']:+5.2f}  MaxDD {m['max_drawdown']:+6.1%}  "
        f"fills {int((res.bars['position'].diff().abs() > 0).sum()):>4}  exposure {m['exposure']:.0%}"
    )


def main() -> None:
    for sym in config.SYMBOLS:
        data.sync_cache(sym)
    with open_holdout(reason=REASON):
        now = pd.Timestamp.now(tz="UTC")
        for sym in config.SYMBOLS:
            raw = data.load_klines(sym, end=now)
            print(f"{sym}  holdout {config.HOLDOUT_START.date()} .. {raw.index[-1]}")
            for family, name in CANDIDATES:
                tf = (vt.GRID_BY_NAME if family == "voltarget_trend" else rb.GRID_BY_NAME)[name]["tf"]
                df = resample(raw, tf, incomplete="drop")
                _, target = _target(family, name, df)
                keep = df.index >= config.HOLDOUT_START
                for cost in (config.BASE_COST_BPS, config.STRESS_COST_BPS):
                    res = backtest.run_backtest(df.loc[keep], target.loc[df.index[keep]], cost)
                    print(_line(f"{name} @{cost:g}bps", res))
            df1h = resample(raw, "1h", incomplete="drop")
            print(_line("buy-and-hold @5bps", backtest.buy_and_hold(df1h.loc[df1h.index >= config.HOLDOUT_START])))


if __name__ == "__main__":
    main()
