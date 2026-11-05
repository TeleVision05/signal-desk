# Intraday crypto research harness

`src/profit_engine/intraday/` is a research harness for **intraday, long/flat, spot**
strategies on Binance GLOBAL `BTCUSDT` and `ETHUSDT`. Its job is to make it hard for a
backtest to lie: fills happen at the next bar's open, labels carry end times for purging,
the holdout is sealed, every configuration tried is counted, and promotion requires
clearing a fixed list of criteria on both symbols.

Nothing here trades. It never touches the paper books (`paper_ml*.py`, `server.py`).

## Setup (once per machine)

```bash
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/pre-commit install
.venv/bin/python -m profit_engine.intraday.sanity    # builds data/cache/intraday/*.parquet
```

`sanity` seeds the cache from already-verified dumps when present, otherwise downloads
the SHA256-verified monthly archives from `data.binance.vision`, then tops up to the
latest **closed** 5m bar from `data-api.binance.vision`. It prints a gap report and a
buy-and-hold line per symbol and timeframe. Run it **once** before launching parallel
agents; `load_klines` only reads the cache, so agents never race on the download.
`--offline` re-prints the summary without touching the network.

## Module map

| module | what you call | contract |
|---|---|---|
| `data` | `load_klines(sym, start=TRAIN_START, end=HOLDOUT_START)` | index = bar OPEN time (UTC); closed bars only; no fabricated bars (gaps stay gaps) |
| `resample` | `resample(df, "15m"/"1h"/"4h", incomplete=...)` | `label='left', closed='left'`, epoch-anchored. A bar missing any 5m sub-bar raises `IncompleteBarError` unless you pass `incomplete="drop"` or `"flag"` |
| `features` | `make_feature_frame(primary, other, params=...)`, or each function | causal (value at `t` uses bars `<= t`), pure. Windows are in bars of the frame's timeframe |
| `labels` | `forward_return(df, h)`, `triple_barrier(df, h, pt, sl, vol_window)` | start at `open[t+1]`; return `t1` = when the label is known (for purging) |
| `splits` | `walk_forward_splits(index, t1, ...)`, `purged_kfold`, `purged_tail_split` | monthly test blocks over 2020-01..2026-06. `t1` is REQUIRED (`None` raises unless `allow_unpurged=True`). Rows before a block are purged by `t1`; rows after it must start at or after the block's latest `t1` plus the embargo |
| `backtest` | `run_backtest(df, target, cost_bps_per_side)`, `buy_and_hold`, `vol_matched_buy_and_hold` | `target[t]` decided at `t` close, filled at `t+1` open, P&L open-to-open. `target` must be a float or a Series on the bars' own timestamps |
| `metrics` | `summarize(result)` | daily-P&L Sharpe (sqrt 365), Sortino, CAGR, MaxDD, PF, t-stats, months, years |
| `overfitting` | `dsr_from_daily`, `pbo_cscv`, `permutation_test` | per-period (daily) Sharpe throughout |
| `trials` | `log_trial(...)`, `n_trials()`, `returns_matrix()` | append-only JSONL + one parquet per call; counts RUNS (config x seed x code x data x returns); series must cover the whole OOS window |
| `promotion` | `evaluate(Candidate)`, `verdict(table)`, `holdout_check(...)` | PASS only if every row passes on both symbols |
| `holdout` | `open_holdout(reason=...)` | the only way past 2026-07-01; audited |

## Timing contract (read this before writing a strategy)

```
bar t:   [open_t ............ close_t]  <- features use everything up to close_t
                                         <- your target[t] is decided here
bar t+1: [open_{t+1} ...]                <- filled here, at open_{t+1}
P&L of target[t] = open_{t+2} / open_{t+1} - 1   (minus cost on |Δposition|)
```

* Targets must be in `[0, 1]` (long/flat; `ValueError` otherwise). `NaN` means flat.
* A fill happens only when the target changes. Cost is `cost_bps_per_side / 1e4 * |target - drifted weight| * equity`
  at each fill (for 0/1 targets that is `|Δposition|`). Base 5, stress 10.
* The last bar has no next open, so the book is forced flat there (exit cost charged).
* `max_round_trips_per_day` and `min_hold_bars` are applied to executed positions.
* Between fills the book holds shares, so a fractional weight drifts with the price
  (`result.bars["weight"]`). A constant 0.5 target is a true 50% hold, not a free
  5-minute rebalance. Vol-targeting strategies that change the target every bar pay for
  every rebalance; round your targets if you do not want that.
* Target timestamps must be bar open times: a tz-naive, shifted or integer-indexed target
  raises rather than silently becoming a flat book.
* Data gaps are flagged, not hidden: `result.bars["span_bars"]` (> 1 across a hole), trade
  columns `crosses_gap` and `holding_time`, and `result.params["bars_held_across_gap"]`.
* Halted bars (`volume == 0`, open = stale previous close) are not tradeable: a fill due on
  one is deferred to the next bar with volume, and the earlier position is carried (it earns
  the halt move it was really exposed to). See `params["n_stale_bars"]`,
  `params["fills_deferred"]` and the trade column `crosses_stale`. A frame without a
  `volume` column gets no halt detection (`params["stale_check"]` is False).
* The bar index must be a tz-aware, strictly increasing DatetimeIndex. An unsorted or
  duplicated frame (an unsorted `concat`) raises instead of inventing P&L.
* Days are UTC calendar days; every day in the span is in `result.daily` (0 when flat).

## End-to-end example

`python -m profit_engine.intraday.example` runs the whole loop with a logistic regression
on 1h bars and prints the promotion table (it writes to a throwaway registry; it FAILs).
The core of it:

```python
from profit_engine.intraday import backtest, config, data, features, labels, metrics, promotion, splits, trials
from profit_engine.intraday.provenance import data_hash
from profit_engine.intraday.resample import resample

SEED, TF, FAMILY = 7, "1h", "logreg_v1"
PARAMS = {"C": 0.1, "horizon": 4, "threshold": 0.55}

raw = {s: data.load_klines(s) for s in config.SYMBOLS}              # [2017-09-01, 2026-07-01)
bars = {s: resample(df, TF, incomplete="drop") for s, df in raw.items()}

def oos_positions(df, other):
    x = features.make_feature_frame(df, other, params=FEATURE_PARAMS)  # aligned on common bars
    df = df.loc[x.index]
    lab = labels.forward_return(df, PARAMS["horizon"])                  # ret + t1
    y = (lab["ret"] > 0).astype(int)
    has_x, has_y = x.notna().all(axis=1).to_numpy(), lab["ret"].notna().to_numpy()
    target = pd.Series(0.0, index=df.index)
    for fold in splits.walk_forward_splits(df.index, lab["t1"]):        # purged, monthly
        train, test = fold.train & has_x & has_y, fold.test & has_x
        model = make_model(seed=SEED).fit(x[train], y[train])           # tune ONLY on x[train]
        target[test] = (model.predict_proba(x[test])[:, 1] > PARAMS["threshold"]).astype(float)
    return target

for sym in config.SYMBOLS:
    target = oos_positions(bars[sym], bars["ETHUSDT" if sym == "BTCUSDT" else "BTCUSDT"])
    oos = (target.index >= config.EVAL_START) & (target.index < config.EVAL_END)
    oos_bars, oos_target = bars[sym].loc[target.index[oos]], target[oos]
    for cost in (config.BASE_COST_BPS, config.STRESS_COST_BPS):
        res = backtest.run_backtest(oos_bars, oos_target, cost)
        trials.log_trial(FAMILY, PARAMS, TF, sym, cost, metrics.summarize(res), res.daily,
                         data_hash(raw[sym]), SEED)

table = promotion.evaluate(promotion.Candidate(FAMILY, PARAMS, TF, base=..., stress=..., bars=...))
print(table.to_string(index=False), promotion.verdict(table))
```

## Rules for experiment agents

1. **Never touch the holdout.** Data at or after `2026-07-01 00:00 UTC` raises
   `HoldoutViolation` in `load_klines`, both label functions, all three splitters,
   `run_backtest`, every public `metrics` helper, `dsr_from_daily` / `pbo_cscv` (DataFrame) /
   `permutation_test`, `log_trial` and `promotion`. A result built inside `open_holdout`
   still raises when analysed after the block closes. The guard fails closed: a frame or
   series whose index is not a DatetimeIndex (e.g. after `reset_index`) raises `TypeError`.
   Even inside `open_holdout`, `log_trial` refuses any series reaching `EVAL_END` and
   `promotion.evaluate` refuses to run: holdout results go only to `holdout_check`. Raw plumbing (`binance.*`,
   `sync_cache`, `resample`, `features`) is not guarded; its output reaches a score only
   through a guarded call.
   `open_holdout(reason=...)` exists for ONE final check of a candidate that already PASSed
   promotion, run by the human or orchestrator, never by an experiment agent. Every open is
   appended to `research/intraday/holdout_access.jsonl` with the reason and caller. The path
   is not a parameter (tests monkeypatch `config.HOLDOUT_AUDIT_LOG`).
2. **Log every run you evaluate**, winners and losers, at base cost and at stress cost:
   `trials.log_trial(...)`. The DSR deflates by the distinct-RUN count across ALL families
   and kinds (benchmarks included), where a run is (family, params, timeframe, symbol, seed,
   git sha, data hash) plus a hash of its logged returns: a new seed, a code change (even in
   an untracked script) or new data is a new trial; an identical re-run is not, and neither
   is re-scoring the same spec at stress cost. Each run keeps its own return series for PBO.
   Still commit experiment code, so a run can be reproduced. The series must cover the OOS
   window `[EVAL_START, EVAL_END)` to within 7 days at each end; a partial, earlier or later
   series raises. The registry computes each run's Sharpe from the saved returns; the
   `sharpe` / `sharpe_daily` you pass in `metrics` are overwritten. An unlogged run makes
   every other result look better than it is.
3. **One family per agent** (e.g. `rule_baselines`, `logreg`, `lgbm`, `meta_label`,
   `voltarget_trend`). Each family is its own JSONL file, so parallel agents never append
   to the same file. Rule baselines log with `kind="baseline"`; promotion requires beating
   the best of them, so the baseline agent should run first.
4. **Fixed seeds.** Pass `random_state`/`seed` everywhere and log it.
5. **Tune only inside training windows.** Thresholds, hyper-parameters, feature choices:
   select them with `purged_kfold` / `purged_tail_split` on the rows of `fold.train`, never
   by looking at OOS results. Choosing a threshold from the OOS table is a new trial; log it.
6. **Do not drop incomplete bars silently.** Use `incomplete="drop"` knowingly (the sanity
   run shows how many) or `"flag"` and handle them.
7. **Backtest the stitched OOS positions once** over `[EVAL_START, EVAL_END)`, not per fold.
8. Do not edit or delete registry lines. Do not commit `trials/returns/` (gitignored).

## Promotion criteria (`promotion.evaluate`)

Walk-forward OOS 2020-01-01..2026-06-30, each of BTCUSDT and ETHUSDT separately:

| criterion | threshold |
|---|---|
| net Sharpe @ 5 bps/side | >= 1.0 |
| net Sharpe @ 10 bps/side | > 0 |
| DSR (N = distinct runs in the whole registry, variance of run daily Sharpes) | >= 0.95 |
| PBO, CSCV with 16 blocks over the registry's base-cost return matrix for the symbol | <= 0.20 |
| OOS trades | >= 500 |
| t-stat of mean net bps/trade, or of daily P&L | >= 3 |
| profit factor (trade level) | >= 1.2 |
| max drawdown | >= -25% |
| positive months | >= 60% |
| worst calendar year | >= -10% |
| best month share of total (log) P&L | < 30% |
| Sharpe beats buy-and-hold, vol-matched buy-and-hold, best `kind="baseline"` trial | > |
| random-timing permutation p (same trades and gaps, shuffled order) | < 0.05 |
| backtest starts in [2020-01-01, +7 days] and ends in [2026-06-24, 2026-07-01) | yes |
| stress run holds the base positions on the same bars; `bars` equals the base bars | yes |
| costs are 5 and 10, config is logged | yes |

Holdout (`promotion.holdout_check`, only inside `open_holdout`): net P&L > 0 and Sharpe
>= 0.5 at 5 bps/side, both symbols, holdout-only data.

## Data facts (measured 2026-11-05)

| fact | value |
|---|---|
| history | 2017-08-17 04:00 .. latest closed bar, 967,880 bars per symbol |
| missing 5m bars | 1,944 (0.20%) in 32 gaps; largest 2018-02-08 00:25 .. 2018-02-10 06:15 (645 bars) |
| off-grid bars dropped | 241 per symbol (2018 restart timestamps like `09:58:14.789`) |
| incomplete resampled bars 2017-09..2026-06 | 15m: 12, 1h: 28, 4h: 48 |
| zero-volume (halted) 5m bars | BTC 493, ETH 1,448 before 2026-07-01 (42 each in 2020-01..2026-06); 932 / 1,817 through 2026-11-05 per the sanity summary (audit round 2; re-check: `(load_klines(sym)["volume"] <= 0).sum()`) |
| buy-and-hold 2020-01..2026-06 @5bps | BTC Sharpe 0.84, MaxDD -77%; ETH Sharpe 0.89, MaxDD -79% |

## Known limitations

* Fills are at the bar open with a flat cost; no spread model, slippage curve or queue.
* Rebalances are full: any change of target re-sets the weight to the target and pays for
  the whole distance from the drifted weight (no partial or banded rebalancing).
* The permutation test measures timing skill only; a high-turnover strategy can pass it and
  still lose money after costs (other criteria catch that).
* PBO needs at least 2 logged trials for the symbol and uses only the dates all trials share
  (every logged series covers the window to within 7 days, so this trims at most the edges).
* The holdout guard prevents accidents; it is not a security boundary.
