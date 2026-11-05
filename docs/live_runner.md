# Signal Desk Live (local paper runner)

`src/profit_engine/live/` paper-trades intraday crypto books (`BTCUSDT`, `ETHUSDT`, long/flat
spot) on **closed** Binance bars, on your Mac, while it is awake. **Paper money only:** it
never places an order anywhere. It is separate from the Render app (`server.py`,
`paper_*.py`) and never touches Supabase: the live store is a local SQLite file, and the code
never reads `DATABASE_URL`, even when it is set.

By default (`config/live_books.json`) it runs **four books, $1,000 of paper cash each, 5 bps
per side**, two strategies on two coins, side by side (see [The research books](#the-research-books)):

| book_id | strategy | bars | dashboard label |
|---|---|---|---|
| `btc-trend-1h`, `eth-trend-1h` | Trend (vol-targeted), research config `vt_1h_slow4_z_s30` | 1h | not promoted — best research candidate (forward test) |
| `btc-lgbm-4h`, `eth-lgbm-4h` | LightGBM 4h, research config `lgbm_4h_h30_pt2_sl1` | 4h | not promoted — comparison model (backtest trailed buy-and-hold) |

Neither strategy passed the promotion gate; these books are a forward test, compared on the
dashboard with simply buying the coin and holding it. The demo strategy is still built in
but no longer in the default books. A book can also name an importable factory or a
versioned model artifact (see [Adding a promoted model](#adding-a-promoted-model)).

## Quick start

```bash
.venv/bin/pip install -e .                       # adds the signal-desk-live command
signal-desk-live train lgbm --symbol BTCUSDT     # train this month's LightGBM model (and ETHUSDT)
signal-desk-live retrain                         # train any LightGBM book's model missing for this month
signal-desk-live tick --no-notify                # check every book once, print JSON
signal-desk-live status                          # equity and position per book
signal-desk-live dashboard                       # http://127.0.0.1:8765 (Ctrl-C to stop)
signal-desk-live install --dry-run               # show the launchd files it would write
signal-desk-live install [--stay-awake-on-ac]    # run in the background at login
signal-desk-live uninstall                       # stop and remove the background agents
```

`scripts/macos/install.sh` and `uninstall.sh` do the same without the console script (they
use `.venv/bin/python` and `PYTHONPATH=src`). Install from the checkout you want to run: the
agents point at this repo's `src/` and this venv's python by absolute path.

## How one check (a "tick") works

Every 30 s (with +/-10 % jitter), for each enabled book:

| step | what happens |
|---|---|
| 1. clock | Ask Binance for its time. All "is this bar closed?" questions use the **exchange** clock, so a Mac clock that is off cannot make an unfinished bar look closed (a skew over 60 s is logged). |
| 2. new bar? | If no bar of the book's timeframe can have closed since the last one processed, skip to step 6 (a small request for the last 30 minutes of 5m bars). |
| 3. bars | Fetch recent 5m klines from `data-api.binance.vision` (`intraday.binance.fetch_rest`). A 5m bar counts as closed once `open + 5m + settle (2 s) <= exchange time`; the in-progress bar is dropped from everything except its **open**. Resample with the harness `resample(..., incomplete="drop")`. If a data gap leaves fewer than `warmup_bars` complete bars, the lookback is doubled (up to 3 times) so a warmed-up book keeps deciding on the same window the harness uses; if it still falls short, a `warming_up` event says so. |
| 4. guards | No trade, and an event instead, when: the newest complete bar ended more than 2 timeframes ago (`stale_data`, checked even when that bar was already traded, so a feed that stops publishing is never reported as up to date; the heartbeat turns "degraded"); the fill price is off by more than 25 % from the decision close (`bad_price`); the fill bar has not started yet (retry next tick); no 5m bar of the fill bar has traded yet (`halted` once a closed 5m bar shows zero volume, i.e. an exchange halt; a bar in progress with no trades yet just waits, without an event). |
| 5. decide + fill | The strategy sees the last `warmup_bars` closed bars and returns a target in [0, 1]. A stateful strategy (the trend book's band) also gets its own last target from the database (what it asked for, even if a loss pause overrode it; 0 for a new book). The target fills at the open of the **bar now in progress**, which is the next bar that exists, exactly as the harness fills: normally the bar right after the decision bar; after a bar dropped for missing 5m data (`gap`), the one after that. During a halt the fill waits until a 5m bar of the fill bar trades and then uses the fill bar's open (the harness price); a fill bar with no trading at all is skipped and the next decision supersedes it. Cost `cost_bps_per_side` (5) on the traded fraction. Claim, fill, position and events are **one SQLite transaction**; a daily-loss pause is read inside that transaction, so a pause set by another process a moment earlier is always applied. |
| 6. mark | Equity is marked at the latest closed 5m close (once per 5m bar). |
| 7. loss limit | If equity is down `max_daily_loss` (5 %) from the equity at 00:00 UTC (the position held at midnight, valued at the 23:55 bar's close): sell to cash at the next 5m open, pause until next UTC midnight, event + notification. If the next 5m bar is not served yet, later ticks in the same 5m bar re-check. While paused, decisions are still recorded (as "stay in cash"). |

This is the harness timing contract (`docs/intraday_harness.md`): decide at bar *t* close,
fill at bar *t+1* open, cost on |Δposition|, never the in-progress bar.
`tests/live/test_live_parity.py` replays a bar stream through the runner and checks that
fill times, prices, targets and costs equal `run_backtest` exactly, and equity to 1e-9;
`tests/live/test_live_guards.py` does the same with a halt or a missing 5m bar at the start of
a fill bar.

### Trading each bar at most once

`decisions` has `UNIQUE(book_id, bar_open_ts)`. `Store.commit_bar` opens `BEGIN IMMEDIATE`,
claims the bar with `INSERT ... ON CONFLICT DO NOTHING`, and in the **same** transaction
writes the fill, the new cash/units/target and any catch-up event. So:

| what happens | result |
|---|---|
| the same bar is ticked twice | second tick: claim conflicts, nothing written |
| two runners tick at once | SQLite serialises them; one claims, the other gets `already_done` |
| crash between claim and fill | impossible: both roll back together; the bar is retried next tick |
| crash after commit | the bar is done; a restart sees `last_bar_ts` and moves on |

The kill switch uses the same pattern (setting `paused_until` is its claim). WAL mode lets
the dashboard read while the runner writes; `busy_timeout` is 30 s.

## Sleep and wake

**A sleeping Mac runs nothing.** launchd does not run agents during sleep, and Binance bars
that close meanwhile are not traded. On wake, the next tick acts on the **newest closed bar
only**, fills it at the next bar's open, and records a `catch_up` event with how many decision
bars were not acted on. That count also includes bars skipped for a halt, a bad price or a
strategy error (the event says so; the earlier events say which). Bars dropped for missing
5m data are not counted: they never existed as decision bars. Missed bars are never replayed (replaying would trade at prices you could
no longer get). The loss-limit baseline is the equity at 00:00 UTC even when the Mac slept
through midnight: on the first mark of a new day the runner fetches the 23:55 bar and values
the position held at midnight at its close (the last mark before midnight stands in if that
bar is missing). A loss made between midnight and wake therefore counts against the day.

`install --stay-awake-on-ac` wraps the runner in `/usr/bin/caffeinate -s`, which stops the
Mac from sleeping **only while it is on AC power**. On battery, or with the lid closed and
no external display, macOS can still sleep. No sudo is involved.

## Where things live

| what | where | override |
|---|---|---|
| database | `~/Library/Application Support/SignalDesk/live.sqlite` | `SIGNAL_DESK_LIVE_DB` (a file path; a URL is refused) |
| books config | `~/Library/Application Support/SignalDesk/books.json` (copied there by `install` if absent), else `config/live_books.json` | `SIGNAL_DESK_BOOKS` |
| model artifacts | `~/Library/Application Support/SignalDesk/models/` (LightGBM: `lgbm_4h_h30_pt2_sl1/<SYMBOL>/<YYYY-MM>-<created>/`) | `SIGNAL_DESK_MODELS_DIR` |
| 5m history for training | `~/Library/Application Support/SignalDesk/market_data/<SYMBOL>_5m.parquet` | `SIGNAL_DESK_DATA_DIR` |
| app folder (all of the above) | `~/Library/Application Support/SignalDesk` | `SIGNAL_DESK_HOME` |
| JSON logs | `~/Library/Logs/SignalDesk/runner.jsonl` (rotated: 5 x 5 MB), `tick.jsonl` | `SIGNAL_DESK_LOGS` |
| launchd stdout/stderr | `~/Library/Logs/SignalDesk/{runner,dashboard}.{out,err}.log`: crash output only, **not rotated** (the runner agent runs `run --no-console-log`, so its JSON log is not copied here) | |
| launch agents | `~/Library/LaunchAgents/com.signaldesk.{live,dashboard,retrain}.plist` | |
| single-instance lock | `~/Library/Application Support/SignalDesk/runner.lock` (`flock`, freed if the process dies) | |

Notifications: `"notifications": false` in books.json or `SIGNAL_DESK_NOTIFY=0` turns them
off. At most one per book per 5 minutes; anything held back is summarised in the next one.

## The launchd agents

`install` writes three per-user agents and loads them with `launchctl bootstrap gui/$UID`:

* `com.signaldesk.retrain`: `python -m profit_engine.live.cli retrain` at 06:30 local time on
  the 1st and 2nd of each month (launchd runs a time missed during sleep on wake; the 2nd
  catches a Mac that was off on the 1st, and is a no-op otherwise). It has no `KeepAlive`: it
  runs, exits and waits for the next date. `install` reports it as scheduled, not running.
* `com.signaldesk.live`: `python -m profit_engine.live.cli run --no-console-log`
* `com.signaldesk.dashboard`: `python -m profit_engine.live.cli dashboard` (127.0.0.1:8765).
  It answers only requests addressed to `127.0.0.1`, `localhost` or `[::1]` (any other `Host`
  header gets a 400), so a web page cannot read it through DNS rebinding. The status line
  calls the runner late only after its own planned next check (`next_check_s` on the
  heartbeat; up to 10 minutes while Binance is unreachable) plus 2 minutes.

The runner and dashboard have `RunAtLoad` (start at login), `KeepAlive` (restart if they exit),
`ThrottleInterval` 30 s, `WorkingDirectory` = the repo, and logs in
`~/Library/Logs/SignalDesk/`. `uninstall` runs `launchctl bootout` and deletes the plists;
your database, models and logs are kept. Nothing is loaded unless you run `install` yourself.
`install` also copies `config/live_books.json` to `~/Library/Application Support/SignalDesk/books.json`
when there is no file there, so you have one to edit. An existing file is never overwritten:
`install --use-default-books` replaces it, after copying it to `books.json.bak-<time>`.
`install --no-dashboard` also stops and removes a dashboard agent left by an earlier full
install. `SIGNAL_DESK_*` path variables set at install time are written into the agents as
absolute paths (the launchd log files too), so the agents and your terminal open the same
files. `ExitTimeOut` is 30 s: on SIGTERM the runner stops retrying Binance at once, so only a
request already in flight (10 s timeout) delays the clean exit.

`install` waits 5 s after loading and checks each agent with `launchctl print`. It says
"Installed and started ... it is running" only if the agent is running then; otherwise it
says the agent is not staying up and exits 1. launchctl's own messages go to
`~/Library/Logs/SignalDesk/install.jsonl`, not to the terminal.

A second runner (say a manual `signal-desk-live run` while the agent is loaded) waits for
the lock instead of exiting, and the dashboard waits for port 8765 to be free: neither turns
`KeepAlive` into a relaunch every 30 s.

### If the agents do not start

UNVERIFIED (not yet tried on a real login): the agents run this venv's python from inside
`~/Documents`, which macOS privacy protection (TCC) can block for launchd-started programs.
`install` now reports this case ("not staying up") instead of claiming success.
If `runner.err.log` shows `Operation not permitted` or missing packages, either give the
python binary the venv points to (`readlink -f .venv/bin/python`) Full Disk Access in System
Settings > Privacy & Security, or clone the repo outside `~/Documents`, `~/Desktop` and
`~/Downloads` and install from there.

## books.json

The default is `config/live_books.json` (the four books above). Entry shapes:

```json
{
  "cost_bps_per_side": 5.0, "poll_seconds": 30, "max_daily_loss": 0.05, "starting_equity": 1000,
  "books": [
    {"book_id": "btc-trend-1h", "title": "BTC trend, vol-targeted (1h)", "strategy": "voltarget_trend",
     "symbol": "BTCUSDT", "timeframe": "1h", "params": {"config": "vt_1h_slow4_z_s30"}},
    {"book_id": "btc-lgbm-4h", "title": "BTC LightGBM (4h)", "strategy": "lgbm",
     "symbol": "BTCUSDT", "timeframe": "4h", "params": {"config": "lgbm_4h_h30_pt2_sl1"}},
    {"book_id": "btc-demo-4h", "strategy": "demo_sma_trend", "symbol": "BTCUSDT", "timeframe": "4h",
     "params": {"fast": 10, "slow": 30}},
    {"book_id": "btc-model", "model": "my_model_folder"},
    {"book_id": "eth-custom", "strategy": "my_pkg.strategies:make", "symbol": "ETHUSDT", "timeframe": "1h"}
  ]
}
```

All settings: `cost_bps_per_side`, `poll_seconds`, `jitter`, `max_daily_loss` (0 turns it
off), `stale_after_bars`, `max_price_jump`, `settle_seconds`, `starting_equity`,
`notifications`, `notify_min_interval_s`. Per book: `title`, `enabled`, `starting_equity`,
`cost_bps_per_side`, `max_daily_loss`. Numbers must be JSON numbers within range (for example
`poll_seconds` 5 to 3600, `settle_seconds` 0 to 120, `max_daily_loss` 0 to 0.99) and
`notifications` / `enabled` must be `true` or `false` without quotes (`registry.LIMITS`). The
file is re-read every loop; a broken edit, including a wrongly typed value, keeps the last
good settings and logs a `config_error` event. A new `book_id` starts a new book
with fresh paper cash; renaming one starts over (the old book stays in the database).
Changing an existing book's `symbol` or `timeframe` is refused: that book is skipped with a
`config_conflict` event (its coins would otherwise be priced in the other market). Use a new
`book_id` for the new setup. `"enabled": false` stops checking a book and the dashboard
shows it as "Turned off".

`demo_sma_trend` is **demo — not promoted**: long when the fast moving average of closes is
above the slow one. It exists to exercise the runner, and the dashboard says so. It is no
longer in the default books.

## The research books

Code: `src/profit_engine/live/strategies/`. Both strategies are **ports** of the research code
in `experiments/intraday/` (which is not a package and is never imported at runtime); every
function's docstring names the experiment function it mirrors. The parity tests import the
experiment modules for comparison only and skip, saying how to build it, when the research
cache `data/cache/intraday/` is absent.

### Trend (vol-targeted), `voltarget_trend`

`vt_1h_slow4_z_s30` (`voltarget.py`): on 1h bars, for lookbacks of 7, 14, 30 and 60 days take
`clip(log return / (EWMA vol x sqrt(n)), 0, 1)`, average them, scale by
`min(1, 30% / annualised EWMA vol)` (vol half-life 3 days), round to 5% steps, and move the
target only when the new value is more than 20 points from the current one (a value that
rounds to 0 exits at once). Fractional targets: the book can hold, say, 35% BTC.

* **State.** The band is path dependent. The runner passes the strategy its own last target
  from the `decisions` table (`Store.strategy_target`), and the strategy applies ONE step of
  the band to the newest bar. It never re-runs the band over its window. A new book starts
  flat, like the research did before its first signal; until the band first moves it can
  hold a different amount than the research path that started in 2017 would.
* **Warm-up: 3600 bars (150 days).** The EWMA vol has infinite memory. Measured 2026-11-05 on
  the cached 1h bars (300 random bars per symbol), the worst relative difference between a
  window's raw exposure and the full-history one was 1.5e-09 at 2400 bars, 6e-10 at 2600,
  3e-11 at 3000 and 4e-14 at 3500. `tests/live/test_live_trend_parity.py` re-measures it on
  every bar of 2025-01..2026-06 (must stay < 1e-9).
* **Parity** (`test_live_trend_parity.py`): replaying every 1h bar of 2025-01..2026-06 bar by
  bar, with the strategy's own output fed back (the research state is used once, as the
  starting state), gives exactly the research target; so does a replay from the first cached
  bar starting flat with no research state (2017-09..2019-12); and
  `test_live_runner_trend.py` runs the book through `tick()` hourly on cached bars and gets
  the same fills, prices and targets as `run_backtest`, costs and equity to 1e-9.

### LightGBM 4h, `lgbm`

`lgbm_4h_h30_pt2_sl1` (`lgbm_core.py`, `lgbm_artifacts.py`, `lgbm.py`): long for the next 4h
bar when the model's probability that a long entry hits +2 sigma before -1 sigma within 30
bars (and beats a round trip of costs) is above the month's threshold; flat otherwise, and
flat on any bar where a feature is missing (no matching bar of the other coin, too little
history), as the research did. Features use both coins (cross-asset returns).

* **Training** (`signal-desk-live train lgbm --symbol BTCUSDT [--month YYYY-MM]`): the research
  walk-forward fold for that month, on every closed bar from 2017-09-01 up to the first
  instant of the month (UTC): same features, triple-barrier labels, uniqueness weights, model
  parameters (seed 7, `deterministic=True`, 4 threads), and threshold from a purged 4-fold
  inner CV inside the training window. Training refuses to run unless the 5m history reaches
  past that instant. The 5m history lives in `market_data/` (synced with the harness's
  `sync_cache`; the first sync copies the research cache instead of downloading 9 years).
  The research label and inner-CV functions refuse bars after the research holdout
  (2026-07-01); the ports in `lgbm_core.py` are the same code without that research guard,
  because a production model must train on last month's bars. Nothing is written to the
  research holdout audit log.
* **Artifacts**: `models/lgbm_4h_h30_pt2_sl1/<SYMBOL>/<YYYY-MM>-<created UTC>/` holds
  `model.txt` (LightGBM's text format: no pickle) and `metadata.json` (config, model
  parameters, feature columns, threshold and its quantile, labels, training window, data hash
  of both coins' 5m bars, git sha, lightgbm / scikit-learn / numpy / pandas versions, the
  model file's SHA256). A new artifact is written to a hidden temporary folder and renamed
  into place only after it trained, found a threshold, and reloaded with the same features
  and trees. Books load the newest folder.
* **Refusals.** A book refuses to decide, keeps its position and shows "No usable model" on
  the dashboard when there is no artifact, the model file does not match its SHA256, or the
  installed library versions differ from the recorded ones. Retrain to fix it.
* **Monthly retrain** (`signal-desk-live retrain`, run by `com.signaldesk.retrain`): for each
  LightGBM book, trains the current month's model if it does not exist yet. If training fails,
  or no threshold made money (the research would trade flat that month), the previous model
  stays in use and an event says so. The month is judged on this Mac's clock, in UTC.
* **Warm-up: 3000 bars (500 days).** `emadist_180` is an EWM with infinite memory. Measured
  2026-11-05 (150 random bars per symbol): worst relative feature difference 1.7e-07 at 2000
  bars, 1.3e-09 at 2500, 5e-10 at 3000 (there a rolling-sum rounding difference, not the EWM).
* **Parity** (`test_live_lgbm_parity.py`): the real `train` command for 2026-06 on the cached
  bars gives the research fold's threshold bitwise and, through the saved text model, the
  research model's probabilities bitwise; the live book's decision on every June 2026 4h bar,
  computed from a 3000-bar window, equals the research target, with the same probability.

### Comparing with buy and hold

The dashboard groups the books by strategy, with the status label on each card. Each card
also shows what the $1,000 would be worth had it bought the coin at the first price the book
saw and held it (before fees), how far the book is ahead or behind that, and draws it as a
dashed line on the chart.

### Fetching history

The trend books need 150 days of 5m bars and the LightGBM books 500 days (both coins). The
first check after the runner starts downloads that from Binance (about 150 requests of 1000
bars per coin; the first `tick` of all four books took 2 min 7 s on 2026-11-05); after that
the in-memory cache keeps the widest span any book asked for and only the newest bars are
fetched. The first fetch asks for `warmup_bars` plus
2%, so a few missing bars do not force a second, doubled download.

## Adding a promoted model

1. Get a PASS from `promotion.evaluate` (and the holdout check) in the research harness.
2. Create `~/Library/Application Support/SignalDesk/models/<name>/` with the fitted model
   (`joblib.dump(model, "model.joblib")`) and a `metadata.json`:

   ```json
   {
     "name": "lgbm_v3", "version": "2026-11-20", "symbol": "BTCUSDT", "timeframe": "1h",
     "warmup_bars": 300, "other_symbol": "ETHUSDT",
     "kind": "proba_threshold", "threshold": 0.55,
     "feature_params": {}, "feature_columns": ["ret_1", "rv_48", "x_ret_12"],
     "artifact": "model.joblib", "artifact_sha256": "<sha256sum of model.joblib>",
     "promoted": true, "promotion": {"verdict": "PASS", "trials_ref": "research/intraday/trials/..."}
   }
   ```

   `proba_threshold` builds features with `intraday.features.make_feature_frame` on the
   last `warmup_bars` closed bars (plus `other_symbol` on the same bars) and goes long when
   `predict_proba[:, 1] > threshold`. For anything else use `"kind": "python"` with
   `"entry": "package.module:factory"`; `factory(metadata)` returns an object with `name`,
   `symbol`, `timeframe`, `warmup_bars`, `other_symbol` and `decide(bars, other) -> float |
   Decision` (see `live/strategy.py`).
3. Add a book with `"model": "<name>"` to books.json. The next loop picks it up.

A book is labelled **promoted** only when `promoted` is `true` and `promotion.verdict` is
`"PASS"`; anything else shows as "demo — not promoted". The SHA256 is checked on every load
and a mismatch refuses the book. Loading a joblib file runs pickled code: only put
artifacts you built yourself in the models folder. Keep `warmup_bars` large enough for
every feature window, and check parity first: `strategy.target_series(strategy, bars)`
replays it exactly as the runner decides, so `run_backtest(bars, target_series(...))` is a
backtest of the live code path.

## Known limitations

* The research books differ from their backtests where the runner differs from the
  research: a new trend book starts flat; bars missed while the Mac sleeps are not replayed,
  so the band can follow a different path afterwards; the daily loss limit (5 %, on by
  default) can force a sale the backtest never made (the trend strategy keeps its own state
  through the pause and the book returns to it the next bar); a LightGBM month with no
  profitable threshold keeps last month's model where the research stayed in cash.
* A `signal-desk-live tick` from the command line downloads the full warm-up history every
  time (the cache lives only as long as the process).

* Paper fills at the bar open with a flat cost: no spread, slippage or partial fills.
* A fill is booked at the next bar's open even when the runner gets to it late (up to one
  bar later, e.g. just after wake). The decision only uses data up to the decision bar's
  close, but the fill price is from up to one timeframe earlier than the moment of booking.
* Halts: like the harness, a fill is skipped only when the whole fill bar has no volume. While
  the fill bar's 5m bars show no trading, the runner waits; once one trades it books the fill
  at the fill bar's open (the harness price), so that fill is booked up to a few minutes late
  at the earlier, possibly stale, price. Tested against `run_backtest` for a halt of the first
  5m bar and of the whole bar.
* Gaps: the harness drops a bar with any missing 5m data and fills at the next bar that
  exists; the runner does the same when the missing 5m bar is the first one of the fill bar
  (tested), or any earlier one. A 5m bar missing LATER in a fill bar is invisible when the
  runner fills at that bar's open, so there the harness (which skips the whole bar) and the
  runner differ. Parity is exact on gapless data.
* The daily loss limit is checked on 5m closes while the Mac is awake; a loss that happens
  during sleep is seen (and acted on) only after wake.
