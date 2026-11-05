# Signal Desk Live (local paper runner)

`src/profit_engine/live/` paper-trades intraday crypto books (`BTCUSDT`, `ETHUSDT`, long/flat
spot) on **closed** Binance bars, on your Mac, while it is awake. **Paper money only:** it
never places an order anywhere. It is separate from the Render app (`server.py`,
`paper_*.py`) and never touches Supabase: the live store is a local SQLite file, and the code
never reads `DATABASE_URL`, even when it is set.

The runner does not care which strategy wins the research. A book names a strategy: the
built-in demo, an importable factory, or a versioned model artifact (see
[Adding a promoted model](#adding-a-promoted-model)).

## Quick start

```bash
.venv/bin/pip install -e .                       # adds the signal-desk-live command
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
| 3. bars | Fetch recent 5m klines from `data-api.binance.vision` (`intraday.binance.fetch_rest`). A 5m bar counts as closed once `open + 5m + settle (2 s) <= exchange time`; the in-progress bar is dropped from everything except its **open**. Resample with the harness `resample(..., incomplete="drop")`. |
| 4. guards | No trade, and an event instead, when: the newest complete bar ended more than 2 timeframes ago (`stale_data`); the newest bar is missing 5m data (`gap`; the book waits for the next complete bar rather than trade an older one at a past price); the fill price is off by more than 25 % from the decision close (`bad_price`); the next bar has not started yet (retry next tick). |
| 5. decide + fill | The strategy sees the last `warmup_bars` closed bars and returns a target in [0, 1]. The target fills at the **next bar's open** (the open of the bar now in progress, fixed once it starts) with `cost_bps_per_side` (5) on the traded fraction. Claim, fill, position and events are **one SQLite transaction**. |
| 6. mark | Equity is marked at the latest closed 5m close (once per 5m bar). |
| 7. loss limit | If equity is down `max_daily_loss` (5 %) from the first mark of the UTC day: sell to cash at the next 5m open, pause until next UTC midnight, event + notification. While paused, decisions are still recorded (as "stay in cash"). |

This is the harness timing contract (`docs/intraday_harness.md`): decide at bar *t* close,
fill at bar *t+1* open, cost on |Δposition|, never the in-progress bar.
`tests/live/test_live_parity.py` replays a bar stream through the runner and checks that
fill times, prices, targets and costs equal `run_backtest` exactly, and equity to 1e-9.

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
bars were skipped. Missed bars are never replayed (replaying would trade at prices you could
no longer get). The first mark after midnight UTC sets that day's loss-limit baseline, so a
Mac that sleeps through midnight starts the day's limit from its first mark after waking.

`install --stay-awake-on-ac` wraps the runner in `/usr/bin/caffeinate -s`, which stops the
Mac from sleeping **only while it is on AC power**. On battery, or with the lid closed and
no external display, macOS can still sleep. No sudo is involved.

## Where things live

| what | where | override |
|---|---|---|
| database | `~/Library/Application Support/SignalDesk/live.sqlite` | `SIGNAL_DESK_LIVE_DB` (a file path; a URL is refused) |
| books config | `~/Library/Application Support/SignalDesk/books.json`, else `config/live_books.example.json` | `SIGNAL_DESK_BOOKS` |
| model artifacts | `~/Library/Application Support/SignalDesk/models/<name>/` | `SIGNAL_DESK_MODELS_DIR` |
| app folder (all of the above) | `~/Library/Application Support/SignalDesk` | `SIGNAL_DESK_HOME` |
| JSON logs | `~/Library/Logs/SignalDesk/runner.jsonl` (rotated), `tick.jsonl` | `SIGNAL_DESK_LOGS` |
| launchd stdout/stderr | `~/Library/Logs/SignalDesk/{runner,dashboard}.{out,err}.log` | |
| launch agents | `~/Library/LaunchAgents/com.signaldesk.{live,dashboard}.plist` | |
| single-instance lock | `~/Library/Application Support/SignalDesk/runner.lock` (`flock`, freed if the process dies) | |

Notifications: `"notifications": false` in books.json or `SIGNAL_DESK_NOTIFY=0` turns them
off. At most one per book per 5 minutes; anything held back is summarised in the next one.

## The launchd agents

`install` writes two per-user agents and loads them with `launchctl bootstrap gui/$UID`:

* `com.signaldesk.live`: `python -m profit_engine.live.cli run`
* `com.signaldesk.dashboard`: `python -m profit_engine.live.cli dashboard` (127.0.0.1:8765)

Both have `RunAtLoad` (start at login), `KeepAlive` (restart if they exit),
`ThrottleInterval` 30 s, `WorkingDirectory` = the repo, and logs in
`~/Library/Logs/SignalDesk/`. `uninstall` runs `launchctl bootout` and deletes the plists;
your database and logs are kept. Nothing is loaded unless you run `install` yourself.

## books.json

```json
{
  "cost_bps_per_side": 5.0, "poll_seconds": 30, "max_daily_loss": 0.05, "notifications": true,
  "books": [
    {"book_id": "btc-demo-4h", "title": "BTC demo trend (4h)", "strategy": "demo_sma_trend",
     "symbol": "BTCUSDT", "timeframe": "4h", "params": {"fast": 10, "slow": 30}},
    {"book_id": "btc-lgbm", "title": "BTC LightGBM (1h)", "model": "lgbm_v3"},
    {"book_id": "eth-custom", "strategy": "my_pkg.strategies:make", "symbol": "ETHUSDT", "timeframe": "1h"}
  ]
}
```

All settings: `cost_bps_per_side`, `poll_seconds`, `jitter`, `max_daily_loss` (0 turns it
off), `stale_after_bars`, `max_price_jump`, `settle_seconds`, `starting_equity`,
`notifications`, `notify_min_interval_s`. Per book: `title`, `enabled`, `starting_equity`,
`cost_bps_per_side`, `max_daily_loss`. The file is re-read every loop; a broken edit keeps
the last good settings and logs a `config_error` event. A new `book_id` starts a new book
with fresh paper cash; renaming one starts over (the old book stays in the database).

`demo_sma_trend` is **demo — not promoted**: long when the fast moving average of closes is
above the slow one. It exists to exercise the runner, and the dashboard says so.

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

* Paper fills at the bar open with a flat cost: no spread, slippage or partial fills.
* A fill is booked at the next bar's open even when the runner gets to it late (up to one
  bar later, e.g. just after wake). The decision only uses data up to the decision bar's
  close, but the fill price is from up to one timeframe earlier than the moment of booking.
* The harness defers fills on halted (zero-volume) bars; the runner cannot know whether the
  in-progress bar will end up with zero volume and does not.
* On data with gaps, the harness fills a decision at the next bar that exists, while the
  runner (which cannot see the future) uses the next bar's open as soon as it starts. Parity
  is exact on gapless data.
* The daily loss limit is checked on 5m closes while the Mac is awake; a loss that happens
  during sleep is seen (and acted on) only after wake.
