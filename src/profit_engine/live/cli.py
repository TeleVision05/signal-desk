"""``signal-desk-live run|tick|status|dashboard|train|retrain|install|uninstall``."""

from __future__ import annotations

import argparse
import filecmp
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from profit_engine.live import macos, paths, queries, training
from profit_engine.live.app import serve
from profit_engine.live.market import BinanceBarSource
from profit_engine.live.notify import Notifier
from profit_engine.live.registry import ConfigError, load_config
from profit_engine.live.runner import tick
from profit_engine.live.service import run_forever, setup_logging
from profit_engine.live.store import Store, StoreError

PAPER = "PAPER MONEY ONLY - no real orders are ever placed."


def _config_or_exit():

    try:
        return load_config()
    except (ConfigError, ValueError, OSError) as exc:
        print(f"Could not load the books settings ({paths.books_path()}): {exc}", file=sys.stderr)
        return None


def cmd_run(args: argparse.Namespace) -> int:

    setup_logging("runner", to_stderr=not args.no_console_log)
    return run_forever()


def cmd_tick(args: argparse.Namespace) -> int:

    if args.no_notify:
        os.environ["SIGNAL_DESK_NOTIFY"] = "0"
    setup_logging("tick", to_stderr=False)
    config = _config_or_exit()
    if config is None:
        return 2
    store = Store(paths.db_path())
    notifier = Notifier(config.settings.notifications, config.settings.notify_min_interval_s)
    try:
        result = tick(config, store, BinanceBarSource(), notifier)
    finally:
        store.close()
    print(json.dumps({"paper": True, "db": str(store.path), "config": str(config.source), **result}, indent=2))
    return 0 if result["ok"] else 1


def _status_line(book: dict) -> str:
    pos = f"{book['units']:.6f} {book['coin']}" if book["units"] > 0 else "cash"
    last = book["last_decision"]["bar_open_ts"] if book["last_decision"] else "none yet"
    paused = f", paused until {book['paused_until']}" if book["paused_until"] else ""
    return (
        f"- {book['title']} [{book['label']}] {book['symbol']} {book['timeframe']}: "
        f"equity ${book['equity']:,.2f} ({book['return_pct']:+.2f}%), holding {pos}, last bar {last}{paused}"
    )


def cmd_status(_: argparse.Namespace) -> int:

    db = paths.db_path()
    Store(db).close()
    ov = queries.overview(db)
    print(PAPER)
    print(f"Database: {db}")
    beat = ov["heartbeat"].get("runner")
    print(f"Runner: last checked {beat['age_s']:.0f}s ago ({beat['status']})" if beat else "Runner: never ran")
    for book in ov["books"]:
        print(_status_line(book))
    if not ov["books"]:
        print("No books yet: run `signal-desk-live tick` once, or install the background runner.")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:

    serve(port=args.port)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train (and publish) one LightGBM model, even if one exists for that month."""
    setup_logging("train", to_stderr=False)
    now = pd.Timestamp.now(tz="UTC")
    cutoff = training.art.month_start(pd.Timestamp(args.month, tz="UTC") if args.month else now)
    data_dir = Path(args.data_dir) if args.data_dir else paths.market_data_dir()
    coin = args.symbol.removesuffix("USDT")
    print(f"Training the {coin} LightGBM model for {cutoff:%B %Y} on closed bars before {cutoff:%Y-%m-%d} UTC...")
    try:
        frames = training.load_5m(data_dir, sync=not args.no_sync, now=now)
        folder = training.train_and_publish(frames, args.symbol, cutoff, paths.models_dir(), args.config)
    except training.art.NoThreshold:
        print(f"No setting made money in training, so no model was saved and the book stays in cash for "
              f"{cutoff:%B %Y}, as the backtest did.", file=sys.stderr)  # fmt: skip
        return 1
    except training.DataNotReady as exc:
        print(f"The price history is not complete yet ({exc}). Nothing was saved; try again later.", file=sys.stderr)
        return 1
    except Exception:  # a download or training failure: plain words here, the detail in train.jsonl
        logging.getLogger("signal_desk.training").exception("train failed")
        print(f"Training did not finish, so nothing was saved and any earlier model is kept. The details are in "
              f"{paths.logs_dir() / 'train.jsonl'}.", file=sys.stderr)  # fmt: skip
        return 1
    print(f"Saved {folder}")
    return 0


def cmd_retrain(args: argparse.Namespace) -> int:
    """Train this month's model for every LightGBM book that lacks one (what the monthly job runs)."""
    setup_logging("retrain", to_stderr=False)
    results = training.retrain(sync=not args.no_sync)
    for r in results:
        print(r["message"])
    return 0 if all(r["status"] in ("trained", "up_to_date") for r in results) else 1


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _backup_path(user: Path) -> Path:
    """A backup name no earlier backup has: two installs in the same second get -2, -3, ..."""
    base = user.with_name(f"books.json.bak-{_utc_now():%Y%m%dT%H%M%SZ}")
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    return path


def install_books(use_default: bool, dry_run: bool, out=print) -> None:
    """Give the user an editable books.json: copy the repo default when there is none. An edited
    file is replaced only with ``--use-default-books``, and then only after a backup."""
    user, default = paths.user_books_path(), paths.DEFAULT_BOOKS
    if user.exists() and filecmp.cmp(user, default, shallow=False):
        return
    if user.exists() and not use_default:
        out(f"Kept your books settings at {user}. To switch to the four research books in {default}, run "
            "install again with --use-default-books (your file is backed up first).")  # fmt: skip
        return
    backup = _backup_path(user) if user.exists() else None
    if dry_run:
        first = f"back up {user} to {backup} and " if backup else ""
        out(f"Would {first}copy {default} to {user}")
        return
    user.parent.mkdir(parents=True, exist_ok=True)
    if backup:
        shutil.copy2(user, backup)
        out(f"Backed up your books settings to {backup}")
    shutil.copyfile(default, user)
    out(f"Copied the four research books to {user}; edit that file to change them.")


def cmd_install(args: argparse.Namespace) -> int:

    if not args.dry_run:
        setup_logging("install", to_stderr=False)  # launchctl's raw output goes here, not to the terminal
    install_books(args.use_default_books, args.dry_run)
    plan = macos.default_plan(stay_awake_on_ac=args.stay_awake_on_ac, dashboard=not args.no_dashboard)
    return macos.install(plan, dry_run=args.dry_run)


def cmd_uninstall(args: argparse.Namespace) -> int:
    if not args.dry_run:
        setup_logging("install", to_stderr=False)
    return macos.uninstall(macos.default_plan(), dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="signal-desk-live", description=f"Signal Desk Live paper runner. {PAPER}")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the paper-trading loop (what launchd keeps alive)")
    r.add_argument(
        "--no-console-log", action="store_true",
        help="log to the rotated runner.jsonl only, not also to stderr (launchd's stderr file never rotates)",
    )  # fmt: skip
    r.set_defaults(fn=cmd_run)
    t = sub.add_parser("tick", help="check every book once and print the result")
    t.add_argument("--no-notify", action="store_true", help="do not show macOS notifications")
    t.set_defaults(fn=cmd_tick)
    sub.add_parser("status", help="print each book's equity and position").set_defaults(fn=cmd_status)
    d = sub.add_parser("dashboard", help="serve the dashboard on 127.0.0.1")
    d.add_argument("--port", type=int, default=8765)
    d.set_defaults(fn=cmd_dashboard)
    tr = sub.add_parser("train", help="train a LightGBM model now (normally the monthly retrain does it)")
    tr.add_argument("model", choices=["lgbm"])
    tr.add_argument("--symbol", required=True, choices=list(training.SYMBOLS))
    tr.add_argument("--month", help="month the model trades, YYYY-MM (default: this month)")
    tr.add_argument("--config", default=training.DEFAULT_CONFIG, choices=sorted(training.art.WARMUP_BARS))
    tr.add_argument("--data-dir", help="folder with <SYMBOL>_5m.parquet (default: the app's market_data)")
    tr.add_argument("--no-sync", action="store_true", help="use the 5m history as it is; download nothing")
    tr.set_defaults(fn=cmd_train)
    rt = sub.add_parser("retrain", help="train this month's model for each LightGBM book that lacks one")
    rt.add_argument("--no-sync", action="store_true", help="use the 5m history as it is; download nothing")
    rt.set_defaults(fn=cmd_retrain)
    i = sub.add_parser("install", help="install and start the launchd agents")
    i.add_argument("--dry-run", action="store_true", help="print what would be written and run")
    i.add_argument("--stay-awake-on-ac", action="store_true", help="keep the Mac awake while on AC power")
    i.add_argument("--no-dashboard", action="store_true", help="install only the runner agent")
    i.add_argument("--use-default-books", action="store_true",
                   help="replace your books.json with the repo default (backed up first)")  # fmt: skip
    i.set_defaults(fn=cmd_install)
    u = sub.add_parser("uninstall", help="stop the launchd agents and remove their files")
    u.add_argument("--dry-run", action="store_true", help="print what would be removed")
    u.set_defaults(fn=cmd_uninstall)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except StoreError as exc:  # already written for the user: no traceback
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
