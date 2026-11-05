"""``signal-desk-live run|tick|status|dashboard|install|uninstall``."""

from __future__ import annotations

import argparse
import json
import os
import sys

from profit_engine.live import macos, paths, queries
from profit_engine.live.app import serve
from profit_engine.live.market import BinanceBarSource
from profit_engine.live.notify import Notifier
from profit_engine.live.registry import ConfigError, load_config
from profit_engine.live.runner import tick
from profit_engine.live.service import run_forever, setup_logging
from profit_engine.live.store import Store

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


def cmd_install(args: argparse.Namespace) -> int:

    plan = macos.default_plan(stay_awake_on_ac=args.stay_awake_on_ac, dashboard=not args.no_dashboard)
    return macos.install(plan, dry_run=args.dry_run)


def cmd_uninstall(args: argparse.Namespace) -> int:

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
    i = sub.add_parser("install", help="install and start the launchd agents")
    i.add_argument("--dry-run", action="store_true", help="print what would be written and run")
    i.add_argument("--stay-awake-on-ac", action="store_true", help="keep the Mac awake while on AC power")
    i.add_argument("--no-dashboard", action="store_true", help="install only the runner agent")
    i.set_defaults(fn=cmd_install)
    u = sub.add_parser("uninstall", help="stop the launchd agents and remove their files")
    u.add_argument("--dry-run", action="store_true", help="print what would be removed")
    u.set_defaults(fn=cmd_uninstall)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
