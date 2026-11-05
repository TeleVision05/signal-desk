#!/bin/sh
# Install (or reinstall) the Signal Desk Live launchd agents for THIS checkout.
#   scripts/macos/install.sh --dry-run            # print the plists and launchctl commands only
#   scripts/macos/install.sh --stay-awake-on-ac   # also keep the Mac awake while on AC power
# Uses the repo's .venv python unless PYTHON is set. No sudo; per-user agents only.
set -eu
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"
PYTHONPATH="$REPO/src" exec "$PY" -m profit_engine.live.cli install "$@"
