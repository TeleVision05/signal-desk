#!/bin/sh
# Stop the Signal Desk Live launchd agents and remove their plists. Data and logs are kept.
#   scripts/macos/uninstall.sh --dry-run   # print what would be removed
set -eu
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"
PYTHONPATH="$REPO/src" exec "$PY" -m profit_engine.live.cli uninstall "$@"
