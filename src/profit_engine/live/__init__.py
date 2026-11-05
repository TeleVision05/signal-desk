"""Signal Desk Live: local paper trading of intraday crypto books on closed Binance bars.

PAPER MONEY ONLY. Usage guide: ``docs/live_runner.md``. Modules:

* ``strategy`` / ``models`` / ``registry`` - strategy contract, model artifacts, books.json
* ``market`` - Binance 5m bars and exchange clock (reuses ``intraday.binance`` / ``intraday.data``)
* ``store`` / ``accounting`` / ``queries`` - local SQLite, exactly-once bar commits, read views
* ``runner`` / ``service`` - one tick; the long-running loop (lock, signals, backoff, JSON logs)
* ``notify`` / ``app`` / ``macos`` / ``cli`` - notifications, dashboard, launchd, command line
"""
