"""Holdout guard.

These entry points call :func:`assert_pre_holdout` on the timestamps they are about to
touch: ``data.load_klines``; ``labels.forward_return`` / ``triple_barrier``; ``splits``
``walk_forward_splits`` / ``purged_kfold`` / ``purged_tail_split``; ``backtest.run_backtest``;
every public ``metrics`` helper and ``summarize``; ``overfitting.dsr_from_daily`` /
``pbo_cscv`` (DataFrame input) / ``permutation_test``; ``trials.log_trial``; ``promotion``.
They raise :class:`HoldoutViolation` for anything at or after ``HOLDOUT_START`` unless the
caller is inside ``with open_holdout(reason=...)``, which first appends an audit record
(UTC timestamp, reason, caller file:line:function, git sha) to
``config.HOLDOUT_AUDIT_LOG`` (``research/intraday/holdout_access.jsonl``). The path is not
a parameter: tests isolate it by monkeypatching ``config.HOLDOUT_AUDIT_LOG``.

Not guarded (raw plumbing, not scoring): ``binance.*`` downloaders, ``data.sync_cache``,
``resample`` and ``features``. Their output only reaches a score through a guarded call.

The guard is honour-system code, not a security boundary: its job is to make an
accidental peek impossible and a deliberate one visible in the audit log.
"""

from __future__ import annotations

import contextvars
import getpass
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from profit_engine.intraday import config
from profit_engine.intraday.provenance import git_sha

_OPEN: contextvars.ContextVar[str | None] = contextvars.ContextVar("holdout_open", default=None)
MIN_REASON_CHARS = 10


class HoldoutViolation(RuntimeError):
    """Raised when code touches data at or after HOLDOUT_START outside open_holdout()."""


def holdout_is_open() -> bool:
    return _OPEN.get() is not None


def _caller(depth: int) -> str:
    frame = sys._getframe(depth)
    return f"{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}"


class open_holdout:  # noqa: N801 - used as a context manager, reads like a function
    """Context manager that unseals the holdout for the enclosed block and audits it.

    >>> with open_holdout(reason="final check of lgbm_v3 after promotion PASS"):
    ...     ...
    """

    def __init__(self, reason: str) -> None:
        if not isinstance(reason, str) or len(reason.strip()) < MIN_REASON_CHARS:
            raise ValueError(f"open_holdout needs a reason of at least {MIN_REASON_CHARS} characters")
        self.reason = reason.strip()
        self.caller = _caller(2)
        self._token: contextvars.Token | None = None

    def __enter__(self) -> open_holdout:
        path = Path(config.HOLDOUT_AUDIT_LOG)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "reason": self.reason,
            "caller": self.caller,
            "user": getpass.getuser(),
            "git_sha": git_sha(),
            "holdout_start": config.HOLDOUT_START.isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._token = _OPEN.set(self.reason)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _OPEN.reset(self._token)
            self._token = None


def _latest(ts: object) -> pd.Timestamp | None:
    """Largest timestamp in a scalar / index / Series / DataFrame index, or None if empty."""
    if isinstance(ts, (pd.Series, pd.DataFrame)):
        ts = ts.index
    if isinstance(ts, pd.Index):
        if len(ts) == 0:
            return None
        ts = ts.max()
    if ts is None:
        return None
    out = pd.Timestamp(ts)
    return out.tz_localize("UTC") if out.tzinfo is None else out.tz_convert("UTC")


def assert_pre_holdout(ts: object, what: str) -> None:
    """Raise HoldoutViolation if ``ts`` (or its latest element) is >= HOLDOUT_START.

    ``ts`` may be a timestamp, a DatetimeIndex, or a Series/DataFrame (its index is
    checked). For a half-open range end, pass ``end - 1ns``.
    """
    latest = _latest(ts)
    if latest is None or latest < config.HOLDOUT_START or holdout_is_open():
        return
    raise HoldoutViolation(
        f"{what} reaches {latest.isoformat()}, at or after HOLDOUT_START "
        f"{config.HOLDOUT_START.isoformat()}. Use `with open_holdout(reason=...)` "
        "only for the final, logged holdout check."
    )
