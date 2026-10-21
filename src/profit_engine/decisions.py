"""Persist each paper sync decision so the desk can show p vs threshold history."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "data" / "decision_log.json"
MAX_ROWS = 120


def _load() -> list[dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    try:
        data = json.loads(LOG_PATH.read_text())
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []


def append_decision(row: dict[str, Any]) -> list[dict[str, Any]]:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = _load()
    # Dedupe same bar+sync minute to avoid spam on rapid retries
    key = (row.get("bar_date"), row.get("synced_at", "")[:16], row.get("action"))
    rows = [
        r
        for r in rows
        if (r.get("bar_date"), str(r.get("synced_at") or "")[:16], r.get("action")) != key
    ]
    rows.append(row)
    rows = rows[-MAX_ROWS:]
    LOG_PATH.write_text(json.dumps(rows, indent=2, default=str))
    return rows


def load_decisions(limit: int = 60) -> list[dict[str, Any]]:
    rows = _load()
    return list(reversed(rows[-limit:]))


def explain_decision(row: dict[str, Any]) -> str:
    """Plain-English summary of one comparison."""
    p = row.get("proba")
    thr = float(row.get("threshold") or 0.55)
    signal = row.get("signal")
    action = row.get("action") or "hold"
    if p is None:
        base = "Model probability unavailable for this bar."
    else:
        p = float(p)
        pct = p * 100
        thr_pct = thr * 100
        gap = (p - thr) * 100
        if signal == 1:
            base = (
                f"p(up)={pct:.1f}% ≥ threshold {thr_pct:.0f}% "
                f"(+{gap:.1f} pts) → LONG signal."
            )
        else:
            base = (
                f"p(up)={pct:.1f}% < threshold {thr_pct:.0f}% "
                f"({gap:.1f} pts) → FLAT signal."
            )
    outcomes = {
        "hold": "No trade — kept prior position.",
        "signal_entry": "Bought TQQQ (entered long).",
        "signal_exit": "Sold TQQQ (signal flipped flat).",
        "stop_exit": "Sold TQQQ (stop loss hit).",
    }
    return f"{base} {outcomes.get(action, action)}"
