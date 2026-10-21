"""Persist each paper sync decision (Postgres via DATABASE_URL, else JSON file)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .db import database_url

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "data" / "decision_log.json"
MAX_ROWS = 120
RUN_ID = "ml-tqqq-hgb-paper-v1"


def _dedupe_key(row: dict[str, Any]) -> str:
    return f"{row.get('bar_date')}|{str(row.get('synced_at') or '')[:16]}|{row.get('action')}"


def _load_file() -> list[dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    try:
        data = json.loads(LOG_PATH.read_text())
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []


def _pg_conn():
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(database_url(), row_factory=dict_row)


def append_decision(row: dict[str, Any]) -> list[dict[str, Any]]:
    key = _dedupe_key(row)
    url = database_url()
    if url:
        payload = json.dumps(row, default=str)
        with _pg_conn() as conn:
            with conn.transaction():
                conn.execute("DELETE FROM paper_decisions WHERE dedupe_key = %s", (key,))
                conn.execute(
                    """
                    INSERT INTO paper_decisions(run_id, bar_date, synced_at, action, dedupe_key, payload)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        row.get("run_id") or RUN_ID,
                        row.get("bar_date"),
                        row.get("synced_at"),
                        row.get("action"),
                        key,
                        payload,
                    ),
                )
                # Cap history
                conn.execute(
                    """
                    DELETE FROM paper_decisions
                    WHERE id NOT IN (
                      SELECT id FROM paper_decisions ORDER BY id DESC LIMIT %s
                    )
                    """,
                    (MAX_ROWS,),
                )
        return load_decisions(limit=MAX_ROWS)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_file()
    rows = [r for r in rows if _dedupe_key(r) != key]
    rows.append(row)
    rows = rows[-MAX_ROWS:]
    LOG_PATH.write_text(json.dumps(rows, indent=2, default=str))
    return rows


def load_decisions(limit: int = 60, *, run_id: str | None = None) -> list[dict[str, Any]]:
    url = database_url()
    if url:
        with _pg_conn() as conn:
            if run_id:
                rows = conn.execute(
                    """
                    SELECT payload FROM paper_decisions
                    WHERE run_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (run_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT payload FROM paper_decisions
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (limit,),
                ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            payload = r["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            out.append(dict(payload))
        return out
    rows = _load_file()
    if run_id:
        rows = [r for r in rows if (r.get("run_id") or RUN_ID) == run_id]
    return list(reversed(rows[-limit:]))


def explain_decision(row: dict[str, Any]) -> str:
    """Plain-English summary of one comparison."""
    p = row.get("proba")
    thr = float(row.get("threshold") or 0.55)
    signal = row.get("signal")
    action = row.get("action") or "hold"
    asset = str(row.get("asset") or "TQQQ")
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
        "signal_entry": f"Bought {asset} (entered long).",
        "signal_exit": f"Sold {asset} (signal flipped flat).",
        "stop_exit": f"Sold {asset} (stop loss hit).",
    }
    return f"{base} {outcomes.get(action, action)}"
