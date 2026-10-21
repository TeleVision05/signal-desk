"""Trade alert emails (paper fills)."""

from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import requests

DEFAULT_TO = "tarun.vadapalli.1@gmail.com"
TRADE_ACTIONS = frozenset({"signal_entry", "signal_exit", "stop_exit"})
ROOT = Path(__file__).resolve().parents[2]
STATUS_PATH = ROOT / "data" / "notify_last.json"


def _enabled() -> bool:
    return os.environ.get("TRADE_ALERTS", "1") == "1"


def _to_addr() -> str:
    return (os.environ.get("TRADE_ALERT_EMAIL") or DEFAULT_TO).strip()


def _subject(info: dict[str, Any]) -> str:
    action = str(info.get("action") or "trade")
    side = {
        "signal_entry": "BUY",
        "signal_exit": "SELL",
        "stop_exit": "STOP SELL",
    }.get(action, action.upper())
    return f"[Signal Desk] {side} TQQQ paper trade"


def _body(info: dict[str, Any]) -> str:
    lines = [
        "Paper trade executed on Signal Desk.",
        "",
        f"Action:  {info.get('action')}",
        f"Signal:  {info.get('signal')}",
        f"Proba:   {info.get('proba')}",
        f"Equity:  {info.get('equity')}",
        f"Cash:    {info.get('cash')}",
        f"Shares:  {info.get('shares')}",
        f"Mark:    {info.get('mark')}",
        f"Synced:  {info.get('synced_at')}",
        f"Run:     {info.get('run_id')}",
        "",
        "Live desk: https://signal-desk-upb4.onrender.com",
    ]
    return "\n".join(lines)


def _save_status(payload: dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload["at"] = datetime.now(timezone.utc).isoformat()
    STATUS_PATH.write_text(json.dumps(payload, indent=2, default=str))


def load_notify_status() -> dict[str, Any]:
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _send_resend(to: str, subject: str, body: str) -> str | None:
    key = os.environ.get("RESEND_API_KEY", "").strip()
    if not key:
        return None
    from_addr = os.environ.get("TRADE_ALERT_FROM", "Signal Desk <onboarding@resend.dev>")
    r = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"from": from_addr, "to": [to], "subject": subject, "text": body},
        timeout=20,
    )
    if not r.ok:
        raise RuntimeError(f"Resend HTTP {r.status_code}: {r.text[:240]}")
    return "resend"


def _send_smtp(to: str, subject: str, body: str) -> str | None:
    host = os.environ.get("SMTP_HOST", "").strip()
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASS", "").strip()
    if not (host and user and password):
        return None
    port = int(os.environ.get("SMTP_PORT", "587"))
    from_addr = os.environ.get("TRADE_ALERT_FROM", user)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to
    msg.set_content(body)
    with smtplib.SMTP(host, port, timeout=20) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
    return "smtp"


def _send_ntfy(to: str, subject: str, body: str) -> str | None:
    """Push + optional email via ntfy (email needs NTFY_EMAIL=1 on ntfy.sh paid, push is free)."""
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return None
    headers = {
        "Title": subject[:120],
        "Priority": "high",
        "Tags": "chart_with_upwards_trend,moneybag",
    }
    # ntfy.sh email delivery when enabled
    if os.environ.get("NTFY_EMAIL", "0") == "1":
        headers["Email"] = to
    r = requests.post(f"https://ntfy.sh/{topic}", headers=headers, data=body.encode("utf-8"), timeout=15)
    if not r.ok:
        raise RuntimeError(f"ntfy HTTP {r.status_code}: {r.text[:200]}")
    return "ntfy"


def send_trade_alert(info: dict[str, Any]) -> dict[str, Any]:
    """Email/push on real paper fills. Never raises into the trading loop."""
    result: dict[str, Any] = {"ok": False, "via": None, "to": _to_addr(), "action": info.get("action")}
    if not _enabled():
        result["error"] = "TRADE_ALERTS disabled"
        _save_status(result)
        return result
    action = str(info.get("action") or "")
    if action not in TRADE_ACTIONS:
        result["error"] = f"skipped non-trade action={action}"
        _save_status(result)
        return result
    to = _to_addr()
    if not to:
        result["error"] = "missing TRADE_ALERT_EMAIL"
        _save_status(result)
        return result
    subject = _subject(info)
    body = _body(info)
    errors: list[str] = []
    try:
        for sender in (_send_resend, _send_smtp, _send_ntfy):
            try:
                via = sender(to, subject, body)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{sender.__name__}: {exc}")
                continue
            if via:
                result.update({"ok": True, "via": via, "error": None})
                print(f"[notify] sent via {via} → {to} ({action})", flush=True)
                _save_status(result)
                return result
        result["error"] = (
            "No working email transport. Set RESEND_API_KEY or SMTP_HOST/SMTP_USER/SMTP_PASS "
            "(Gmail app password) on Render. "
            + ("; ".join(errors) if errors else "FormSubmit removed — it was timing out.")
        )
        print(f"[notify] FAILED: {result['error']}", flush=True)
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        print(f"[notify] email failed: {exc}", flush=True)
    _save_status(result)
    return result
