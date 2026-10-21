"""Trade email alerts from the backend (paper fills only)."""

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


def email_configured() -> bool:
    if os.environ.get("RESEND_API_KEY", "").strip():
        return True
    return bool(
        os.environ.get("SMTP_HOST", "").strip()
        and os.environ.get("SMTP_USER", "").strip()
        and os.environ.get("SMTP_PASS", "").strip()
    )


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
        f"Action:      {info.get('action')}",
        f"Signal:      {info.get('signal')}",
        f"p(up):       {info.get('proba')}",
        f"Threshold:   {info.get('threshold')}",
        f"Equity:      {info.get('equity')}",
        f"Cash:        {info.get('cash')}",
        f"Shares:      {info.get('shares')}",
        f"Mark:        {info.get('mark')}",
        f"Bar:         {info.get('bar_date')}",
        f"Synced:      {info.get('synced_at')}",
        f"Explanation: {info.get('explanation')}",
        "",
        "Desk: https://signal-desk-upb4.onrender.com",
    ]
    return "\n".join(str(x) for x in lines)


def _save_status(payload: dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload["at"] = datetime.now(timezone.utc).isoformat()
    STATUS_PATH.write_text(json.dumps(payload, indent=2, default=str))


def load_notify_status() -> dict[str, Any]:
    if not STATUS_PATH.exists():
        return {"ok": False, "configured": email_configured()}
    try:
        data = json.loads(STATUS_PATH.read_text())
        data["configured"] = email_configured()
        return data
    except Exception:  # noqa: BLE001
        return {"ok": False, "configured": email_configured()}


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


def send_trade_alert(info: dict[str, Any]) -> dict[str, Any]:
    """Email on real paper fills. Never raises into the trading loop."""
    result: dict[str, Any] = {
        "ok": False,
        "via": None,
        "to": _to_addr(),
        "action": info.get("action"),
        "configured": email_configured(),
    }
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
    if not email_configured():
        result["error"] = (
            "Email not configured. Set RESEND_API_KEY or SMTP_PASS (Gmail app password) on Render."
        )
        print(f"[notify] FAILED: {result['error']}", flush=True)
        _save_status(result)
        return result

    subject = _subject(info)
    body = _body(info)
    errors: list[str] = []
    try:
        for sender in (_send_resend, _send_smtp):
            try:
                via = sender(to, subject, body)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{sender.__name__}: {exc}")
                continue
            if via:
                result.update({"ok": True, "via": via, "error": None})
                print(f"[notify] emailed via {via} → {to} ({action})", flush=True)
                _save_status(result)
                return result
        result["error"] = "; ".join(errors) if errors else "no transport succeeded"
        print(f"[notify] FAILED: {result['error']}", flush=True)
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        print(f"[notify] email failed: {exc}", flush=True)
    _save_status(result)
    return result
