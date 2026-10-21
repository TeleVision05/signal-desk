"""Trade alert emails (paper fills)."""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from typing import Any

import requests

DEFAULT_TO = "tarun.vadapalli.1@gmail.com"
TRADE_ACTIONS = frozenset({"signal_entry", "signal_exit", "stop_exit"})


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


def _send_resend(to: str, subject: str, body: str) -> bool:
    key = os.environ.get("RESEND_API_KEY", "").strip()
    if not key:
        return False
    from_addr = os.environ.get("TRADE_ALERT_FROM", "Signal Desk <onboarding@resend.dev>")
    r = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"from": from_addr, "to": [to], "subject": subject, "text": body},
        timeout=30,
    )
    r.raise_for_status()
    return True


def _send_smtp(to: str, subject: str, body: str) -> bool:
    host = os.environ.get("SMTP_HOST", "").strip()
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASS", "").strip()
    if not (host and user and password):
        return False
    port = int(os.environ.get("SMTP_PORT", "587"))
    from_addr = os.environ.get("TRADE_ALERT_FROM", user)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to
    msg.set_content(body)
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
    return True


def _send_formsubmit(to: str, subject: str, body: str) -> bool:
    """Zero-config path: FormSubmit (confirm the address once via their email)."""
    if os.environ.get("TRADE_ALERT_FORMSUBMIT", "1") != "1":
        return False
    r = requests.post(
        f"https://formsubmit.co/ajax/{to}",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        json={
            "_subject": subject,
            "_template": "box",
            "_captcha": "false",
            "message": body,
            "desk": "signal-desk",
        },
        timeout=30,
    )
    r.raise_for_status()
    return True


def send_trade_alert(info: dict[str, Any]) -> None:
    """Email on real paper fills. Never raises into the trading loop."""
    if not _enabled():
        return
    action = str(info.get("action") or "")
    if action not in TRADE_ACTIONS:
        return
    to = _to_addr()
    if not to:
        return
    subject = _subject(info)
    body = _body(info)
    try:
        if _send_resend(to, subject, body):
            print(f"[notify] emailed via Resend → {to} ({action})", flush=True)
            return
        if _send_smtp(to, subject, body):
            print(f"[notify] emailed via SMTP → {to} ({action})", flush=True)
            return
        if _send_formsubmit(to, subject, body):
            print(f"[notify] emailed via FormSubmit → {to} ({action})", flush=True)
            return
        print(f"[notify] skipped — no email transport configured for {to}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[notify] email failed: {exc}", flush=True)
