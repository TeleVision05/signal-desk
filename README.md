# Signal Desk

Paper-only **TQQQ walk-forward ML** system with one deployable host:

1. **Backend** (Render) — syncs the model, executes paper trades, emails on fills  
2. **Frontend** — same host, live dashboard (polls `/api/state` every 30s)  
3. **Notifications** — backend emails `TRADE_ALERT_EMAIL` on BUY / SELL / stop  

Live: https://signal-desk-upb4.onrender.com

## Architecture

```text
Yahoo data → walk-forward HGB → p(up) vs 55% → paper ledger (Postgres)
                                      ↓
                         /api/state  +  email on fill
                                      ↓
                              Signal Desk UI
```

Money, fills, equity marks, and decision history live in **Supabase Postgres**
(`DATABASE_URL`). Without that env var, the backend falls back to local SQLite.

Weekday schedule (ET): pre-open ~9:35, hourly during RTH, after close.  
GitHub Action `render-wake-sync.yml` wakes the free Render dyno and hits `/api/sync`.

## Local run

```bash
pip install -r requirements.txt && pip install -e .
signal-desk   # http://localhost:8080
```

## Env (Render)

| Key | Purpose |
|-----|---------|
| `DATABASE_URL` | Supabase Postgres URI (session pooler). Required on Render for durable paper money |
| `PAPER_CASH` | Starting paper cash (default `1000`) |
| `CRON_SECRET` | Optional sync auth |
| `TRADE_ALERT_EMAIL` | Inbox for trade emails |
| `RESEND_API_KEY` | **Preferred** free email (Resend) |
| `SMTP_HOST` / `SMTP_USER` / `SMTP_PASS` | Or Gmail SMTP + app password |

## Research CLI (optional)

```bash
profit-engine ml --sweep
profit-engine paper --ml --cash 1000
```

See `FINDINGS_ML.md` for backtest notes.
