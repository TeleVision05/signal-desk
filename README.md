# Signal Desk

Paper-only multi-book ML desk (fake money only — no live broker/exchange orders):

1. **TQQQ paper** — US equity RTH, Yahoo daily bars, walk-forward HGB  
2. **BTC paper** — 24/7, Binance public daily klines, same HGB pattern, local ledger fills  

One deployable host:

1. **Backend** (Render) — syncs models, executes **paper** trades, emails on fills  
2. **Frontend** — same host, live dashboard with book switcher (polls `/api/state`)  
3. **Notifications** — backend emails `TRADE_ALERT_EMAIL` on paper BUY / SELL / stop  

Live: https://signal-desk-upb4.onrender.com

## Architecture

```text
Yahoo (TQQQ)  ─┐
               ├→ walk-forward HGB → p(up) vs 55% → paper ledger (Postgres)
Binance (BTC) ─┘                                      ↓
                                         /api/state  +  email on paper fill
                                                      ↓
                                              Signal Desk UI
```

Money, fills, equity marks, and decision history live in **Supabase Postgres**
(`DATABASE_URL`). Without that env var, the backend falls back to local SQLite.

Schedules:  
- **TQQQ paper** — weekdays ET: pre-open ~9:35, hourly RTH, after close  
- **BTC paper** — every 4 hours UTC (including weekends)  

GitHub Action `render-wake-sync.yml` wakes the free Render dyno and hits `/api/sync?book=all`.

## Local run

```bash
pip install -r requirements.txt && pip install -e .
signal-desk   # http://localhost:8080
```

## Env (Render)

| Key | Purpose |
|-----|---------|
| `DATABASE_URL` | Supabase Postgres URI (session pooler). Required on Render for durable paper money |
| `PAPER_CASH` | Starting paper cash per book (default `1000`) |
| `CRON_SECRET` | Optional sync auth |
| `TRADE_ALERT_EMAIL` | Inbox for paper trade emails |
| `RESEND_API_KEY` | **Preferred** free email (Resend) |
| `SMTP_HOST` / `SMTP_USER` / `SMTP_PASS` | Or Gmail SMTP + app password |

## Research CLI (optional)

```bash
profit-engine ml --sweep
profit-engine paper --ml --cash 1000
```

See `FINDINGS_ML.md` for backtest notes.
