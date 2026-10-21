# profit-engine / Signal Desk

Paper-only trading lab + **Signal Desk** web dashboard for the promoted walk-forward **TQQQ ML** book. Uses **real Yahoo Finance / FRED data** and a **fake-money SQLite wallet**. No live orders.

## Signal Desk (deployable dashboard)

```bash
pip install -e .
pip install -r requirements.txt
# optional: build initial JSON state
python -c "from profit_engine.dashboard_state import build_state; build_state()"
signal-desk   # http://localhost:8080
```

Env vars: `PORT`, `PAPER_CASH` (default 1000), `CRON_SECRET` (required for `/api/sync`).

Docker: `docker build -t signal-desk . && docker run -p 8080:8080 -e CRON_SECRET=... signal-desk`

Deploy target: GitHub repo under **TeleVision05** + Render (`render.yaml`) or any Docker host. Weekday scheduler resyncs paper at 21:15 UTC; GitHub Actions is a backup cron.

This replaces prior local experiments (ORB day-trading, crypto arb, sports arb) with a low-turnover monthly system.

- Research: `~/reports/Consistent automated trading edge.md`
- Measured results: [`FINDINGS.md`](FINDINGS.md)

## Reality check

- Beating the S&P **every day or week** is not a realistic design goal (S&P itself is green only ~55–58% of weeks).
- In locked, costed tests, **gem_qqq_veu** beat SPY CAGR and CPI with milder drawdowns than buy-and-hold QQQ. Classic SPY/VEU GEM beat CPI but lagged SPY CAGR in 2008–2026.

## Setup

```bash
cd ~/Documents/Git/profit-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

## Commands

```bash
profit-engine fetch
profit-engine backtest
profit-engine backtest --oos-start 2019-01-01
profit-engine walkforward
profit-engine iterate --oos-start 2019-01-01
profit-engine paper
profit-engine aggressive              # riskier daily/weekly bakeoff with stops
profit-engine paper --aggressive      # TQQQ mom10 + hard/trailing stops paper book
profit-engine ml --sweep              # walk-forward ML grid (train on real features)
profit-engine paper --ml --cash 1000  # promoted ML TQQQ paper book
```

Outputs land in `output/`. Paper ledger: `data/paper_ledger.sqlite`.
Aggressive results: [`FINDINGS_AGGRESSIVE.md`](FINDINGS_AGGRESSIVE.md).  
ML results: [`FINDINGS_ML.md`](FINDINGS_ML.md).

## Frozen gem_qqq_veu rules (paper default)

1. At each **completed** month-end, compute 12-month total returns (adjusted closes).
2. Relative: hold the stronger of `QQQ` vs `VEU`.
3. Absolute: if that winner’s 12-month return ≤ 0%, hold `AGG` instead.
4. Fill at the **next session open**; charge **7.5 bp** round-trip on turnover.
