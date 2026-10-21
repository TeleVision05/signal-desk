# Walk-forward ML findings (real Yahoo data, paper only)

Trained **HistGradientBoosting / LogisticRegression** on price+volume+cross-asset features with **purged walk-forward** (train ~2y, test ~1m, embargo = label horizon). Long/flat on `TQQQ`/`QQQ` with hard+trailing stops. Costs 10 bp RT.

## Sweep honesty

- Grid size **N=12** pre-registered cells (not infinite tune-until-green).
- **Median** last-3y CAGR across grid ≈ **13.9%** (many cells lose to SPY).
- Top cells are real walk-forward OOS predictions, but picking the best cell still has **selection bias** — treat paper as the next true OOS.

## Promoted paper book: `ml_TQQQ_hgb_h5_t0.55`

Rules (frozen):

1. Features at close: returns, vol, SMA distances, RSI, ATR%, volume z, Donchian, SPY/QQQ/VIX extras.
2. Label: 5-day forward return > 0.35 × 20d vol × √5.
3. Walk-forward HGB; go long when P(up) ≥ **0.55**.
4. Stops: **8% hard**, **12% trailing**.
5. Fill next open; 10 bp RT costs.

| Window | Result on **$1,000** | Notes |
| --- | --- | --- |
| **Last 3 months** (2026-07-20→10-20) | **$1,160.83 (+$160.83 / +16.1%)** | MaxDD ~−2.8%, 12 trades |
| Last 3 years | ~**+30.9% CAGR** vs SPY ~23.7% | MaxDD ~−30%, beats SPY |
| Full WF sample | ~11% CAGR | Does **not** beat buy-hold SPY over entire WF life |

## Max-aggression alt: `ml_TQQQ_logreg_h3_t0.55`

| Window | **$1,000** |
| --- | --- |
| Last 3 months | **$1,220.08 (+$220 / +22.0%)** |
| Last 3 years | ~**35.0% CAGR** vs SPY ~23.7% |
| Full sample | ~16.6% CAGR, MaxDD **~−70%** |

## Still not “green every week”

Even the winners: ~**36–41%** positive weeks, median week ~**0%**. Profit is lumpy. Stops help; they do not create a salary-like equity curve.

## Commands

```bash
profit-engine ml --sweep
profit-engine paper --ml --cash 1000
```

Compared to prior mom10 book (~+$10 on $1k last 3m), ML is the clear upgrade **for this recent window**. Keep running paper and do not retune on paper P&L.
