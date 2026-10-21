# Aggressive strategies with stop-loss (paper only)

Real Yahoo daily OHLC for `TQQQ` / `SOXL` / `QQQ` / `SPY`. Entries at next open. **Hard + trailing stops** use the daily Low (fill at stop). Costs: **10 bp** round-trip.

## Reality check (read this)

- **Significant profit every day or most weeks is not what the data shows** — even the best leveraged books here are green only ~**37–45%** of weeks and ~**30%** of days.
- Average weekly return can look strong (~**0.5–1%**) because a few huge up weeks dominate; **median week is often ~0%**.
- Drawdowns of **−40% to −80%** are normal for these books. Stops cut some tails; they do **not** make leveraged ETFs “safe.”

## Bakeoff (2015 → 2026-10, top names)

| Strategy | CAGR | Avg week | Green weeks | MaxDD | Stops |
| --- | ---: | ---: | ---: | ---: | ---: |
| SOXL mom10 hard10/trail15 | ~31% | ~0.84% | ~42% | ~−67% | many |
| **TQQQ mom10 hard10/trail15 (promoted)** | **~22%** | **~0.54%** | **~45%** | **~−50%** | moderate |
| TQQQ sma50 hard12/trail20 | ~23% | ~0.54% | ~43% | ~−55% | few |

## Sealed OOS 2022+

| Strategy | CAGR vs SPY | Avg week | MaxDD |
| --- | --- | ---: | ---: |
| SOXL mom10 | ~31% (beats SPY) | ~1.0% | ~−55% |
| **TQQQ mom10** | **~30% (beats SPY)** | **~0.72%** | **~−44%** |
| TQQQ sma50 hard12 | ~27% | ~0.66% | ~−40% |

OOS median CAGR across top names ~**26%**; ~**88%** of those OOS rows beat SPY — still not “green every week.”

## Promoted paper book

**`tqqq_mom10_hard10_trail15`**

1. Universe: `TQQQ` only (3× Nasdaq-100).
2. Signal (at close): long if 10-day return > 0, else flat.
3. Fill: next session open.
4. Stops (intrabar via Low): **10% hard** from entry and **15% trailing** from peak since entry.
5. Paper command: `profit-engine paper --aggressive`

Chosen over SOXL for similar OOS CAGR with **less catastrophic** full-sample drawdowns. Use SOXL only if you explicitly want max aggression.

## vs prior GEM monthly book

| Book | Horizon | Intent |
| --- | --- | --- |
| `gem_qqq_veu` | monthly | beat inflation + SPY with milder DD |
| `tqqq_mom10` + stops | daily bars / weekly-ish turnover | higher CAGR, much higher risk |

Run both as separate paper `run_id`s; do not mix capital in one ledger without a written allocation rule.
