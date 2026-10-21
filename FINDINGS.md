# Findings (paper only, real Yahoo + FRED data)

Numbers are net of a **7.5 bp** round-trip cost on turnover. Fake capital: **$100,000**.

Full research: `/Users/tarunvadapalli/reports/Consistent automated trading edge.md`.

## Local bots on disk (do not extend as primary)

| Path | What it was |
| --- | --- |
| `~/Documents/Git/CryptoTrade` | Funding / lending / sports / DEX paper engine |
| `~/Documents/Python/TradingStrategy` | ORB equity Flask backtester |
| `~/Documents/Python/Trader` | Intraday SPY |
| `~/Documents/Python/DayTrader` | Polygon day-trade replay |
| `~/Documents/PaperTrade` | Binance ML / brackets |
| `~/Documents/Python/crypto-arbitrage-lab` | CEX arb |
| `~/Documents/Python/Arbitrage`, `~/sports-arbitrage-system` | Sports arb |
| `~/Documents/Bot` | Discord fills bot |

## Promoted strategy: GEM on QQQ / VEU / AGG

Classic dual momentum, but risk-on pair is **QQQ vs VEU** (not SPY vs VEU). Absolute filter: if winner’s 12-month return ≤ 0% → hold **AGG**. Signal at completed month-end; fill next open.

### Full sample ~2008-04 → 2026-10-16

| Strategy | CAGR | Excess vs SPY | MaxDD | Beats CPI |
| --- | ---: | ---: | ---: | --- |
| **gem_qqq_veu** | **13.51%** | **+1.77%** | **−26.9%** | **yes (~11% real)** |
| gem_spy_veu (classic) | 8.55% | −3.19% | −23.7% | yes |
| Faber SPY SMA | 8.07% | −3.78% | −23.2% | yes |
| Buy & hold SPY | 10.62% | — | −52.9% | yes |
| Buy & hold QQQ | 15.91% | +5.29% | −50.4% | yes |

### Sealed OOS 2019-02 → 2026-10-16

| Strategy | CAGR | Excess vs SPY | MaxDD | Beats CPI |
| --- | ---: | ---: | ---: | --- |
| **gem_qqq_veu** | **17.84%** | **+1.70%** | **−26.9%** | **yes** |
| gem_spy_veu | 11.63% | −4.51% | −23.7% | yes |
| Buy & hold SPY | 16.14% | — | −23.8%* | yes |
| Buy & hold QQQ | 21.36% | +5.22% | −33.7% | yes |

\*SPY BH MaxDD on this short OOS window is milder than the GFC-era full sample.

## Walk-forward (2y OOS slices)

| Variant | Mean excess vs SPY | Fractions beat SPY | Beat CPI |
| --- | ---: | ---: | ---: |
| gem_spy_veu | −3.86% | 0/5 | 4/5 |
| **gem_qqq_veu** | **−0.37%** | **3/5** | **4/5** |

gem_qqq_veu still loses some slices (notably 2021–2023 rate-shock / tech drawdown). It is not magic every period — but it is the only automated book we tested that clears inflation **and** beats SPY CAGR on the long sample and sealed 2019+ window.

## What “consistent” does and does not mean

- **Daily profit every day**: unrealistic. Weekly hit rates here are ~**58–67%**, including buy-and-hold.
- **Beat inflation**: yes for gem_qqq_veu (and for SPY/QQQ BH).
- **Beat S&P CAGR**: yes for **gem_qqq_veu** in both the full sample and the sealed 2019+ window, with **smaller drawdowns than QQQ buy-and-hold**.
- Classic SPY/VEU GEM **does not** beat SPY CAGR in this bullish sample; its edge is drawdown control, not raw return.

## Paper wallet

```bash
profit-engine paper   # books gem_qqq_veu into data/paper_ledger.sqlite
```

Current signal (completed Sep 2026 month-end): see CLI output. Re-run after each month-end close.

## Commands

```bash
cd ~/Documents/Git/profit-engine
source .venv/bin/activate
profit-engine fetch
profit-engine backtest
profit-engine backtest --oos-start 2019-01-01
profit-engine walkforward
profit-engine paper
```
