# Regime-gated exposure (family `regime_gated`, kind `model`)

Script: `experiments/intraday/regime_gated.py`. Long/flat spot, BTCUSDT and ETHUSDT, walk-forward
OOS window [2020-01-01, 2026-07-01), monthly refit, every config logged at 5 and 10 bps/side,
seed 7. Bars: 5m cache resampled with `incomplete="drop"`.

The idea: do not predict direction on every bar. Stage (a) forecasts next-24h realized volatility
with a HAR model (and, in the `logit` model, the sign of the next H-bar return); stage (b) maps the
current bar to a discrete regime cell and is long only while that cell's TRAINING-window mean net
forward return is positive with some evidence, holding at least `min_hold` bars per entry.

## Pre-registration (written 2026-11-05 before any OOS run)

18 configs per symbol (36 runs, x2 costs). Up to 6 more per symbol may be added only if logged and
marked `post_hoc` below; the total stays <= 24 per symbol.

Common definitions (D = bars per day: 24 on 1h, 6 on 4h; W = 7D; M = 30D):

| piece | definition |
|---|---|
| per-bar realized variance | rv2_t = sum of squared 5m log close returns inside bar t (from the 5m cache) |
| HAR inputs (known at t close) | log sqrt(mean rv2 over the last D, W, M bars) |
| HAR target | log sqrt(mean rv2 over bars t+1..t+D) = next-24h realized vol; label end t1 = open of bar t+1+D |
| HAR fit | OLS with intercept on the fold's purged training rows; forecast `vf` for every row |
| vol bucket | `vf` vs the training rows' 40% / 80% quantiles: low / mid / high ("extreme") |
| trend_l | close > EMA(M bars) (30 days) |
| trend_s | log(close_t / close_{t-D}) > 0 (last 24h) |
| flow bucket | taker imbalance over the last 4h (`features.taker_imbalance`, 4 bars on 1h, 1 bar on 4h) vs the training rows' 10% / 90% quantiles: sell shock / normal / buy shock |
| return label | `labels.forward_return(df, H)` = log(open[t+1+H] / open[t+1]) |
| net label | return label minus a full round trip at base cost (2 x 5 bps = 10 bps) per label, regardless of how long the cell persists (conservative) |
| gate | for each regime cell, on the fold's purged training rows: n, mean net label, t_adj = mean / (std / sqrt(n / H)) (overlap-adjusted). Cell is ON if n / H >= 30, mean > 0 and t_adj >= 1.0 |
| position | enter at the first ON bar; hold at least `min_hold` bars; then exit at the first OFF bar (re-entry allowed at once). Caps entries at 3/day |
| purging | `walk_forward_splits` with t1 = max(return-label t1, HAR-target t1), default 1-day embargo; monthly test blocks |

Models (the regime variables that define the cells):

| model | cells | stage (a) |
|---|---|---|
| `har_trend` | trend_l (2) x trend_s (2) x vol bucket (3) = 12 | HAR vol forecast |
| `har_flow` | trend_l (2) x flow bucket (3) x vol bucket (3) = 18; the sell-shock cells are the reversal hypothesis | HAR vol forecast |
| `logit` | probability quintile (5) x vol bucket (3) = 15 | HAR vol forecast + `LogisticRegression(C=0.1, random_state=7)` on standardized [ret_D, ret_W, ret_M, log(close/EMA_M), vf, log rv_D - log rv_M, tib_4h, tib_D, volratio(D, M), vwapdev_D] predicting 1{return label > 0}. Quintile edges come from inner purged 5-fold OOS probabilities on the training rows (`purged_kfold(..., within=fold)`), so the cell means are not in-sample model outputs |

Horizons: H = 4h or 24h. min_hold = 8h for H = 4h and 24h for H = 24h.

| # | name | model | tf | H (bars) | min_hold (bars) | train |
|---|---|---|---|---|---|---|
| 1 | trend_1h_h4_exp | har_trend | 1h | 4 | 8 | expanding from 2017-09-01 |
| 2 | trend_1h_h24_exp | har_trend | 1h | 24 | 24 | expanding |
| 3 | trend_4h_h1_exp | har_trend | 4h | 1 | 2 | expanding |
| 4 | trend_4h_h6_exp | har_trend | 4h | 6 | 6 | expanding |
| 5 | flow_1h_h4_exp | har_flow | 1h | 4 | 8 | expanding |
| 6 | flow_1h_h24_exp | har_flow | 1h | 24 | 24 | expanding |
| 7 | flow_4h_h1_exp | har_flow | 4h | 1 | 2 | expanding |
| 8 | flow_4h_h6_exp | har_flow | 4h | 6 | 6 | expanding |
| 9 | logit_1h_h4_exp | logit | 1h | 4 | 8 | expanding |
| 10 | logit_1h_h24_exp | logit | 1h | 24 | 24 | expanding |
| 11 | logit_4h_h1_exp | logit | 4h | 1 | 2 | expanding |
| 12 | logit_4h_h6_exp | logit | 4h | 6 | 6 | expanding |
| 13 | trend_1h_h24_roll24 | har_trend | 1h | 24 | 24 | rolling 24 months |
| 14 | trend_4h_h6_roll24 | har_trend | 4h | 6 | 6 | rolling 24 months |
| 15 | flow_1h_h24_roll24 | har_flow | 1h | 24 | 24 | rolling 24 months |
| 16 | flow_4h_h6_roll24 | har_flow | 4h | 6 | 6 | rolling 24 months |
| 17 | logit_1h_h24_roll24 | logit | 1h | 24 | 24 | rolling 24 months |
| 18 | logit_4h_h6_roll24 | logit | 4h | 6 | 6 | rolling 24 months |

No threshold, quantile or hyper-parameter above is chosen from OOS results: the gate threshold
(t_adj >= 1.0, n / H >= 30), bucket quantiles and C are fixed here; everything fitted (HAR
coefficients, bucket edges, logit weights, quintile edges, cell means) uses only `fold.train` rows.
15m is not used: a 15m entry layer would add turnover against ~5 bps/side costs for no tested
benefit. A `--smoke` mode runs the same code with 2019 as the evaluation window (pre-OOS training
data only) and logs nothing; it is used for debugging only.

<!-- RESULTS:BEGIN -->
## Results (OOS 2020-01-01..2026-06-30, generated by the script, n_trials in registry after this run = 173)

Every run, sorted by base-cost Sharpe (`long bars in uptrend` = share of held bars decided while close > EMA 30d; `corr B&H` = daily correlation with buy-and-hold):

| config | tf | sym | Sharpe @5 | Sharpe @10 | CAGR @5 | MaxDD @5 | trades | trades/day | net bps/trade @5 / @10 | PF @5 | +months | worst year @5 | exposure | long bars in uptrend | corr B&H |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trend_4h_h6_exp | 4h | BTC | 1.13 | 0.87 | 41.0% | -48.1% | 599 | 0.25 | 44.4 / 34.4 | 1.44 | 51% | -32.3% | 37% | 88% | 0.60 |
| trend_1h_h24_exp | 1h | ETH | 0.91 | 0.72 | 40.5% | -47.3% | 648 | 0.27 | 45.0 / 35.0 | 1.33 | 51% | -16.5% | 34% | 80% | 0.67 |
| trend_1h_h24_exp | 1h | BTC | 0.89 | 0.57 | 31.5% | -46.4% | 807 | 0.34 | 28.1 / 18.0 | 1.28 | 46% | -34.8% | 45% | 86% | 0.64 |
| logit_4h_h6_exp | 4h | ETH | 0.89 | 0.81 | 29.4% | -37.3% | 187 | 0.08 | 104.3 / 94.2 | 1.72 | 37% | -6.2% | 9% | 46% | 0.48 |
| trend_4h_h6_exp | 4h | ETH | 0.84 | 0.68 | 33.5% | -47.0% | 493 | 0.21 | 50.1 / 40.0 | 1.34 | 51% | -14.6% | 27% | 80% | 0.62 |
| logit_1h_h24_exp | 1h | ETH | 0.80 | 0.74 | 26.4% | -44.9% | 176 | 0.07 | 104.4 / 94.2 | 1.62 | 38% | -5.1% | 8% | 45% | 0.50 |
| flow_1h_h24_exp | 1h | BTC | 0.74 | 0.46 | 19.6% | -51.1% | 551 | 0.23 | 26.5 / 16.5 | 1.26 | 42% | -38.1% | 33% | 93% | 0.51 |
| trend_1h_h24_roll24 | 1h | BTC | 0.73 | 0.49 | 18.6% | -39.1% | 465 | 0.20 | 29.8 / 19.8 | 1.30 | 36% | -0.0% | 24% | 80% | 0.45 |
| logit_4h_h6_roll24 | 4h | BTC | 0.68 | 0.44 | 14.9% | -44.0% | 384 | 0.16 | 28.3 / 18.2 | 1.30 | 46% | -13.1% | 18% | 52% | 0.45 |
| flow_4h_h6_exp | 4h | BTC | 0.67 | 0.44 | 17.2% | -46.6% | 457 | 0.19 | 29.4 / 19.4 | 1.26 | 41% | -31.4% | 34% | 96% | 0.52 |
| logit_4h_h1_exp | 4h | ETH | 0.61 | 0.55 | 8.2% | -20.2% | 55 | 0.02 | 105.7 / 95.6 | 1.81 | 12% | -5.2% | 1% | 40% | 0.16 |
| trend_4h_h6_roll24 | 4h | BTC | 0.55 | 0.33 | 12.1% | -39.3% | 391 | 0.16 | 25.2 / 15.1 | 1.23 | 42% | -12.8% | 22% | 80% | 0.44 |
| logit_4h_h6_exp | 4h | BTC | 0.50 | 0.30 | 9.0% | -52.6% | 282 | 0.12 | 24.4 / 14.4 | 1.25 | 35% | -15.2% | 14% | 50% | 0.37 |
| trend_1h_h4_exp | 1h | BTC | 0.50 | 0.45 | 3.2% | -11.0% | 24 | 0.01 | 97.0 / 86.9 | 2.00 | 3% | 0.0% | 1% | 100% | 0.11 |
| trend_4h_h6_roll24 | 4h | ETH | 0.49 | 0.35 | 12.2% | -60.0% | 356 | 0.15 | 35.2 / 25.2 | 1.21 | 38% | -22.5% | 20% | 69% | 0.49 |
| logit_1h_h24_roll24 | 1h | BTC | 0.47 | 0.22 | 9.0% | -49.2% | 397 | 0.17 | 18.4 / 8.4 | 1.19 | 45% | -28.2% | 18% | 51% | 0.41 |
| logit_1h_h4_exp | 1h | BTC | 0.31 | 0.00 | 1.2% | -6.1% | 84 | 0.04 | 10.1 / 0.1 | 1.29 | 8% | 0.0% | 1% | 60% | 0.11 |
| flow_4h_h6_roll24 | 4h | ETH | 0.22 | 0.10 | 2.0% | -52.3% | 234 | 0.10 | 19.1 / 9.1 | 1.11 | 24% | -16.7% | 14% | 78% | 0.38 |
| logit_4h_h1_exp | 4h | BTC | 0.19 | 0.11 | 1.0% | -7.7% | 32 | 0.01 | 23.4 / 13.3 | 1.32 | 5% | -7.2% | 0% | 72% | 0.13 |
| flow_1h_h24_roll24 | 1h | BTC | 0.17 | -0.01 | 1.5% | -52.0% | 231 | 0.10 | 9.1 / -0.9 | 1.08 | 14% | -14.5% | 12% | 85% | 0.32 |
| trend_4h_h1_exp | 4h | ETH | 0.16 | 0.15 | 1.0% | -37.3% | 21 | 0.01 | 52.1 / 42.1 | 1.27 | 4% | 0.0% | 1% | 0% | 0.31 |
| flow_4h_h1_exp | 4h | BTC | 0.16 | 0.13 | 0.4% | -3.3% | 5 | 0.00 | 50.7 / 40.7 | 1.77 | 1% | -1.8% | 0% | 36% | 0.07 |
| logit_4h_h6_roll24 | 4h | ETH | 0.16 | 0.03 | -2.0% | -54.6% | 330 | 0.14 | 8.0 / -2.0 | 1.05 | 29% | -23.4% | 16% | 46% | 0.52 |
| flow_4h_h6_roll24 | 4h | BTC | 0.10 | -0.08 | -0.5% | -56.5% | 257 | 0.11 | 5.6 / -4.4 | 1.04 | 18% | -25.1% | 16% | 92% | 0.35 |
| logit_1h_h24_roll24 | 1h | ETH | 0.06 | -0.09 | -4.6% | -58.2% | 329 | 0.14 | 0.6 / -9.4 | 1.00 | 31% | -26.7% | 15% | 52% | 0.50 |
| logit_1h_h24_exp | 1h | BTC | 0.03 | -0.17 | -1.4% | -56.1% | 269 | 0.11 | 1.3 / -8.7 | 1.01 | 29% | -31.1% | 12% | 43% | 0.32 |
| trend_1h_h24_roll24 | 1h | ETH | 0.03 | -0.12 | -7.2% | -80.0% | 413 | 0.17 | 1.0 / -9.0 | 1.01 | 29% | -26.8% | 22% | 67% | 0.51 |
| trend_1h_h4_exp | 1h | ETH | 0.01 | -0.01 | -2.3% | -41.8% | 22 | 0.01 | -42.5 / -52.4 | 0.82 | 3% | -14.1% | 1% | 0% | 0.31 |
| flow_1h_h4_exp | 1h | BTC | 0.00 | 0.00 | 0.0% | 0.0% | 0 | 0.00 | 0.0 / 0.0 | 0.00 | 0% | 0.0% | 0% | nan% | nan |
| trend_4h_h1_exp | 4h | BTC | 0.00 | 0.00 | 0.0% | 0.0% | 0 | 0.00 | 0.0 / 0.0 | 0.00 | 0% | 0.0% | 0% | nan% | nan |
| flow_4h_h1_exp | 4h | ETH | -0.03 | -0.09 | -0.0% | -2.5% | 5 | 0.00 | -4.5 / -14.5 | 0.92 | 1% | -1.5% | 0% | 30% | 0.02 |
| flow_1h_h24_exp | 1h | ETH | -0.03 | -0.16 | -4.8% | -64.2% | 242 | 0.10 | -3.3 / -13.3 | 0.98 | 13% | -31.7% | 13% | 91% | 0.35 |
| flow_4h_h6_exp | 4h | ETH | -0.03 | -0.14 | -4.1% | -43.3% | 176 | 0.07 | -3.1 / -13.1 | 0.98 | 14% | -17.9% | 11% | 94% | 0.31 |
| flow_1h_h4_exp | 1h | ETH | -0.04 | -0.12 | -1.3% | -26.6% | 62 | 0.03 | -6.0 / -16.0 | 0.95 | 21% | -11.5% | 1% | 20% | 0.17 |
| logit_1h_h4_exp | 1h | ETH | -0.22 | -0.26 | -3.2% | -35.3% | 29 | 0.01 | -56.0 / -65.9 | 0.78 | 3% | -21.1% | 0% | 34% | 0.21 |
| flow_1h_h24_roll24 | 1h | ETH | -0.31 | -0.44 | -14.3% | -82.2% | 293 | 0.12 | -22.4 / -32.3 | 0.88 | 18% | -37.3% | 15% | 76% | 0.40 |

Both symbols side by side, sorted by the weaker symbol:

| config | BTC Sharpe @5 | ETH Sharpe @5 | min |
|---|---|---|---|
| trend_1h_h24_exp | 0.89 | 0.91 | 0.89 |
| trend_4h_h6_exp | 1.13 | 0.84 | 0.84 |
| logit_4h_h6_exp | 0.50 | 0.89 | 0.50 |
| trend_4h_h6_roll24 | 0.55 | 0.49 | 0.49 |
| logit_4h_h1_exp | 0.19 | 0.61 | 0.19 |
| logit_4h_h6_roll24 | 0.68 | 0.16 | 0.16 |
| flow_4h_h6_roll24 | 0.10 | 0.22 | 0.10 |
| logit_1h_h24_roll24 | 0.47 | 0.06 | 0.06 |
| logit_1h_h24_exp | 0.03 | 0.80 | 0.03 |
| trend_1h_h24_roll24 | 0.73 | 0.03 | 0.03 |
| trend_1h_h4_exp | 0.50 | 0.01 | 0.01 |
| trend_4h_h1_exp | 0.00 | 0.16 | 0.00 |
| flow_4h_h1_exp | 0.16 | -0.03 | -0.03 |
| flow_1h_h24_exp | 0.74 | -0.03 | -0.03 |
| flow_4h_h6_exp | 0.67 | -0.03 | -0.03 |
| flow_1h_h4_exp | 0.00 | -0.04 | -0.04 |
| logit_1h_h4_exp | 0.31 | -0.22 | -0.22 |
| flow_1h_h24_roll24 | 0.17 | -0.31 | -0.31 |

Calendar-year net returns @5 bps:

| config | sym | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| trend_4h_h6_exp | BTC | 92% | 228% | -32% | 31% | 66% | -1% | 1% |
| trend_1h_h24_exp | ETH | 68% | 86% | 14% | 17% | 52% | 73% | -16% |
| trend_1h_h24_exp | BTC | 132% | 144% | -35% | 38% | 39% | -13% | -5% |
| logit_4h_h6_exp | ETH | 42% | 71% | 7% | -1% | 35% | 65% | -6% |
| trend_4h_h6_exp | ETH | 54% | 151% | 27% | -5% | -7% | 78% | -15% |
| logit_1h_h24_exp | ETH | 22% | 45% | 29% | 3% | 36% | 53% | -5% |
| flow_1h_h24_exp | BTC | 77% | 147% | -38% | 17% | 9% | -5% | -2% |
| trend_1h_h24_roll24 | BTC | 52% | 39% | -0% | 12% | 23% | 1% | 4% |
| logit_4h_h6_roll24 | BTC | 98% | 18% | 18% | 6% | -6% | -13% | 3% |
| flow_4h_h6_exp | BTC | 38% | 127% | -31% | 19% | 41% | -19% | -4% |
| logit_4h_h1_exp | ETH | -5% | 13% | -3% | 0% | 12% | 43% | 0% |
| trend_4h_h6_roll24 | BTC | 45% | 19% | -13% | 11% | 16% | 12% | -4% |
| logit_4h_h6_exp | BTC | 68% | -15% | -14% | 5% | 29% | 8% | -3% |
| trend_1h_h4_exp | BTC | 16% | 6% | 0% | 0% | 0% | 0% | 0% |
| trend_4h_h6_roll24 | ETH | 31% | 96% | -21% | -2% | -4% | 42% | -22% |
| logit_1h_h24_roll24 | BTC | 56% | -9% | 32% | 21% | 9% | -28% | -1% |
| logit_1h_h4_exp | BTC | 8% | 0% | 0% | 0% | 0% | 0% | 0% |
| flow_4h_h6_roll24 | ETH | 29% | -5% | 3% | 9% | -17% | -0% | 0% |
| logit_4h_h1_exp | BTC | -7% | 21% | 0% | 0% | 0% | 2% | -7% |
| flow_1h_h24_roll24 | BTC | 46% | 15% | -15% | 0% | -10% | -14% | 0% |
| trend_4h_h1_exp | ETH | 2% | 0% | 5% | 0% | 0% | 0% | 0% |
| flow_4h_h1_exp | BTC | -0% | 0% | 0% | -1% | 0% | -2% | 6% |
| logit_4h_h6_roll24 | ETH | -13% | 29% | -21% | 0% | 7% | 21% | -23% |
| flow_4h_h6_roll24 | BTC | 42% | 12% | -25% | 0% | -3% | -15% | 0% |
| logit_1h_h24_roll24 | ETH | 22% | 5% | -9% | 0% | -2% | -27% | -12% |
| logit_1h_h24_exp | BTC | 42% | -29% | -31% | -3% | 38% | 1% | -3% |
| trend_1h_h24_roll24 | ETH | 27% | 0% | -25% | -1% | -27% | 14% | -22% |
| trend_1h_h4_exp | ETH | -14% | 0% | -0% | 0% | 0% | 0% | 0% |
| flow_1h_h4_exp | BTC | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| trend_4h_h1_exp | BTC | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| flow_4h_h1_exp | ETH | -1% | -2% | 2% | -0% | 0% | 0% | 0% |
| flow_1h_h24_exp | ETH | 0% | 14% | -18% | -2% | -32% | 16% | 0% |
| flow_4h_h6_exp | ETH | 6% | -18% | 1% | 1% | -17% | 3% | 0% |
| flow_1h_h4_exp | ETH | 6% | -11% | -7% | 3% | 9% | -5% | 0% |
| logit_1h_h4_exp | ETH | -1% | -21% | 0% | 0% | 0% | 4% | 0% |
| flow_1h_h24_roll24 | ETH | 19% | 8% | -37% | -5% | -29% | -33% | 0% |

Net log P&L by deciding regime cell (top 6 by size) for the three best configs:

* trend_1h_h24_exp BTC: Lup_Sup_Vmid +0.94, Ldn_Sdn_Vhi +0.67, Lup_Sup_Vlo +0.63, Ldn_Sup_Vlo -0.08, Lup_Sdn_Vhi -0.06, Lup_Sdn_Vmid +0.06
* trend_1h_h24_exp ETH: Lup_Sup_Vlo +1.18, Ldn_Sdn_Vhi +0.99, Lup_Sup_Vmid +0.78, Ldn_Sup_Vhi -0.34, Lup_Sdn_Vlo -0.27, Lup_Sdn_Vmid +0.22
* trend_4h_h6_exp BTC: Lup_Sup_Vlo +1.04, Lup_Sup_Vmid +0.73, Ldn_Sdn_Vhi +0.57, Lup_Sdn_Vmid +0.53, Lup_Sdn_Vlo -0.19, Lup_Sup_Vhi +0.13
* trend_4h_h6_exp ETH: Ldn_Sdn_Vhi +1.30, Lup_Sup_Vlo +0.98, Lup_Sdn_Vlo -0.62, Lup_Sup_Vmid +0.34, Lup_Sdn_Vhi +0.30, Ldn_Sup_Vhi -0.30
* logit_4h_h6_exp BTC: P2_Vlo +0.22, P4_Vmid +0.22, P1_Vhi -0.11, P3_Vmid +0.10, P1_Vlo +0.09, P5_Vlo -0.08
* logit_4h_h6_exp ETH: P5_Vhi +1.30, P3_Vhi +0.26, P4_Vhi +0.23, P4_Vmid +0.20, P5_Vmid +0.15, P1_Vmid -0.15
<!-- RESULTS:END -->

## post_hoc runs

None. No config was added after the OOS run (18 per symbol, 36 runs, 72 registry lines).

## Assessment (written after the single logged OOS run, 2026-11-05)

**It does not beat the best baseline.** The best config on the weaker symbol is `trend_1h_h24_exp`
(Sharpe 0.89 BTC / 0.91 ETH at 5 bps, 0.57 / 0.72 at 10 bps); the best single run is
`trend_4h_h6_exp` on BTC (1.13 / 0.87). The baseline to beat is `ema_px_4h_180` / `tsmom_4h_180`
(1.35 BTC, 1.47 ETH). Against buy-and-hold (0.84 BTC / 0.89 ETH) the best config wins by only
0.05 / 0.03 Sharpe, about 0.1 against vol-matched buy-and-hold, with a smaller drawdown (-46% / -47%
vs -77% / -79%) but a much larger one than the 30-day trend rule (-38% / -39%). `promotion.evaluate`
on both configs: FAIL (Sharpe < 1 on ETH, t-stat 2.3-2.9 < 3, MaxDD, positive months 46-51%, worst
year, DSR ~0 at N = 174, below the best baseline; PBO 0.17 BTC / 0.24 ETH). Permutation p passes
(0.002-0.03) except `trend_4h_h6_exp` ETH (0.054), so there is timing skill, just not enough.

**Where the edge comes from.** Two places, neither new. (1) 80-88% of the held bars in the trend
configs are decided while close > EMA(30d): the gate mostly re-learns the trend baseline (cells
`Lup_Sup_Vlo` and `Lup_Sup_Vmid`, trend up + vol not extreme) and adds churn to it (0.25-0.34
trades/day vs 0.09). (2) A capitulation cell, `Ldn_Sdn_Vhi` (below the 30-day EMA, down over the
last 24h, top-20% forecast vol), is consistently ON in training and earns +0.57 to +1.30 log P&L
OOS: buying the day after a crash. The taker-flow reversal hypothesis failed: every `har_flow`
config is <= 0 on ETH, and on BTC its held bars are 92-96% uptrend (it is the trend rule with fewer
trades). The `logit` model trades rarely (8-14% exposure) and works only on ETH (0.80-0.89), driven
by one cell (top probability quintile x high vol); on BTC it is ~0. Stage (a) adds little beyond a
vol bucket: the HAR forecast only decides low / mid / high.

**Cost and timeframe sensitivity.** Short horizons (H = 4h) are dead on both timeframes: the
pre-registered hurdle of one 10 bps round trip per 4h label leaves almost no cell ON (0-84 trades in
6.5 years; three runs never trade). So the intended 0.5-3 trades/day was never reached (max 0.34);
the gate, not the min-hold cap, set turnover. At H = 24h, 1h and 4h decision cadence give similar
results (min Sharpe 0.89 vs 0.84). Doubling cost to 10 bps cuts 0.15-0.3 Sharpe off the trend
configs (net 28-50 bps/trade at base), about three times the baseline's loss, because the gate
flips 3x as often for an edge per trade 4-6x smaller. Rolling 24-month training is worse than
expanding in 5 of 6 pairs: cell means need the long history, and the 24-month windows chase the last
regime (`trend_1h_h24_roll24` ETH: -80% MaxDD).

**Disaster years.** 2022 for BTC: -32% to -38% in every BTC trend / flow config with expanding
training (the baseline's worst year is -12%); the capitulation cell kept buying through the bear
market. `flow_1h_h24_roll24` ETH lost 37% in 2022 and 33% in 2025. The ETH trend configs' worst
year is -15% to -16% (2026 H1). No config has a worst year above -10% while trading >= 500 times.

**Verdict:** regime gating as specified is a noisier, costlier version of the 30-day trend rule
plus a crash-buying cell. It is not a candidate. If anyone revisits it, the two pieces worth
isolating are the capitulation cell on its own (as a separate pre-registered family) and an
amortised cost hurdle for short horizons; both would be new trials.
