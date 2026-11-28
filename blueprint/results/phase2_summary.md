# Phase 2 — entry families: summary (Dev 2020-01 → 2025-07 only)

Neutral exit for every family: structural stop (≥ 0.02 ATR), 1.5R target, 30-min time stop, 5 MNQ, $6.50 RT +
1-tick slippage (≈ 1.15 pts of friction per trade). Edge = net minus a **direction-neutral control** (same bar, same
bracket, average of long and short). An earlier control (random entries ±60 min) was found to look ahead and was
replaced mid-run; see the PLAN change log. Full tables: [`phase2_entries.txt`](phase2_entries.txt),
[`phase2_variants.csv`](phase2_variants.csv).

## Headline
**No variant passes the pre-registered promotion rule unfiltered** (edge t ≥ 2 and net > 0 in both Dev halves).
Every unfiltered variant except one is net negative after costs. The exception, daily-VWAP pullback + 1m iFVG, is
+0.02 pts.

| Finding | Evidence |
|---|---|
| Plain 1m iFVG at a level has a **real but tiny** directional edge, far below costs | +0.17–0.19 pts/trade over control, t ≈ 2.8 on 35–46k trades; net −1.11 |
| **MTF confluence matters:** 1m iFVG inside an aligned 5m or 15m FVG | edge +0.4 to +1.2 pts in all 8 variants vs +0.2 without |
| Trading the 5m / 15m iFVG directly adds nothing | 5m edge ≈ 0; 15m +0.6–0.8 (t < 1.8) with retest entries negative |
| **Limit / retest entries are worse** almost everywhere (fills come when price runs against you) | e.g. 1m iFVG retest −1.04 pts vs control, t −12.5; sweep retest −2.00. Exception: 1m-inside-5m |
| Sweep & reclaim (E2), failed IB break (E3), VWAP pullback (E4) | no significant edge unfiltered; E3 flips sign between halves (h1 −3.8, h2 +2.8 pts) |
| **A1 "trade toward the nearer level" filter helps consistently** | edge up in 25 of 36 variants (median +0.21 pts); +0.6 to +1.5 on all four 1m-in-5m variants |
| E5 order-flow filter | no effect (median +0.002 pts) |
| A2 hourly-agreement filter | too few trades to judge (10–180 per variant, 2022+) |

## Only combinations that pass the rule with a filter applied

| Setup | Trades/mo | Win % | Net pts | $/mo (5 MNQ) | Edge (t) | Halves (net) | Years + |
|---|---|---|---|---|---|---|---|
| 1m iFVG inside aligned 5m FVG, retest, d 0.06, **+A1** | 11.4 | 42.5 | +0.83 | ≈ +$95 | +2.45 (2.93) | +1.07 / +0.60 | 4/6 |
| same, d 0.03, +A1 | 7.9 | 42.7 | +0.82 | ≈ +$65 | +2.65 (2.69) | +0.66 / +0.96 | 4/6 |
| RTH-VWAP pullback + engulfing, **+A1** | 4.6 | 45.3 | +1.34 | ≈ +$60 | +2.24 (2.18) | +2.39 / +0.33 | 3/6 |

Caveats
- **216 variant × filter cells were tested.** Under no edge at all, the best of 216 correlated cells would
  typically reach t ≈ 2.8–3.0. The top result (t 2.9–3.1) is at that level, so on its own it isn't proof.
- What argues for the 1m-in-5m + A1 setup: confluence helps in every variant, A1 helps in all four 1m-in-5m
  variants, and it matches the earlier repo finding (1m iFVG inside a higher-TF FVG near a level was the only setup
  that held across periods).
- The VWAP + engulfing result is fragile. Its sister variants react to A1 in opposite directions (+2.3 vs −1.2).
- At the neutral exit the money is small (≈ $60–95/month on 5 MNQ). Phase 3 (exits) is where this either improves
  or doesn't.
