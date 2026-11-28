# Phase 2b — setup × trigger grid: summary (Dev 2020-01 → 2025-07 only)

7 setups × 11 triggers × 2 targets = 154 cells (PLAN §13). 30-min / 50-pt caps, $6.50 RT + 1-tick slippage, structural
stop (10-bar pullback extreme, ≥ 0.02 ATR). Full table: [`phase2b_grid.txt`](phase2b_grid.txt) / [`phase2b_cells.csv`](phase2b_cells.csv).

## Headline: setup + trigger beats either alone
Two cells pass the pre-registered bar (edge t ≥ 3, net > 0 in both halves):

| Setup → trigger → target | Trades/mo | Win % | Net pts/trade | ≈ $/mo (5 MNQ) | Edge t | Years + | Halves | By year (net pts) |
|---|---|---|---|---|---|---|---|---|
| **S1 ORB continuation → close back above/below EMA9 → 1.5R** | 6.7 | 51 | **+3.81** | **+$256** | 3.17 | **6/6** | +2.45 / +5.06 | 1.8, 3.7, 4.7, 3.5, 3.8, 6.8 |
| **S4 IB fade (narrow IB) → EMA9/21 cross → IB mid** | 1.4 | 70 | **+4.65** | +$67 | 3.29 | 5/6 | +4.01 / +5.18 | 2.0, 4.5, 9.9, −2.1, 8.5, 3.4 |

Context: with 154 cells tested, the best cell would typically reach t ≈ 2.7–2.9 by luck. These two (3.2, 3.3) are
above that, but not by a wide margin. The OOT sample and the lockbox decide.

## What else the grid shows
- **Every setup has at least one trigger that turns it positive**, while 5 of 7 setups lose money entered immediately
  (T0). The setup gives direction and target; the trigger fixes the timing.
- **Triggers vs immediate entry (median across setups):** 1m iFVG inside a 5m FVG +3.1 pts, 1m iFVG +0.5, EMA9/21 cross
  +0.26, EMA9 reclaim +0.23, Supertrend 1m +0.23, RSI(2) +0.18, stochastic 1m +0.11; stochastic 5m −1.0, engulf/pin −0.15.
- The iFVG-inside-5m trigger is the best on average, but it fires rarely (≈ 1–4/mo per setup). **Pooled over all
  setups, its profit is almost all from 2022** (+10.6 pts/trade that year, ≈ 0 or negative in 2021, 2023, 2025), so it is
  not reliable on its own yet.
- **Walk-forward choice per setup** (best trigger × target picked on prior years, scored on the next year, 2022–25): positive
  for 6 of 7 setups (S4 IB fade +4.3, S7 hour magnet +3.3, S1 +3.3, S5 PD magnet +2.7). None is individually
  significant on ~30–150 out-of-fold trades. S6 (8am third) is negative.
- **The 50-pt stop cap** skips about half of the ORB-continuation entries (wide pullbacks around 09:45–10:00).

## Comparison only (not used for selection): continuation held longer
| S1 ORB continuation | 30-min cap, 1.5R | Held to 11:00, stop only | Held to 15:59, stop only |
|---|---|---|---|
| EMA9 reclaim trigger | +3.81 pts, $256/mo | +3.50 pts, $235/mo | **+8.33 pts, $560/mo, 6/6 yrs** |
| No trigger (T0) | +0.41 | +2.11 | +5.12 |

ORB-continuation momentum keeps paying beyond 30 minutes, the same pattern as the repo's ATR-breakout study. Your
30-minute rule leaves money on the table for this setup. That's your call; it stays as reported context.
