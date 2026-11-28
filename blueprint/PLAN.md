# NQ RTH scalping system — research plan (pre-registration)

*Agreed 28 Nov 2026. This file is written **before** any test is run. Everything below — splits, definitions,
variant grids, pass criteria — is fixed now so results can't steer the design. Changes after testing starts get
logged at the bottom with a date and reason.*

Goal: a **systematic blueprint** (bias → entry → stop/target) for NQ intraday scalps, validated out of time, then
coded as a **TradingView (Pine v5) indicator** and a **NinjaTrader 8 (NinjaScript) indicator**.

---

## 1. Trading constraints

| Item | Rule |
|---|---|
| Instrument / size | MNQ, 5 contracts ($2/pt/contract → **$10 per NQ point**) |
| Entry window | RTH only: signals 09:31–15:44 ET; flat by 16:00 |
| Max hold | **30 minutes** (time stop), shorter time stops tested |
| Max stop / max target | **50 points** each; trades whose structural stop would exceed 50 pts are skipped, not capped |
| Costs | **$6.50 per round trip for the 5-lot = 0.65 pt** (to confirm, see §11) |
| Slippage | +1 tick (0.25 pt) on every market entry, stop exit and time exit; limit entries/targets fill only if price trades **through** by 1 tick |
| Fill order | 1-minute bars: if stop and target are both inside one bar, **stop first** (conservative) |

## 2. Data and splits

| Sample | Period | Sessions | Order flow | Status | Use |
|---|---|---|---|---|---|
| **Dev** | 2020-01-02 → 2025-07-09 | 1,427 | yes | open | all design, fitting, variant choice (walk-forward inside) |
| **OOT** | 2025-07-10 → 2026-11-24 | 357 | yes | **partly seen** (earlier iFVG and stats work scored Mar 25–Nov 26) | 2nd-tier check, run once per frozen system |
| **Lockbox** | NQ 2012-01-01 → 2019-12-31 | ~2,000 | **no** (OHLCV only) | **untouched** for entries | final check, run **once**, after OOT |

Rules
- Stats used as features/priors (level hit rates, IB/ORB width classes, range fill) are **recomputed on Dev only**.
  The repo's published tables used 2010–2026 and are not used as model inputs.
- Order-flow features can only be validated Dev → OOT. Any system that needs them is tested on the lockbox
  without them (or not at all, stated as such).
- **Volatility drift:** median RTH range was 30 pts in 2012, 71 in 2019, 184 in 2023. All distances (stops,
  targets, "near a level") are defined in **ATR units**, with the 50-pt cap applied on top. The lockbox is reported
  twice: (a) actual costs in points and (b) costs scaled to the same fraction of ATR as Dev, to separate "the edge
  exists" from "the edge paid after fixed costs in a low-volatility market".
- ATR = 14-session ATR of the **RTH** range (back-adjusted across rolls), known at 09:30.
- Excluded sessions: `partial_sess` days (holidays, early closes, outages) — as defined in the dataset.

## 3. Levels (all known at the moment of use, no lookahead)

| Level | Definition |
|---|---|
| PDH / PDL | prior **RTH** session high/low |
| PWH / PWL | prior week's high/low (Sun 18:00 → Fri 17:00) |
| Session highs/lows | Asia 18:00–02:00, London 02:00–08:00, overnight 18:00–09:30 (all ET) |
| RTH VWAP | session VWAP anchored 09:30 |
| Daily (Globex) VWAP | session VWAP anchored 18:00 (prior evening) |
| Weekly VWAP | anchored Sunday 18:00 |
| IB high / low | 09:30–10:29 high/low, **active only from 10:30** |
| Prev hour high/low | prior completed clock hour (ET) |
| Prev 4h high/low | prior completed 4h candle, anchored 18:00 (18, 22, 02, 06, 10, 14) |

"Near a level": the level lies inside the signal's FVG zone, or within **d × ATR** of the entry price.
Grid d ∈ {0.03, 0.06}. (≈ 5 / 11 pts at today's ATR.)

## 4. iFVG definition

- **FVG** on timeframe T (1m, 5m, 15m; 5m/15m bars clock-aligned): bullish if low[0] > high[−2], zone =
  (high[−2], low[0]); bearish mirror. Minimum size: 1 tick (variant: ≥ 0.02 × ATR).
- **Inversion (iFVG):** a bar on T **closes** through the far side of an unfilled FVG. Bearish FVG closed above →
  bullish iFVG; bullish FVG closed below → bearish iFVG. FVGs older than N bars of T expire (N = 60 on 1m, 24 on
  5m, 16 on 15m). Variant: inversion on a wick through (“traded through”) instead of a close.
- 5m/15m inversions are known only **when the 5m/15m bar closes**. Entries are always executed on 1m.

## 5. Layer 1 — Bias / context (fitted on Dev only)

Bias = **which level price reaches first / what kind of day this is**, not "up or down" (earlier work: direction
labels sit at 51–56% vs a ~53% base).

Labels (one model each):
- **B1 Level race, 09:30 → 11:00 and 11:00 → 16:00:** nearest level above vs nearest below (from §3) — which is hit first.
- **B2 Day type at 10:30:** RTH range ends > 1.0 × IB beyond IB (trend) vs rotation (stays within IB ± 0.5 × IB width).
- **B3 Hourly:** next clock hour takes the near vs far extreme of the prior hour (decision points 10:00–15:00).

Features (all known at decision time): open vs prior-day range and prior value area (quartiles), gap / ATR,
08:00 position in overnight range (thirds), overnight range / ATR, IB and 15-min ORB width / ATR, day and week range
fill percentiles, price vs RTH VWAP, daily (18:00) VWAP and weekly VWAP (in ATR), prior day type, day of week; Dev/OOT only: opening
30-min cumulative delta / volume.

Models, in order; each must beat the one above it on **walk-forward log-loss / Brier** to be kept:
1. Base rate + Dev conditional tables (the existing stats approach)
2. Regularised logistic regression
3. KNN (standardised features, ≤ 8 features, k ∈ {25, 50, 100}, distance-weighted)
4. Gaussian HMM on daily features (returns, range/ATR, overnight range/ATR), 2–3 states, **filtered** state
   probabilities only — used as a regime gate and as a feature
5. Gradient boosting (shallow trees) and a small MLP (1 hidden layer, ≤ 16 units, dropout/weight decay) — B3 only,
   where samples are ~6× more numerous

Walk-forward inside Dev: expanding window, yearly refits, 5-session purge/embargo between train and test.

## 6. Layer 2 — Entry families

Every family produces **candidate trades**; direction must agree with Layer 1 where the family says so.

| ID | Family | Trigger (1m execution) |
|---|---|---|
| E1 | iFVG at a level | 1m iFVG near a level (§3), direction of the inversion. Variants: TF of inversion {1m, 5m, 15m}; MTF confluence {none, 1m iFVG inside aligned 5m FVG, inside aligned 15m FVG}; entry {close of inversion bar, limit retest of the inverted zone within 10 bars} |
| E2 | Sweep & reclaim / CISD | price trades through a level by ≥ 1 tick then a 1m (variant 5m) close back on the other side within 5 bars; enter in the reclaim direction |
| E3 | Failed IB/ORB break | after 10:30, close outside IB then close back inside within N ∈ {5, 15} min; enter toward IB mid. Narrow-IB days only (IB < Dev 33rd pct of IB/ATR) and variant all days |
| E4 | VWAP trend pullback | B2 says trend (p > threshold) or HMM in trend state; price pulls back to VWAP (or ±0.5σ band) in trend direction and prints a 1m iFVG / engulfing in trend direction |
| E5 | Order-flow filter (Dev/OOT only) | applied to E1–E4: skip fades into a level when the 5 bars before the touch show aggressive delta toward the level in the top quintile (rule that held in both earlier periods) |
| E0 | Controls | same timing and direction rules with random entries (direction-matched) — every family is scored against this |

## 7. Layer 3 — Stops and targets

| Stop | Target | Time |
|---|---|---|
| S1 structural (beyond FVG zone / sweep extreme + 1 tick) | T1 fixed R ∈ {1.0, 1.5, 2.0} | time stop ∈ {10, 20, 30} min |
| S2 ATR: {0.05, 0.08} × ATR | T2 next level from §3 in trade direction | |
| | T3 **EV-optimal level**: argmax over levels of P(hit) × dist − (1 − P(hit)) × stop − costs, P from Layer 1 / Dev tables | |
| | T4 scale: 3 contracts at 1R, 2 run to T2 with stop to breakeven | |

All capped: stop ≤ 50 pts (else skip), target ≤ 50 pts (else capped at 50), hold ≤ 30 min.
Labels for the meta-model are the **triple-barrier** outcomes of these exits.

## 8. Layer 4 — Meta-label filter (ML)

A classifier (logistic → gradient boosting) predicts P(candidate trade hits target before stop/time) from Layer-1
probabilities, HMM state, level type, distance to the next level, time of day, iFVG size/ATR, (Dev/OOT) delta
features. Take only trades with P above a threshold chosen in Dev walk-forward. Purged walk-forward as in §5.

## 9. Variant budget, selection and pass criteria

- Max **≤ 48 variants per entry family** (entry × stop × target × time), ≤ 300 combinations in total.
  The total count tested is logged and used for the deflated Sharpe / multiple-testing haircut.
- Variant chosen per family on **Dev walk-forward** (last 2 Dev years out of fold), never on OOT.
- Then frozen → OOT once → lockbox once.

A system **passes** if all hold:

| Check | Dev (walk-forward folds) | OOT | Lockbox |
|---|---|---|---|
| Net pts/trade (after costs + slippage) | > 0, t ≥ 2.0 | > 0 | > 0 on vol-scaled costs; reported on actual costs |
| Profit factor | ≥ 1.15 | ≥ 1.05 | ≥ 1.05 (vol-scaled) |
| Beats random-entry control (E0) | yes, t ≥ 2 | yes | yes |
| Years positive | ≥ 4 of 6 | — | ≥ 5 of 8 |
| Frequency | ≥ 4 trades/month | — | — |
| Deflated Sharpe | > 0.95 probability | — | — |

Also reported: win %, avg R, max drawdown ($, 5 MNQ), worst day, trades/day, MAE/MFE, result by time of day and
by level type.

## 10. Phases

0. **Engine** — level builder, FVG/iFVG detector (1m/5m/15m), 1m fill simulator with the §1 rules, unit tests and
   chart spot-checks against TradingView for a handful of days. *(no results looked at)*
1. **Bias layer** — B1–B3, models 1–5, Dev walk-forward. Output: calibration plots, which features matter.
2. **Entries raw** — E1–E4 + E0 with a neutral exit (S1 / T1 1.5R / 30 min), with and without bias agreement.
3. **Exits** — S/T/time grid on the families that beat E0.
4. **Meta-label** — on the surviving families.
5. **Freeze → OOT → lockbox.** Results reported whatever they are.
6. **Blueprint + indicators** — written rulebook (pre-market checklist, bias read, entry triggers, exits, no-trade
   rules) and Pine v5 + NinjaScript indicators plotting levels, bias, iFVGs and signals with stop/target lines.
   Pine and NinjaScript signals are checked against the Python backtest on the same days.

## 11. Open items (defaults used unless changed)

1. **Costs:** $6.50 round trip assumed for the whole 5-lot (0.65 pt). If it is $6.50 *per contract*, costs are
   3.25 pts per trade — that changes which setups are viable.
2. **Session windows:** Asia 18:00–02:00 / London 02:00–08:00 as above.
3. **VWAP anchors:** both 09:30 RTH and 18:00 Globex daily VWAP are levels (confirmed).
4. **iFVG inversion:** candle close through (main) with wick-through as a variant.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-28 | Plan created | — |
| 2026-11-28 | Defaults in §11 confirmed; added daily VWAP anchored 18:00 alongside RTH VWAP | User request, before any test |
