# NQ NY RTH plan — research inventory

*Compiled 21 Nov 2026. Evaluation only: nothing has been built or re-run yet.*
*Sources: 8 PDFs, nqstats.com (7 modules), 2 AMT articles, your 2 summary files (`NQ_trading_plan.md`, `NQ_best_stats.md`), 16 notebooks, 7 years of 1-min data.*

---

## 1. Inventory

### Data
| Dataset | Span | Columns | Where |
|---|---|---|---|
| `NQ_ticks_YYYY_1min.csv` ×7 | 2 Jan 2020 → 29 Apr 2026 | OHLC, volume, **buy/sell vol, delta, trades**, ET timestamp with offset | uploaded here, quality-checked |
| `nq_15y_1min.csv` | Jun 2010 → Apr 2026 | Date, Time, OHLC (+Volume) | your desktop only, not uploaded |

**Data rules agreed so far (for the uploaded files)**
- Rolls are unadjusted. Switch at 17:00 on roll day in 2020–24; at the 18:00 open on roll Monday from Jun 2025. Multi-day lookbacks must use same-contract levels.
- Exclude from RTH stats: stock-market holidays (210-bar days), 28 Feb 2020 (data stops 10:58), 17 Oct 2025 (stops 15:16).
- Exclude from overnight stats: 18 May 2022 overnight, 28 Nov 2025 overnight (CME outage), the partial 30 Apr 2026 session.
- Keep but flag: 13:15 early closes, the Mar 2020 circuit-breaker halt days.

### Your notebooks
| Notebook | Topic | Validation level |
|---|---|---|
| `Statmap_5` | projected levels from open: median wick ("manipulation"), distribution distance; hourly→daily; 1y/5y/full; ATR regimes | 1y/5y/full comparison; cell 0 uses whole-sample medians (in-sample), cell 1 uses 60-candle lookback |
| `hourly_range_mid__open_retracements_9` | overnight hourly opens/mids revisited in NY; prior-hour break → back to mid/open | descriptive |
| `candle_continuations_and_reversals_1` | C1/C2 → C3 on 15m–1D, by structure, session, weekday | descriptive |
| `comprehensive_range_analysis_MTF_4_1` | P(up/down/inside) by open position × time, first touch, train/test | train/test + shrinkage |
| `Fractal_model_stats_4` | HTF sweep + reclaim, CISD, funnel, MAE/MFE, GC/ES/NQ | descriptive + baseline control |
| `FVG_and_session_outcome_2` | first FVG 9:30–10 / 10–11 → RTH direction | descriptive (see §6.1) |
| `Initial_balance_6_1`, `NQ_IB_Orb_Analysis_1` | IB/ORB: formation order, close half, first break, held extreme | descriptive |
| `magic_hours_5_1` | hour range → break → back to 50%, MAE, runners | descriptive |
| `range_retracements_2` | RTH range 50% retracement by open position / distance / weekday | descriptive |
| `rangelab`, `rangelab_ONE_FILE` | range grid with within-day permutation null, FDR, block bootstrap, train/test, year agreement | **full significance testing** |
| `NQ_range_hit_rates`, `NQ_hit_rates_v2` | ATR-unit continuation/return, up vs down split, lean vs 52.2% base | descriptive, ATR-normalised |

Referenced in your plan but **not uploaded**: the `oos/` folder (`scorecard.md`, `holdout_results.csv`, `fullsample_cells.csv`, `hourly_by_fill.csv`, `scalps.py`, `orderflow.py`, `fibadr.py`, `ifvg*.py`, `rangefill.py`), `FINDINGS.md`.

---

## 2. Proven — survives your out-of-sample, year-by-year or significance checks

From `NQ_best_stats.md` (2010–2026, every stat ≥100 cases, consistent in 88–100% of years) and plan §3 (2020–Mar 2025 vs unseen Mar 2025–Apr 2026).

**Location → reachable levels (the core of a daily map)**
| Context | Outcome | Rate (base) |
|---|---|---|
| RTH opens upper / lower half of prior day | near prior-day extreme by 16:00 | 67% (41% / 34%) |
| RTH opens above PDH / below PDL | far prior-day extreme | 13% / 10% |
| 8am in top third of overnight range | ONL holds to 16:00 / ONH traded by 11:00 | 55% / 67–73% |
| 8am in bottom third | ONH holds / ONL traded by 11:00 | 53% / 61–68% |
| Gap > 0.5 ATR | fills by 11:00 | 13–15% |
| 9:00 hour opens lower / upper half of 8:00 hour | takes near 8:00 extreme | 83% / 84% |
| 10:00 hour opens lower / upper half of 9:00 hour | takes near 9:00 extreme | 73–78% (holds unseen) |
| 11:00 hour opens in one half of 10:00 hour | takes the far 10:00 extreme | 14–15% |
| 9:00-hour sweep of 8:00 high | closes back inside | 26% normally; 34% if day or week range > 75% filled; 51% if both |

**Range width → behaviour**
| Context | Outcome | Rate (base) |
|---|---|---|
| Narrow IB (< 0.38 ATR) | returns to IB mid after first break / extends ≥ 0.5× IB | 64% / 64% |
| Wide IB (> 0.75 ATR) | returns to IB mid / extends ≥ 0.5× IB | 36% / 42% |
| Narrow / wide ORB (9:30–9:45) | both sides taken by 11:00 | 40–41% / 21% |
| 60-min range narrow vs ATR | travels ≥ 1 width past the edge | 59% (40%) |
| Width ÷ ATR on its own | sorts return-to-mid behaviour | ~10× the luck threshold in rangelab (strongest conditioner found) |

**One order-flow rule:** don't rest a limit fade at a level when price arrives with strong aggressive delta behind it (−1.9 pts/trade old data, −2.6 unseen).

---

## 3. Agreement with nqstats.com

| nqstats module | Their headline | Your equivalent | Agree? |
|---|---|---|---|
| RTH Breaks | open above pRTH high → close above 69.9%; doesn't touch pRTH low 88.1% | open above PDH → reaches PDL 13% | yes |
| ALN Sessions | P3/P4 overnight lean → that London extreme breaks 75–81% | 8am third → which overnight extreme holds / is traded | yes, different framing |
| IB Breaks | close vs mid + extreme set last → 74–84% break that side | close-third first-break stats flagged as **geometry** (random walk ≈ same) | yes, but you showed it's mostly where price sits |
| Hour Stats | 09:00 hour return to its open 75.2% | 9:00 takes near 8:00 extreme 83%; magic hours | consistent |
| 1H Continuation | green 9am hour → green RTH 70.6% | direction stats near base rate | **conflict**: nqstats' figure includes 9:30–10:00 overlap (§6.2) |
| Noon Curve | Q2 breaks Q1 high → AM low/PM high 82.1% | not tested | untested; nqstats figure conditioned on outcome (§6.2) |
| AM TBR | ±0.25σ from 8am open → back to open ~74% | `Statmap` median-wick levels hit ~72%; `magic_hours` | similar concept, not directly compared |

---

## 4. Candidates — promising, not proven

1. **1m iFVG inside an aligned 15m FVG, near a key level, retest entry, stop beyond the 15m zone, 20-pt target.** +1.2 pts/trade old data (t ≈ 1.2–1.7, 316 trades), +5.3 unseen (72 trades). Only setup positive in both periods. Found by splitting finely, so expect shrinkage. Planned check: frozen rule on 2015–2019 from the 15-year file.
2. **First 9:30–10:00 FVG → RTH direction** (BISI 71%, both windows BISI 79%). Overlap-biased as measured (§6.1); needs re-measuring from 10:00/11:00.
3. **Statmap projected levels** (median manipulation wick hit ~72%, distribution ~20%). Cell 0 uses whole-sample medians; cell 1 (60-candle lookback) is the honest version.
4. **Range-fill context** (day/week > 75% filled) flips the 9:00 sweep from continuation toward reversal: n = 59–339.

## 5. Failed — tested and didn't hold

- **Direction**: RTH bull/bear, hourly bull/bear, C3 bull/bear, which edge breaks first by open position (rangelab best cell 0.72× luck threshold). Best directional cell: 56% vs 53%.
- **Mechanical entries after costs, both periods**: level touch fade/break at ONH/ONL, PDH/PDL, prior close, hour levels, ORB; fib 0.382–0.786 and ADR (0/144); IB breakout and second-side fade; 1h candle continuation; 5m/15m iFVG (earlier "edge" was a fill-order bug); 1m iFVG inside 15m FVG without the level filter; order flow at levels (real but ~+0.2 pts/trade).
- **15m ORB held to close**: positive but needs holding to 16:00, outside your scalping style.
- **Geometry illusions** (true but uninformative): IB/ORB close-third → first break, outside-bar → C3 first touch.

---

## 6. Method issues found in sources

1. **FVG notebook**: the 9:30–10:00 signal sits inside the 9:30-open → 16:15-close outcome. Part of the 71% is the FVG's own move.
2. **nqstats 1H Continuation**: the 9:00–10:00 signal overlaps the 9:30–16:00 outcome. **Noon Curve**: the Q2 → 82%/72% figures use only days already known to be "opposite-side".
3. **nqstats IB/RTH breaks** count wick touches only; no follow-through measured.
4. **AMT articles**: the 70–80% "return to value" and 80/20 balance figures have no source, sample or instrument.
5. **Zarattini papers**: stocks/ETFs, not NQ futures. Rules (VWAP side, 5-min ORB with relative volume, Noise Area, NR4, VIX/RSI) are testable on NQ but none is proven there.
6. **Statmap cell 0**: levels from whole-sample medians, so in-sample by construction.

---

## 7. Untested — where your original question points

None of the notebooks or the plan touch these. Your uploaded data has the volume and delta needed for all of them.

**Auction Market Theory**
- Prior-day value area / POC from volume profile; RTH open inside vs outside value.
- The "80% rule": open outside value, re-enter, then traverse to the far side.
- Value-area migration (higher/lower/overlapping) as day context.
- IB width vs day type (normal / trend / neutral) using AMT definitions.
- Single prints, poor highs/lows, naked POCs as targets.
- Acceptance vs rejection outside value (time/volume held beyond VAH/VAL).

**VWAP and the papers**
- Price vs session VWAP as day bias; VWAP side by time of day (papers: edge 9:30–12:00 and 15:00–16:00, none midday).
- Opening 5-min relative volume as a trend-day filter.
- Noise Area (avg move to that clock time, 14 days) as a "day is trending" flag.
- NR4 / NR7 / inside-day regime; VIX level; 5-day RSI (dealer-gamma proxy).

**Delta beyond the level-touch test**
- Opening-hour cumulative delta vs price (divergence) as day context.

**nqstats modules not in your work**
- Noon Curve (AM/PM extreme split), measured without the outcome conditioning.
- ALN pattern classification directly, to compare with your 8am-thirds result.

---

## 8. What a plan built from this would look like

The evidence supports a **context plan**, not a signal plan:
1. **Pre-9:30 map**: which levels are realistic by 11:00 (§2), which are low-odds, day regime (gap size, open vs prior day and value area, 8am third, ATR).
2. **9:30–10:00**: ORB/IB width class, which side is likely two-sided, VWAP side.
3. **10:00–11:00**: near 9:00-hour extreme as magnet, IB-width expectations.
4. **Entries**: your own discretion, with the one stand-aside rule (§2), until a candidate (§4) passes its next test.

AMT and VWAP (§7) would slot into steps 1–2 if they survive the same checks your other stats did.

## 9. Open decisions

- Upload the `oos/` folder and `FINDINGS.md`, so validated work isn't redone.
- Upload the 15-year file (zipped or split by year) if the 2015–2019 test, or a longer AMT test, is wanted.
- Which §7 items to test first. Suggested order: prior value area + 80% rule → VWAP side by time of day → IB-width day type → regime filters.
- Output: written plan, NinjaTrader indicator(s) plotting the map, or both.
