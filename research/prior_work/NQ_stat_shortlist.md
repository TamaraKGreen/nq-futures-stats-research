# NQ NY RTH — stat shortlist (nqstats + own stats)

*21 Nov 2026. Every stat here held vs its base rate on 2010–2022 **and** 2023 – May 2026, with rate ≥ 65% or ≤ 35% and ≥ 10 pp from base in **both** periods. Figures: **2023–26** (2010–22) · base. ★ = also beats the flip-only random walk (more than distance + time-of-day volatility). ⚠ = rate moved between periods; quote the 2023–26 figure.*
*Sources: `nqstats_val/results/verdicts.csv`, `nqstats_val/results/own_verdicts.csv`. Own stats were seen up to Mar 2025 during your original research, so for them this is a stability check. Statistics only, no entries.*

## Before 09:30 — the map

**Prior-day location (09:30 open vs prior RTH range)**
| Open | Outcome | Rate | Base |
|---|---|---|---|
| upper half | PDH by close | **67** (67) | 41 |
| lower half | PDH by 11:00 | **8** (9) | 30 |
| lower half | PDH by close | **25** (24) | 41 |
| above PDH | closes above PDH (nqstats) | **69** (68) | 35 |
| above PDH | PDL by 11:00 ★ | **3.5** (3.6) | 22 |
| above PDH | PDL by close ★ | **12** (13) | 33 |
| below PDL | never trades to PDH (nqstats) | **90** (90) | 44 |
| below PDL | PDH by 11:00 / by close | **2.4 / 10** (3.3 / 10) | 30 / 41 |
| inside | no follow-through break (0.1 ATR) of either side | **30** (31) | 19 |

**Gaps (open vs prior close, in ATR)**
| Gap | Outcome | Rate | Base |
|---|---|---|---|
| up ≥ 0.5 | fills by 11:00 ★ | **14** (14) | 32 |
| up ≥ 0.5 | fills by close | **32** (28) | 48 |
| up ≥ 0.5 and above PDH | fills by close | **29** (27) | 48 |
| down ≥ 0.5 | fills by 11:00 / by close | **13 / 31** (13 / 33) | 32 / 48 |

**8am position in the 18:00–08:00 range**
| 8am | Outcome | Rate | Base |
|---|---|---|---|
| top third | overnight (18:00–09:30) high traded 09:30–11:00 | **70** (66) | 51 |
| top third | overnight low traded 09:30–11:00 ⚠ | **24** (30) | 44 |
| top third | 18:00–08:00 high holds to 16:00 ⚠ | **8** (14) | 28 |
| bottom third | overnight high traded 09:30–11:00 | **29** (27) | 51 |
| bottom third | 18:00–08:00 low holds to 16:00 | **16** (12) | 37 |

**Daily candle (yesterday = C2, day before = C1)**
| C2 | Today | Rate | Base |
|---|---|---|---|
| took C1 high, closed above it | takes C2 high / C2 low | **81 / 22** (81 / 18) | 56 / 43 |
| took C1 low only, closed below it | takes C2 high | **26** (22) | 56 |
| outside bar, closed above C1 high (n ≈ 35) | takes C2 high / C2 low | **79 / 14** (82 / 8) | 56 / 43 |

## 09:00–10:00

| Stat | Rate | Base |
|---|---|---|
| 09:00 hour opens upper / lower half of 08:00 → takes that side's 08:00 extreme | **85 / 83** (83 / 83) | 69 / 65 |
| 09:00 hour breaks the 08:00 range in its first 20 min → returns to the 09:00 open (nqstats) | **90** (88) | 65 |
| 09:00 hour returns to its open after breaking the 08:00 range (nqstats) | **78** (77) | 65 |
| 09:00 break only after minute 40 → returns to the open | **28** (34) | 65 |
| 09:30–09:45 ORB wide (top third) → both ORB sides by 11:00 | **19** (17) | 31 |

## 10:00–11:00

**IB (09:30–10:30), nqstats**
| IB | Outcome | By noon | By close | Base (noon / close) |
|---|---|---|---|---|
| closes above mid | IB high broken | **70** (71) | **83** (82) | 47 / 62 |
| closes below mid | IB low broken | **67** (65) | **79** (75) | 40 / 56 |
| low formed first | IB high broken | **71** (67) | **83** (80) | 47 / 62 |
| high formed first | IB low broken | – | **74** (70) | 56 |
| above mid + low first | IB high broken | **76** (73) | **86** (83) | 47 / 62 |
| below mid + high first | IB low broken | **70** (67) | **80** (77) | 40 / 56 |
| wide IB (> 0.75 ATR) ⚠ | both sides broken | – | **7** (12) | 21 |

**Hour position (own)**
| Stat | Rate | Base |
|---|---|---|
| 10:00 opens upper half of 09:00 → takes 09:00 high ⚠ | **80** (73) | 54 |
| 10:00 opens lower half of 09:00 → takes 09:00 low | **74** (74) | 49 |
| 10:00 opens upper / lower half → takes the far 09:00 extreme | **27 / 24** (26 / 26) | 49 / 54 |
| 11:00 opens upper half of 10:00 → takes 10:00 high | **72** (68) | 46 |
| 11:00 opens in one half → takes the far 10:00 extreme | **12–16** (15) | 36–46 |
| 10:00 hour breaks the 09:00 range only after minute 40 → returns to the 10:00 open (nqstats) | **15** (16) | 65 |

**1-hour candles, C3 = 09:00 / 10:00 / 11:00 (pooled)**
| C2 (prior hour) | C3 | Rate | Base |
|---|---|---|---|
| took C1 high, closed above it | takes C2 high ⚠ / C2 low | **80 / 28** (76 / 28) | 56 / 50 |
| took C1 low, closed below it | takes C2 low / C2 high ⚠ | **73 / 25** (71 / 31) | 50 / 56 |
| took C1 low, closed inside | takes C2 high | **72** (69) | 56 |
| outside, closed above C1 high | takes C2 high / C2 low | **81 / 25** (80 / 21) | 56 / 50 |
| outside, closed below C1 low | takes C2 low / C2 high | **75 / 20** (75 / 21) | 50 / 56 |

**Range width 08:00–10:59 (own, rangelab; quintiles of width ÷ ATR)**
| Range | Narrowest fifth (Q1) | Widest fifth (Q5) |
|---|---|---|
| 15 min | back to mid **77** (76); travels ≥ 1 width **66** (67) | back to far edge **17** (17); ≥ 1 width **17** (18) |
| 30 min | back to mid **83** (84); far edge **73** (72); ≥ 1 width **75** (78) ⚠ | far edge **12** (14); ≥ 1 width **13** (16) ⚠ |
| 60 min | back to mid **85** (82); far edge **70** (67) | far edge **11** (13); ≥ 1 width **13** (15) |
Base ≈ 58–64 (mid), 33–42 (far edge), 34–43 (≥ 1 width). Q4 ranges also qualify for "far edge ≤ 27, ≥ 1 width ≤ 28". **None of these beat the random walk**: narrow ranges travel and revert in range-width terms because they're narrow. Use them to size expectations, not as signals.

## Volume / AMT additions (added 22 Nov 2026 — genuinely out of time: definitions never measured before 2023+ was opened)
*Source: `nqstats_val/results/vol_verdicts.csv`. Value areas 70%, VAH/VAL from 1-min profiles; POCs from tick profiles (development 2020–22 for POC rows). RTH and ETH profiles give near-identical results; RTH shown, ETH in brackets where different.*

**Prior-day value (09:30)**
| Stat | Rate 2023–26 (2010–22) | Base |
|---|---|---|
| open inside prior VA → touches VAH by close | **69** (69) [ETH 66] | 51 |
| open above VAH or below VAL → prior POC touched by 11:00 | **27–29** (28–32) | 42 |
| naked daily POC < 0.25 ATR from open → touched by 11:00 | **75** (78) | 52 |
| naked daily POC < 0.25 ATR from open → touched by close ⚠ | **82** (87) | 65 |
| naked daily POC 0.5–1 ATR from open → touched by 11:00 / close | **13 / 33** (15 / 35) | 52 / 65 |

**Prior-week value (09:30)**
| Stat | Rate | Base |
|---|---|---|
| yesterday's value above last week's value → RTH closes outside weekly VA | **88** (91) | 70 |
| yesterday's value below last week's value → RTH closes outside weekly VA | **83** (85) | 70 |
| open inside weekly VA → touches weekly VAL by close | **34** (34) | 16 |
| open below weekly VAL → prior-week POC touched by close | **10** (8) | 25 |
| naked weekly POC 0.5–1 ATR from open → touched by 11:00 | **10** (14) [ETH 6] | 28 |
| naked weekly POC < 0.25 ATR from open → touched by close (n ≈ 48) | **65** (78) [ETH 69] | 46 |

**VWAP and opening volume**
| Stat | Rate | Base |
|---|---|---|
| 10:30 within 1 SD of VWAP → touches VWAP by 12:00 | **86** (85) | 70 |
| opening 5-min relative volume in lowest third → RTH range ≥ 1 ATR | **27** (32) | 43 |

**Beats the random walk but below the 65/35 rule — worth knowing:**
- **VWAP side:** above VWAP at 10:00 / 10:30 / 11:00 → RTH closes above VWAP **62 / 67 / 67%** vs random walk 52–59%; below VWAP → closes below only **46–52%**. VWAP carries information, but it's lopsided: it confirms longs more than shorts (NQ's drift).
- **10:30 price 1–2 SD from VWAP → back to VWAP by 12:00 57%** vs 62% random walk: stretched price keeps going more than chance suggests.
- **Range volume (the one pre-stated volume prediction that held):** a 30-min range (08:00–10:59) built on **high** relative volume returns to its mid **60%** after breaking, vs **74%** for a low-volume range (2010–22: 61 vs 70). High-volume ranges follow through.

**Failed or no edge:** 80% rule (far edge 11–13% by noon, ~30% by close), "open outside value returns inside 70–80%" (actually ~50% by 11:00, ~63% by close), value migration, poor highs/lows, opening delta vs IB direction, IB stats by value context, and four of five volume predictions (pre-open volume made the overnight high *less* likely to trade; S1/S4/S9 splits showed no reliable difference). The 10:00-hour × volume rows in the verdict file are just S4 split three ways (no difference between them) and add nothing new.

## Not on the list (held, but probability too close to base for the 65/35 + 10 pp rule)
AM TBR (08:xx touch → back to 08:00 open 81% vs 76% base), Noon Q2 all-days (63.5% vs 41%), ALN P3/P4, narrow IB → back to mid (63%), 10:00 hour first-20-min break → back to open (73%, 68% in 2010–22), the two sweep effects that beat the random walk (12:00 sweeps 11:00 low only → closes inside 55% vs 44% random walk; 09:00 sweeps 08:00 low only 37% vs 30%).
The Noon Q2 "opposite-side days" version (82% / 77%) is excluded: you can only know a day was opposite-side after the close.

## More than geometry (★) — the effects that beat the random walk in both periods
1. Open above PDH → full retrace to PDL: 12% by close, 3.5% by 11:00, vs 8% / 2% for a random walk. **Confirmed in all three runs** (nqstats, own stats, volume/AMT).
2. Gap up ≥ 0.5 ATR → fills by 11:00: 14% vs 9%.
3. (Below the threshold) 12:00 sweeps only the 11:00 low → closes back inside: 55% vs 44%; 09:00 sweeps only the 08:00 low → 37% vs 30%.
All four are **mean reversion after an up-move or a downside sweep**.
