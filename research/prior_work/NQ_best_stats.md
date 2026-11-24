# NQ: best-performing statistics (full sample, no OOS filter)

*NQ 1-min, Jun 2010 – Apr 2026. "Base" = the rate across all occurrences. "Years" = share of years where the stat leaned the same way as the full sample. Each stat has ≥ 100 occurrences. Full list: `oos/results/fullsample_cells.csv`.*

## Day structure (session)

| # | When… | …this happens | Rate | Base | n | Years |
|---|---|---|---|---|---|---|
| 1 | RTH opens in lower half of prior day | Trades down to PDL (by 16:00) | **67%** | 34% | 1,028 | 100% |
| 2 | RTH opens in upper half of prior day | Trades up to PDH | **67%** | 41% | 1,336 | 100% |
| 3 | RTH opens below PDL | Reaches PDH | **10%** | 41% | 569 | 100% |
| 4 | RTH opens above PDH | Reaches PDL | **13%** | 34% | 1,014 | 100% |
| 5 | Gap down > 0.5 ATR | Reaches PDL | **62%** | 34% | 111 | 100% |
| 6 | Gap up > 0.5 ATR, open above PDH | Gap fills | **27%** | 51% | 406 | 100% |
| 7 | 8am in bottom third of overnight range | Overnight HIGH holds to 16:00 | **53%** | 29% | 1,046 | 100% |
| 8 | 8am in bottom third | Overnight LOW holds to 16:00 | **13%** | 37% | 1,046 | 100% |
| 9 | 8am in top third (pre-8 bullish) | Overnight LOW holds to 16:00 | **55%** | 37% | 1,615 | 100% |
| 10 | 8am in top third (pre-8 bullish) | Overnight HIGH holds to 16:00 | **12%** | 29% | 1,615 | 100% |
| 11 | Narrow IB (< 0.38 ATR), IB bearish | Both IB sides taken | **43%** | 23% | 346 | 100% |
| 12 | Narrow IB | Returns to IB mid after first break | **64%** | 50% | 783 | 100% |
| 13 | Narrow IB | First break extends ≥ 0.5× IB | **64%** | 54% | 783 | 88% |
| 14 | Wide IB (> 0.75 ATR) | Returns to IB mid after first break | **36%** | 50% | 723 | 94% |
| 15 | Wide IB | First break extends ≥ 0.5× IB | **42%** | 54% | 723 | 100% |
| 16 | Narrow ORB | Both ORB sides taken | **65%** | 53% | 788 | 100% |
| 17 | Wide ORB | Both ORB sides taken | **41%** | 53% | 797 | 94% |

## Hourly (clock hours, ET)

| # | When… | …this happens | Rate | Base | n | Years |
|---|---|---|---|---|---|---|
| 18 | 9:00 hour opens in lower half of 8:00 hour | Takes 8:00 low | **83%** | 46% | 1,987 | 100% |
| 19 | 9:00 hour opens in upper half of 8:00 hour | Takes 8:00 high | **84%** | 53% | 2,043 | 100% |
| 20 | 9:00 hour (either half) | Trades through 8:00 mid | **85%** | 68% | 4,030 | 100% |
| 21 | 9:00 hour sweeps only the 8:00 high | Closes back inside | **26%** | 43% | 1,205 | 100% |
| 22 | 11:00 hour opens in lower half of 10:00 hour | Takes 10:00 high | **15%** | 53% | 1,860 | 100% |
| 23 | 11:00 hour opens in upper half | Takes 10:00 low | **14%** | 46% | 2,177 | 100% |
| 24 | 12:00 hour sweeps only the 11:00 low | Closes back inside | **54%** | 43% | 1,382 | 100% |
| 25 | 11:00 hour opens in upper half | Trades through 10:00 mid | **53%** | 68% | 2,177 | 100% |

### Hourly refined by daily / weekly range fill

Range fill, measured the way the "NQ Stats Price Distributions" indicator does it:
- From the day's 18:00 open (or the week's Sunday 18:00 open), take the excursion so far, up and down separately.
- Divide it by volatility (trailing σ of open→close: 252 days for the day, 52 weeks for the week).
- Rank it against all full days or weeks. "Up fill 80%" means the day has already gone further up than 80% of days go in total.
- Measured at the start of each hour. Code: `oos/rangefill.py`; every combination: `oos/results/hourly_by_fill.csv`.

**What changes (9:00 hour only)**

| Stat | Normal | When that side's range is > 75% filled | n |
|---|---|---|---|
| 9:00 sweep of the 8:00 high/low closes back inside | 26% | **34%** (day *or* week > 75%) | 339 |
| | | **51%** (day *and* week > 75%) | 59 |
| 9:00 hour opening in the lower half reaches the far 8:00 **high** | 58% | **44%** (week up-fill > 75%) | 216 |
| 9:00 hour opening in the upper half reaches the far 8:00 **low** | 55% | **46%** (week down-fill > 75%) | 234 |
| 9:00 hour opening in the lower half trades through 8:00 mid | 85% | **90%** (day up-fill > 75%) | 126 |

Exhausted-side 9:00 sweeps closed back inside 33–44% of the time in every five-year block (2010–14, 2015–19, 2020–24, 2025–26).

**What doesn't change**
- The near-side magnets (#18, #19: 9:00 takes the near 8:00 extreme, about 83%; 10:00 takes the near 9:00 extreme, about 73–75%) stay within ±3 pts in every fill bucket. They work whatever the range fill.
- 10:00 and 11:00 stats move only ±5 pts across fill buckets, within noise for these sample sizes.

**Reading it**
- **A 9am sweep normally keeps going.** But if the day or week has already stretched that way past its 75th percentile, a reversal back inside is about one-in-three. If both have, it's a coin flip.
- **Don't expect the far side of the 8:00 hour** when the week is already exhausted in that direction.

## Candles (C1, C2 → what C3 does)

| # | Setup | C3 does… | Rate | Base | n | Years |
|---|---|---|---|---|---|---|
| 26 | Daily outside bar closing below C1 low | Takes C2 high | **9%** | 56% | 146 | 100% |
| 27 | Daily outside bar closing below C1 low | Takes C2 low | **75%** | 43% | 137 | 100% |
| 28 | Daily outside bar closing above C1 high | Takes C2 low | **9%** | 43% | 153 | 100% |
| 29 | Daily C2 took C1 low and closed below it | Takes C2 high | **24%** | 56% | 644 | 100% |
| 30 | 4h outside bar closing above C1 high | Takes C2 high | **81%** | 51% | 1,083 | 100% |

## Range width (rangelab, 15/30/60-min ranges, within 3× duration)

| # | When… | …this happens | Rate | Base | n | Years |
|---|---|---|---|---|---|---|
| 31 | 60-min range, narrow vs ATR | Travels ≥ 1 width past the broken edge | **59%** | 40% | 10,147 | 100% |
| 32 | 60-min range, narrow | Returns to the far edge after breaking | **54%** | 35% | 10,147 | 100% |
| 33 | 60-min range, narrow | Returns to mid | **74%** | 61% | 10,147 | 100% |
| 34 | 15-min range, wide | Travels ≥ 1 width past the edge | **23%** | 40% | 64,467 | 100% |
| 35 | 15-min range, wide | Returns to the far edge | **20%** | 35% | 64,467 | 100% |

## Big numbers that are mostly geometry

These look huge but are mainly where price sits inside the range. A random walk gives almost the same figure:

| Setup | Rate | Random walk |
|---|---|---|
| IB closed in its low third → first break is UP | 15% | 16% |
| ORB closed in its high third → first break is UP | 85% | 85% |
| 1h outside bar closing above C1 high → C3 takes C2 high first | 89% | 87% |
| Daily outside bar closing below C1 low → C3 takes C2 high first | 7% | 15% |

They're true, but they only tell you price tends to hit the closer level first.

## Weakest (direction stats)

Best directional cells were tiny: 11:00 hour closes bullish after a bullish 10:00 hour **56% vs 53%**. RTH, hourly and candle bull/bear barely move off their base rates.
