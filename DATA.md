# Data guide

Everything you need to know to use the data in this repo: what each file is, **what timezone its timestamps are
in**, how prices are adjusted, and how to load, combine and resample it.

> **Rule of thumb:** use the loaders below rather than reading the raw files yourself. They return every symbol
> with timezone-aware timestamps in New York time (`America/New_York`), so you never have to think about UTC vs ET.

- [1. Which file should I use?](#1-which-file-should-i-use)
- [2. Every data file, its timezone and format](#2-every-data-file-its-timezone-and-format)
- [3. Conventions used everywhere](#3-conventions-used-everywhere)
- [4. Loading the data](#4-loading-the-data)
- [5. Combining, converting and resampling](#5-combining-converting-and-resampling)
- [6. Contract rolls and price adjustment](#6-contract-rolls-and-price-adjustment)
- [7. The dev / out-of-time split](#7-the-dev--out-of-time-split)
- [8. Common mistakes](#8-common-mistakes)
- [9. How the timezones were verified](#9-how-the-timezones-were-verified)
- [10. Rebuilding the datasets from the raw files](#10-rebuilding-the-datasets-from-the-raw-files)

---

## 1. Which file should I use?

| You want... | Use | Gives you |
|---|---|---|
| NQ 2020-2026 with order flow (buy/sell volume, delta), contract names, roll adjustment, dev/OOT split | `load()` in `research/scripts/load.py` | `research/data/nq_1min_2020_2026_{dev,oot}.parquet` |
| Long history (NQ from 2010, ES from 2011, gold from 2013), OHLCV only | `load_15y()` in `research/scripts/build_15y.py` | `research/data/futures_1min_15y.parquet` (built from `raw_15y/`) |
| Bars other than 1 minute (5m, 15m, 1h, ...) | `resample()` in `build_15y.py` | from either dataset |
| To run the original notebooks in `research/prior_work/notebooks/` | `python research/scripts/build_15y.py --csv` | `research/data/<sym>_15y_1min_ET.csv` in their `Date,Time,Open,...` format |

Both datasets are **1-minute bars**. Neither contains tick data. `ticks_to_1min()` will turn tick data into
1-minute bars if you have it.

---

## 2. Every data file, its timezone and format

**Read this table before opening any raw file directly.** The 15-year ES and GC files are in **UTC**; everything
else is New York time.

| Path | What | Timezone of timestamps | Coverage | Committed? |
|---|---|---|---|---|
| `research/data/nq_1min_2020_2026_dev.parquet` | NQ research dataset, first 80% of sessions | `America/New_York`, **tz-aware** column `ts` | 2020-01-02 → 2025-07-09 | yes |
| `research/data/nq_1min_2020_2026_oot.parquet` | NQ research dataset, last 20% (out-of-time) | `America/New_York`, tz-aware `ts` | 2025-07-10 → 2026-11-24 | yes |
| `research/data/raw_csv/NQ_ticks_YYYY_1min.csv.gz` | Original NQ 1-minute CSVs with order flow (despite the name, these are 1-minute bars, not ticks) | New York time **with offset** in the text, e.g. `2024-01-01 18:00:00-05:00` | 2020-01-01 → 2026-04-29 | yes (gzip) |
| `research/data/raw_nt_2026.tar.gz` | Raw NinjaTrader JSON pulls used to extend NQ to Nov 2026 | **UTC**, ISO text e.g. `2026-09-20T22:00:00Z`; converted to New York by `build_dataset.py` | 2026 | yes |
| `research/data/roll_table.csv` | NQ contract roll dates and price gaps | dates are 18:00-anchored trading days | 2020-2026 | yes |
| `research/data/raw_15y/nq_1min_YYYY.parquet` | 15-year NQ export, one file per year (one week in Feb 2012 is x100 - corrected by the loader, see §10) | **New York local time**, naive `Date` + `Time` strings | 2010-06-06 → 2025-03-14 | yes |
| `research/data/raw_15y/es_1min_YYYY.parquet` | 15-year ES export | **UTC**, naive `Date` + `Time` strings | 2011-12-04 → 2026-07-31 | yes |
| `research/data/raw_15y/gc_1min_YYYY.parquet` | 15-year gold (GC) export | **UTC**, naive `Date` + `Time` strings | 2013-07-29 → 2026-07-31 | yes |
| `research/data/futures_1min_15y.parquet` | NQ + ES + GC combined, built by `build_15y.py` | `America/New_York`, tz-aware `ts` | NQ 2010 → 2026-11-24, ES/GC → 2026-07-31 | **no** (187 MB; rebuild in ~30 s) |
| `research/data/<sym>_15y_1min_ET.csv` | Optional notebook-format export, built by `build_15y.py --csv` | New York local time, naive `Date` + `Time` | as above | no |

"Naive" means the timestamp has no timezone attached, so nothing in the file tells you which zone it is in. That
is why this table exists.

### Columns

**NQ research dataset** (`load()`):

| column | meaning |
|---|---|
| `ts` | bar **open** time, tz-aware New York |
| `open high low close` | raw front-month prices (what actually traded) |
| `volume trades` | contracts traded, number of trades |
| `buy_vol sell_vol delta` | aggressor buy / sell volume and their difference (see caveat for `src = nt`) |
| `src` | `csv` (original CSVs) or `nt` (NinjaTrader, 2026-04-30 on) |
| `contract` | e.g. `NQH4` |
| `sess` | trading date, 18:00 ET anchor (see §3) |
| `sess_bars`, `partial_sess` | bars in the session; `True` for holidays, early closes, outages (< 1,200 bars) |
| `roll_adj` | add to OHLC for a back-adjusted series (see §6) |
| `split` | `dev` or `oot` |

**15-year dataset** (`load_15y()`):

| column | meaning |
|---|---|
| `symbol` | `nq`, `es` or `gc` |
| `ts` | bar **open** time, tz-aware New York |
| `open high low close volume` | as exported |
| `sess` | trading date, 18:00 ET anchor |
| `src` | `15y` (the export) or `nt` (NQ bars after 2025-03-14, taken from the research dataset) |

**Raw 15-year files** (`raw_15y/*.parquet`): `Date` (`YYYY-MM-DD`), `Time` (`HH:MM:SS`), `Open High Low Close
Volume`. Timezone per the table above.

---

## 3. Conventions used everywhere

- **Timestamps are bar open times.** The bar stamped `09:30` covers 09:30:00-09:30:59. So the regular-hours
  session (09:30-16:00 ET) is the bars stamped `09:30` to `15:59`.
- **Trading day (`sess`) starts at 18:00 New York time.** Bars from Sunday 18:00 to Monday 17:59 belong to
  Monday's session. This matches the CME futures day. `sess = (local wall time + 6 hours).normalize()`.
- **Daylight saving is handled by the timezone.** Because `ts` is tz-aware, 09:30 always means 09:30 in New York,
  summer or winter. Never hard-code a UTC offset like `-5h`.
- **Missing minutes are missing rows.** If nothing traded in a minute, there is no bar. Don't assume 1,380 bars
  per session.
- **Session hours changed over time.** Before 2015 the NQ session closed at 17:15 ET, so older data has bars
  between 17:00 and 17:15. From 2015 the close is 17:00 (last bar 16:59).
- **Prices are in index points.** NQ: 1 point = $20 (MNQ $2), tick 0.25. ES: 1 point = $50 (MES $5), tick 0.25.
  GC: 1 point = $100 (MGC $10), tick 0.10.

---

## 4. Loading the data

### Setup

```bash
git clone https://github.com/TamaraKGreen/nq-futures-stats-research.git
cd nq-futures-stats-research
pip install -r requirements.txt
```

On Windows use `python` instead of `python3`. You can also download the repo as a ZIP from the green **Code**
button on GitHub. Run everything **from the repo's top folder**, so that `from research.scripts...` imports work.
In a notebook elsewhere, add the repo folder to the path first:

```python
import sys; sys.path.insert(0, r"C:\path\to\nq-futures-stats-research")
```

### NQ research dataset (2020-2026, with order flow)

```python
from research.scripts.load import load

df = load("dev")                     # 2020-01-02 -> 2025-07-09 (use this for research)
df = load("all")                     # dev + out-of-time
df = load("all", drop_partial=True)  # without holiday / early-close / outage sessions
df = load("all", adjusted=True)      # back-adjusted prices (see §6)

df.ts.iloc[0]        # Timestamp('2020-01-01 18:00:00-0500', tz='America/New_York') - aware, New York
                     # (first bar of the 2020-01-02 session, which opens the evening before)
```

### 15-year dataset (NQ, ES, GC)

Build it once (about 30 seconds); `load_15y()` also builds it automatically if it's missing:

```bash
python3 research/scripts/build_15y.py
```

```python
from research.scripts.build_15y import load_15y

all3 = load_15y()                 # every symbol
nq   = load_15y("nq")             # one symbol
idx  = load_15y(["nq", "es"])     # several
```

### Regular trading hours only

```python
t = df.ts.dt.hour * 60 + df.ts.dt.minute          # minutes since midnight, New York time
rth = df[(t >= 9 * 60 + 30) & (t < 16 * 60)]      # bars 09:30 .. 15:59
```

### Daily bars

```python
daily = df.groupby("sess").agg(open=("open", "first"), high=("high", "max"),
                               low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
```

For more than one symbol, group by `["symbol", "sess"]`.

---

## 5. Combining, converting and resampling

### Combine the 15-year yearly files into one dataset

`build_15y.py` does this: it reads every `raw_15y/<sym>_1min_*.parquet`, converts each symbol to New York time
(UTC → New York for ES and GC; NQ is already New York time), corrects the known NQ price error (§10), removes
duplicates, checks OHLC consistency and price scale, adds
`sess`, and writes `futures_1min_15y.parquet`.

```bash
python3 research/scripts/build_15y.py                       # all symbols, NQ extended to 2026-11-24
python3 research/scripts/build_15y.py --symbols nq es       # some symbols
python3 research/scripts/build_15y.py --no-extend-nq        # NQ stops where the export does (2025-03-14)
python3 research/scripts/build_15y.py --csv                 # also write notebook-format CSVs (New York time)
```

### Read one raw file yourself (if you must)

```python
import pandas as pd

es = pd.read_parquet("research/data/raw_15y/es_1min_2024.parquet")
naive = pd.to_datetime(es.Date + " " + es.Time)

# ES and GC: the raw times are UTC
es["ts"] = naive.dt.tz_localize("UTC").dt.tz_convert("America/New_York")

# NQ: the raw times are already New York local time
nq = pd.read_parquet("research/data/raw_15y/nq_1min_2024.parquet")
nq["ts"] = pd.to_datetime(nq.Date + " " + nq.Time).dt.tz_localize(
    "America/New_York", ambiguous=True, nonexistent="shift_forward")
```

`ambiguous=True` resolves the repeated 01:00-01:59 hour on the November clock change (the file holds it once);
`nonexistent="shift_forward"` handles the skipped hour in March. `build_15y.py` does the same.

### Put two symbols side by side (e.g. NQ vs ES)

```python
from research.scripts.build_15y import load_15y

d = load_15y(["nq", "es"])
wide = d.pivot_table(index="ts", columns="symbol", values="close", observed=True)  # one column per symbol
wide = wide.dropna()                                                  # only minutes where both traded
```

### Resample to other bar sizes

```python
from research.scripts.build_15y import resample

m15 = resample(nq, "15min")
h1  = resample(load_15y(), "1h")      # works on several symbols at once
```

Bars are clock-aligned (`09:30`, `09:45`, ...; hourly bars on the hour), labelled by their open time, and never
straddle the 18:00 session start. `resample()` needs `symbol` and `sess` columns; for the NQ research dataset add
`df["symbol"] = "nq"` first.

Hourly bars here start on the hour, so the 09:00 bar holds 09:00-09:59 (half pre-market, half RTH). If you want
RTH-aligned hours (09:30-10:29, ...), shift before resampling:

```python
x = nq.assign(ts=nq.ts - pd.Timedelta(minutes=30))
h1_rth = resample(x, "1h").assign(ts=lambda r: r.ts + pd.Timedelta(minutes=30))
```

### Tick data → 1-minute bars

```python
from research.scripts.build_15y import ticks_to_1min

bars = ticks_to_1min(ticks)   # ticks: columns timestamp, price, volume, optional aggressor ('Buy'/'Sell')
```

Timestamps are read as UTC instants (tz-aware input is respected) and returned in New York time, with `trades`,
and `buy_vol / sell_vol / delta` when an aggressor column is present. If your tick timestamps are naive local New
York time, localize them first: `ticks["timestamp"] = ticks.timestamp.dt.tz_localize("America/New_York")`.

### Convert anything to New York time

```python
s.dt.tz_localize("UTC").dt.tz_convert("America/New_York")    # naive UTC -> New York
s.dt.tz_convert("America/New_York")                          # aware (any zone) -> New York
s.dt.tz_localize("America/New_York", ambiguous=True,
                 nonexistent="shift_forward")                # naive New York wall time -> aware
s.dt.tz_localize(None)                                       # aware -> naive local wall time (for display/CSV)
```

---

## 6. Contract rolls and price adjustment

Futures expire, so a continuous series jumps when it switches from one contract to the next. NQ and ES roll
quarterly (March, June, September, December); gold rolls every other month.

- **NQ research dataset:** prices are raw front-month. `roll_table.csv` lists all 27 rolls in 2020-2026 with the
  price gap. `roll_adj` turns the series into a **back-adjusted** one (`load(..., adjusted=True)` does it for
  you): history is shifted so there are no jumps at rolls, and the latest contract keeps its real prices.
- **15-year NQ:** raw front-month (it matches the research dataset's raw prices, not the adjusted ones). No roll
  table; roll gaps are in the data.
- **15-year ES and GC:** adjustment method not verified. Large session-open gaps cluster in the ES roll months, so
  ES looks unadjusted too.

**When it matters:** anything inside one session (opening range, intraday levels, VWAP) is fine on raw prices.
Anything that compares prices **across sessions** (prior-day high/low, multi-day ATR, weekly levels, moving
averages over days) will be distorted on the day of a roll. Use back-adjusted prices for NQ 2020-2026, or skip roll
days.

---

## 7. The dev / out-of-time split

The NQ research dataset is split chronologically: **dev** = sessions to 2025-07-09 (80%), **OOT** = 2025-07-10 on
(20%). Develop and tune on dev only; look at OOT once, at the end, to check whether a result holds up. Looking at OOT
repeatedly while developing turns it into another dev set.

The 15-year data isn't split. If you use it for research, pick your own cut-off date before you start, and don't
let OOT dates leak into it if you're also using the NQ research split (e.g. stop the 15-year sample at
2025-07-09).

---

## 8. Common mistakes

1. **Reading the raw 15-year files directly.** You'd get UTC times for ES/GC and the uncorrected Feb 2012 NQ week.
   Use `load_15y()`.
2. **Treating the ES/GC raw files as New York time.** They're UTC. Using `09:30` on raw ES gives you 04:30 or
   05:30 New York time. Use `load_15y()`, or convert as in §5.
3. **Using calendar dates as trading days.** Sunday 18:00 ET belongs to Monday's session. Use `sess`.
4. **Hard-coding a UTC offset.** New York is UTC-5 in winter and UTC-4 in summer. Use timezone conversion.
5. **Comparing prices across a roll on raw data.** See §6.
6. **Expecting a bar every minute.** Quiet minutes and halts have no rows.
7. **Mixing bar-open and bar-close labels.** Here the stamp is the open. Some platforms (e.g. TradingView exports,
   some NinjaTrader settings) stamp the close; shift by one bar when comparing.
8. **Peeking at the out-of-time data while developing.** See §7.

---

## 9. How the timezones were verified

They were checked against the data, not taken from file names:

- **NQ 15-year export → New York time.** Compared minute by minute with the NQ research dataset (which is New York
  time) over 2020-2025: at the same timestamp, 96.9% of closes are identical and volume correlates at 0.996.
  Shifting by any whole number of hours destroys the match (≤ 0.5% identical). The other ~3% are around contract
  rolls, where the two sources switch contracts on different days.
- **ES and GC 15-year exports → UTC.** The Sunday session opens at 23:00 in January and 22:00 in July, which is
  18:00 New York time in both seasons. After converting UTC → New York, the busiest minutes are 09:30 and
  15:59-16:00 for ES, and 08:30 and 13:29 (the COMEX close) for gold, as they should be.
- **NQ research dataset → New York time.** The original CSVs carry the offset in the text (`-05:00`/`-04:00`).
  The NinjaTrader rows overlap the CSVs for 360 bars and their OHLCV match exactly.

---

## 10. Rebuilding the datasets from the raw files

**15-year combined dataset:** `python3 research/scripts/build_15y.py` (see §5).

**NQ research dataset** (only needed if you change how it's built; the result is already committed):

```bash
python3 research/scripts/build_base.py research/data/raw_csv research/data/nq_base_2020_2026.parquet
tar -xzf research/data/raw_nt_2026.tar.gz -C research/data
python3 research/scripts/build_dataset.py
```

`research/data/README.md` has the details: roll detection, how order flow was filled in for the NinjaTrader rows,
and dropped bars.

### Caveats

- `src = nt` rows (NQ from 2026-04-30): `buy_vol`/`sell_vol` are up-tick/down-tick volume, not aggressor-classified
  (delta correlation 0.97 with the CSV values on the overlap).
- `src = csv` rows: about 0.8% of volume is unclassified (`buy_vol + sell_vol < volume`).
- **Corrected error in the raw NQ export:** 2012-02-05 18:00 → 2012-02-10 16:14 (6,231 bars, one week) has every
  price multiplied by 100 (e.g. `251975.00` for `2519.75`). `build_15y.py` divides those bars by 100 (they then sit
  exactly on the 0.25 tick and join the bars either side); the raw file in `raw_15y/` is left as delivered. Anything
  computed straight from the raw NQ file or the original `nq_15y_1min.csv` includes that week uncorrected: it barely
  moves intraday stats, but any rolling volatility or range measure that spans it is badly inflated (a 10-year
  baseline of daily returns stays roughly 10x too high until Feb 2022). `build_15y.py` also checks every build for
  price-scale errors like this (bars more than 50% away from the median of the surrounding 21 sessions) and warns
  if it finds any.
- 15-year data has one gap longer than four days per symbol: NQ 2014-06-11 → 06-15, ES 2011-12-04 → 12-11 (start of
  the file), GC 2018-09-07 → 09-11. GC also has noticeably fewer bars in 2018-2019.
