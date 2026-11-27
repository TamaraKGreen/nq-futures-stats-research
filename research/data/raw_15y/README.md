# 15-year 1-minute futures data (NQ, ES, GC)

One parquet file per year: `<sym>_1min_<year>.parquet`, columns `Date, Time, Open, High, Low, Close, Volume`
(Date/Time are strings), split by year from the original exports (`nq_15y_1min.csv`, `ES_15y_1min.csv`,
`GC_15y_1min.csv`) with no other changes.

| symbol | files | rows | from -> to | timestamps |
|---|---|---|---|---|
| NQ | 16 | 4,974,698 | 2010-06-06 -> 2025-03-14 | New York local time |
| ES | 16 | 5,127,103 | 2011-12-04 -> 2026-07-31 | **UTC** |
| GC | 14 | 4,280,231 | 2013-07-29 -> 2026-07-31 | **UTC** |

How the timezones were established:
- NQ: compared bar by bar with our 2020-2025 NQ dataset (New York time). At the same timestamp 96.9% of closes
  are identical and volume correlates 0.996; shifting by any whole number of hours destroys the match. The other
  ~3% are around contract rolls. NQ looks like unadjusted front-month (it does not match our back-adjusted series).
- ES, GC: the Sunday open is 23:00 in January and 22:00 in July (18:00 ET either way). After converting UTC -> ET,
  the busiest minutes are 09:30 and 15:59-16:00 for ES, 08:30 and 13:29 (the COMEX close) for GC.

So the original notebooks that read `nq_15y_1min.csv` with ET clock times were right for NQ; the same code on the
ES or GC files would need a UTC -> ET conversion first.

**Known error (corrected by `build_15y.py`, not in these files):** NQ 2012-02-05 18:00 -> 2012-02-10 16:14,
6,231 bars with every price x100. Use the loader, or divide those bars by 100 yourself.

Checks on arrival: years join with no duplicate timestamps and in order. One gap > 4 days per symbol:
NQ 2014-06-11 -> 06-15, ES 2011-12-04 -> 12-11 (first week of the file), GC 2018-09-07 -> 09-11.

## One combined dataset

```bash
python3 research/scripts/build_15y.py
```

writes `research/data/futures_1min_15y.parquet` (~190 MB, ~15M rows, not committed - over GitHub's file limit;
takes ~30 s): columns `symbol, ts (America/New_York), open, high, low, close, volume, sess (18:00 ET anchor), src`.
NQ is extended past 2025-03-14 to 2026-11-24 from our research dataset (`src = nt`); `--no-extend-nq` turns that off.

```python
from research.scripts.build_15y import load_15y, resample, ticks_to_1min
df  = load_15y()                  # all symbols; load_15y("nq") or load_15y(["nq", "es"]) for some
h1  = resample(df, "1h")          # any bar size, per symbol, never across the 18:00 session boundary
bars = ticks_to_1min(ticks)       # tick data (timestamp, price, volume[, aggressor]) -> 1-minute OHLCV + delta
```
