# 15-year 1-minute futures data (ES, GC; NQ to follow)

One parquet file per year: `<sym>_1min_<year>.parquet`, columns `Date, Time, Open, High, Low, Close, Volume`
(Date/Time are strings). Continuous front-month contract as exported from the original CSVs
(`ES_15y_1min.csv`, `GC_15y_1min.csv`), split by year with no other changes.

| symbol | files | rows | from -> to |
|---|---|---|---|
| ES | 16 | 5,127,103 | 2011-12-04 -> 2026-07-31 |
| GC | 14 | 4,280,231 | 2013-07-29 -> 2026-07-31 |

Checks on arrival: years join with no duplicate timestamps and in order; only one gap > 4 days per symbol
(ES 2011-12-04 -> 12-11, start of the file; GC 2018-09-07 -> 09-11).

**Timestamps look like UTC, not New York time.** The Sunday session open is at 23:00 in January and 22:00 in
July (18:00 ET either way), and ES volume peaks at 20:00 UTC in January / 19:00 in July (the 15:00 ET close).
Convert before using ET clock times such as 09:30:

```python
import glob, pandas as pd
df = pd.concat(map(pd.read_parquet, sorted(glob.glob("research/data/raw_15y/es_1min_*.parquet"))), ignore_index=True)
df["ts"] = pd.to_datetime(df.Date + " " + df.Time).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
```
