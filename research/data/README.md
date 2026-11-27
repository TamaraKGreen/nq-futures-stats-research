# NQ 1-minute dataset (2020-01-02 -> 2026-11-24)

Build (from repo root; the original NQ_ticks_YYYY_1min.csv files are in raw_csv/, gzipped):
`python3 research/scripts/build_base.py research/data/raw_csv research/data/nq_base_2020_2026.parquet && tar -xzf research/data/raw_nt_2026.tar.gz -C research/data && python3 research/scripts/build_dataset.py`

Load: `from research.scripts.load import load; df = load("dev")`  (or run with research/scripts on sys.path)

| file | contents |
|---|---|
| (build output) nq_1min_2020_2026.parquet | everything combined; not committed - use load("all")
| nq_1min_2020_2026_dev.parquet | sessions 2020-01-02 -> 2025-07-09 (1,427, 80%) |
| nq_1min_2020_2026_oot.parquet | sessions 2025-07-10 -> 2026-11-24 (357, 20%) - do not touch during research |
| roll_table.csv | 27 rolls: session the new contract starts, gap (new - old, pts), how detected |
| raw_csv/NQ_ticks_YYYY_1min.csv.gz | original 1-minute CSVs, 2020-01-01 -> 2026-04-29 (gzip; pandas reads them directly) |
| raw_nt_2026.tar.gz | raw NinjaTrader JSON pulls (NQM6/NQU6/NQZ6) |

Columns: `ts` (ET, tz-aware, bar open time) `open high low close volume buy_vol sell_vol delta trades`
`src` (csv|nt) `contract` `sess` (trading date, 18:00 ET anchor, Sunday evening -> Monday)
`sess_bars` `partial_sess` (<1200 bars: holidays, early closes, 2020-03 limit days, 2025-01-09, 2025-11-28 CME outage)
`roll_adj` (raw + roll_adj = back-adjusted price) `split` (dev|oot).

Prices are RAW front-month (what actually traded). Use raw for anything inside one session;
use raw + roll_adj for anything that compares across sessions spanning a roll (prior-day levels, multi-day ATR).

Caveats
- nt rows: buy_vol/sell_vol are up/down-tick volume, not aggressor-classified (delta corr 0.97 vs csv on overlap).
  OHLCV match the csv exactly on the 360-bar overlap.
- csv rows: ~0.8% of volume is unclassified (buy_vol + sell_vol < volume).
- 17:00-17:59 ET prints (27 roll-marker / pre-open bars) dropped.
- 2025 Jun/Sep/Dec rolls had no marker; roll date/gap inferred from the session-open gap.
