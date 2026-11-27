# NQ futures stats research

Out-of-time testing of intraday NQ statistics and strategy ideas on 1-minute data (Jan 2020 – Nov 2026).

- `research/data/` — combined 1-minute dataset, split chronologically 80/20: dev (to 2025-07-09) and OOT (2025-07-10 on). See its README.
- `research/data/raw_15y/` — 15-year 1-minute NQ / ES / GC split by year; `research/scripts/build_15y.py` combines them (see its README).
- `research/scripts/load.py` — `load(split="dev"|"oot"|"all", adjusted=False, drop_partial=False)`.
- `research/ranges/` — 15/30/60m time-based ranges: retracements, break direction, continuation entries.
- `research/stats/` — hourly stats, 8 AM range, value area / POC / gap opens, first-90-minute bias, Statmap, key levels.
- `research/fvg/` — FVG / iFVG as a directional bias.
- `research/strategies/` — ATR breakout and variants, Kuant Gapline ORB (overnight and day versions), Statmap reversion.
- `research/output/` — results text for every study.
- `research/prior_work/` — earlier notes and source files; `notebooks/` holds the original Jupyter notebooks.
- `oos/` — the out-of-sample scorecard package, run on this dataset, plus fresh-period checks in `oos/results/`.

Intermediate parquet files (`research/output/*.parquet`, `oos/cache/`) are not committed; the scripts regenerate them.
Requires Python 3 with pandas, numpy and pyarrow.
