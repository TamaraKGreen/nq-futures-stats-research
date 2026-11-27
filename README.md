# NQ futures stats research

Research on intraday NQ (Nasdaq-100 futures) statistics and strategy ideas, tested **out of time**: every idea is
developed on one period and then checked on a later period it never saw. Includes 1-minute data for NQ, ES and gold
going back to 2010, the code for every study, and the results.

**New here? Start with [`DATA.md`](DATA.md)** — it explains every data file, which timezone it's in, and how to
load, combine and resample it. Then open [`examples/quickstart.ipynb`](examples/quickstart.ipynb).

## Quick start

```bash
git clone https://github.com/TamaraKGreen/nq-futures-stats-research.git
cd nq-futures-stats-research
pip install -r requirements.txt
python3 research/scripts/build_15y.py      # one-off, ~30 s: builds the combined 15-year dataset
```

(Windows: use `python` instead of `python3`. No git? Use **Code → Download ZIP** on GitHub.)

```python
from research.scripts.load import load              # NQ 2020-2026 with order flow
from research.scripts.build_15y import load_15y, resample

nq  = load("dev")                # NQ research dataset, development period
all3 = load_15y()                # NQ (2010-), ES (2011-), gold (2013-) 1-minute bars
h1  = resample(load_15y("es"), "1h")
```

Every loader returns timestamps **in New York time, timezone-aware**, whatever the source file used.

## The data at a glance

| Dataset | Symbols | Period | Bars | Extras | Timezone of the raw files |
|---|---|---|---|---|---|
| NQ research dataset | NQ | 2020-01 → 2026-11 | 1 minute | buy/sell volume, delta, trades, contract, roll adjustment, dev/OOT split | New York |
| 15-year dataset | NQ, ES, GC | 2010-06 / 2011-12 / 2013-07 → 2026 | 1 minute | OHLCV only | NQ New York; **ES and GC UTC** |

Details, columns, caveats and how each timezone was verified: [`DATA.md`](DATA.md).

## What's in the repo

| Path | Contents |
|---|---|
| [`DATA.md`](DATA.md) | **Data guide** — files, timezones, loading, combining, resampling, rolls, common mistakes |
| [`examples/`](examples/) | `quickstart.ipynb`: load, check timezones, combine symbols, resample, a first statistic |
| `research/data/` | NQ research dataset (dev/OOT parquet), original CSVs, NinjaTrader pulls, roll table — [README](research/data/README.md) |
| `research/data/raw_15y/` | 15-year NQ / ES / GC, one parquet file per year — [README](research/data/raw_15y/README.md) |
| `research/scripts/` | `load.py` (NQ research dataset), `build_15y.py` (combine / resample / ticks → bars / CSV export), dataset build scripts |
| `research/ranges/` | 15/30/60-minute time-based ranges: retracements, which side breaks first, continuation entries |
| `research/stats/` | Hourly stats, 8 AM range, value area / POC / gap opens, first-90-minute bias, Statmap, key levels |
| `research/fvg/` | FVG / iFVG as a directional bias |
| `research/strategies/` | ATR breakout and variants, Kuant Gapline ORB (overnight and day-trading versions), Statmap reversion |
| `research/output/` | Results (text) for every study above |
| `research/prior_work/` | Earlier notes, strategy source code, and 31 original Jupyter notebooks in `notebooks/` |
| `oos/` | Out-of-sample scorecard package run on this data; fresh-period checks in `oos/results/` |

Large intermediate files (`research/output/*.parquet`, `oos/cache/`, `research/data/futures_1min_15y.parquet`)
are not committed; the scripts regenerate them.

## Running the original notebooks

The notebooks in `research/prior_work/notebooks/` read files from the author's PC (e.g.
`C:\Users\...\nq_15y_1min.csv`). To run them on this repo's data, write the CSVs in their format and change the
path at the top of each notebook:

```bash
python3 research/scripts/build_15y.py --csv      # writes research/data/{nq,es,gc}_15y_1min_ET.csv
```

These CSVs are in **New York time for every symbol**. (The original ES and GC exports were UTC, so results from
those notebooks on ES/GC intraday windows will differ from the originals.)

## Requirements

Python 3.10+ with pandas, numpy and pyarrow (`pip install -r requirements.txt`). Jupyter and matplotlib for the
notebooks.

## Disclaimer

Research only, not trading advice. Past statistics don't guarantee future results; most of the effects studied here
are small, and several that looked strong in-sample did not hold up out of time (see `research/output/`). Market
data may be subject to the original data vendors' terms; check before redistributing it further.
