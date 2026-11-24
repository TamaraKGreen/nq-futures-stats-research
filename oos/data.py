"""
ADAPTER: same interface as the original oos/data.py (kept unchanged in
research/prior_work/oos/data.py), but bars come from the research dataset
(research/data, Jan 2020 -> Nov 2026, unadjusted front month).

Periods (unchanged): discovery <= 2025-03-14, holdout 2025-03-17 .. 2026-04-29.
NEW: fresh = 2026-04-30 onwards. No earlier work has seen it. period_of() still
labels it 'holdout' so the original code runs unchanged; use fresh_of() to split it out.
NOTE: no 2010-2019 bars here, so discovery covers 2020-01 .. 2025-03 only.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research" / "data"          # only so `from .data import DATA` keeps working
CACHE = Path(__file__).resolve().parent / "cache"
NY = "America/New_York"
DISCOVERY_END = pd.Timestamp("2025-03-14")
HOLDOUT_START = pd.Timestamp("2025-03-17")
HOLDOUT_END = pd.Timestamp("2026-04-29")
FRESH_START = pd.Timestamp("2026-04-30")
INNER_SPLIT = "2022-06-01"      # original used 2019-06-01; with 2020+ data only that leaves no train set


def load_nq(refresh=False) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "nq_joined.parquet"
    if f.exists() and not refresh:
        return pd.read_parquet(f)
    sys.path.insert(0, str(ROOT))
    from research.scripts.load import load
    d = load("all")
    out = pd.DataFrame({"Datetime": d["ts"], "Open": d["open"].astype("float64"), "High": d["high"].astype("float64"),
                        "Low": d["low"].astype("float64"), "Close": d["close"].astype("float64"),
                        "Volume": d["volume"].astype("float64"),
                        "BuyVol": d["buy_vol"].astype("float64"), "SellVol": d["sell_vol"].astype("float64"),
                        "Delta": d["delta"].astype("float64"), "Trades": d["trades"].astype("float64")})
    out = out.sort_values("Datetime").reset_index(drop=True)
    out.to_parquet(f)
    return out


def add_clock(bars: pd.DataFrame) -> pd.DataFrame:
    b = bars.copy()
    dt = b["Datetime"]
    b["rth_date"] = (dt + pd.Timedelta(hours=6)).dt.tz_localize(None).dt.normalize()
    b["mod"] = (dt.dt.hour * 60 + dt.dt.minute).astype("int16")
    return b


def period_of(dates) -> np.ndarray:
    d = pd.to_datetime(pd.Series(np.asarray(dates)))
    if d.dt.tz is not None:
        d = d.dt.tz_localize(None)
    return np.where(d <= DISCOVERY_END, "discovery",
                    np.where(d >= HOLDOUT_START, "holdout", "gap")).astype(object)


def fresh_of(dates) -> np.ndarray:
    """discovery / holdout (to 2026-04-29) / fresh (2026-04-30 on)."""
    d = pd.to_datetime(pd.Series(np.asarray(dates)))
    if d.dt.tz is not None:
        d = d.dt.tz_localize(None)
    return np.where(d <= DISCOVERY_END, "discovery",
                    np.where(d >= FRESH_START, "fresh",
                             np.where(d >= HOLDOUT_START, "holdout", "gap"))).astype(object)
