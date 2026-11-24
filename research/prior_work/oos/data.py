"""
NQ 1-minute bars, joined from your two files, with a lockbox.

  discovery : everything up to the 2025-03-14 RTH session. This is what
              nq_15y_1min.csv covers, i.e. everything your notebooks have seen.
  holdout   : 2025-03-17 RTH session onwards (to 2026-04-29). Only the 2020-2026
              parquet has it and none of the CSV-based notebooks touched it.

The two files agree bar-for-bar where they overlap (median close diff 0.00);
they differ only around contract rolls, so we take the 15y file up to the
14 Mar 2025 close and the 2020-2026 file after it.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
CACHE = Path(__file__).resolve().parent / "cache"
NY = "America/New_York"

DISCOVERY_END = pd.Timestamp("2025-03-14")      # last RTH date your notebooks saw
HOLDOUT_START = pd.Timestamp("2025-03-17")
INNER_SPLIT = "2019-06-01"                      # your existing train/test split

_COLS = ["open", "high", "low", "close", "volume"]


def load_nq(refresh=False) -> pd.DataFrame:
    """Bars in rangelab format: Datetime (tz ET), Open, High, Low, Close, Volume."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "nq_joined.parquet"
    if f.exists() and not refresh:
        return pd.read_parquet(f)
    a = pd.read_parquet(DATA / "nq_combined_1min_bars_15y_csv.parquet", columns=_COLS)
    b = pd.read_parquet(DATA / "nq_combined_1min_bars_2020_2026.parquet", columns=_COLS)
    a.index = a.index.tz_convert(NY)
    b.index = b.index.tz_convert(NY)
    cut = pd.Timestamp("2025-03-14 17:00", tz=NY)
    df = pd.concat([a[a.index < cut], b[b.index >= cut]])
    df = df[~df.index.duplicated(keep="first")].sort_index()
    out = pd.DataFrame({
        "Datetime": df.index,
        "Open": df["open"].to_numpy("float64"),
        "High": df["high"].to_numpy("float64"),
        "Low": df["low"].to_numpy("float64"),
        "Close": df["close"].to_numpy("float64"),
        "Volume": df["volume"].to_numpy("float64"),
    })
    out = out.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    out.to_parquet(f)
    return out


def add_clock(bars: pd.DataFrame) -> pd.DataFrame:
    """Add rth_date (session labelled by its RTH day; 18:00 belongs to the next
    day) and mod (minute of day, ET)."""
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
