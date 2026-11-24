"""Load the NQ 1-minute research dataset. Run from the repo root or pass `root`."""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def load(split="dev", adjusted=False, drop_partial=False):
    """
    split        : 'dev' (default), 'oot', or 'all'. Keep 'oot' untouched until final validation.
    adjusted     : add roll_adj to OHLC (back-adjusted series for cross-session comparisons).
    drop_partial : drop holiday / early-close / outage sessions.
    """
    parts = ["dev", "oot"] if split == "all" else [split]
    df = pd.concat([pd.read_parquet(DATA / f"nq_1min_2020_2026_{p}.parquet") for p in parts],
                   ignore_index=True)
    if drop_partial:
        df = df[~df.partial_sess]
    if adjusted:
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] + df["roll_adj"]
    return df.reset_index(drop=True)
