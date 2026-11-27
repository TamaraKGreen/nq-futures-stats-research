"""
Combine the yearly 15-year files in research/data/raw_15y/ into one clean 1-minute OHLCV dataset.

    python3 research/scripts/build_15y.py                 # all symbols -> research/data/futures_1min_15y.parquet
    python3 research/scripts/build_15y.py --symbols nq es --no-extend-nq

Output columns (one row per symbol per minute that traded):
    symbol        nq | es | gc
    ts            bar open time, tz-aware America/New_York
    open high low close volume
    sess          trading date, 18:00 ET anchor (Sunday evening -> Monday)
    src           15y (the original CSV export) | nt (our NQ dataset, used only to extend NQ, see below)

Timezones of the raw files (checked, not assumed):
    nq  New York local time. 96.9% of 2020-2025 closes match our NinjaTrader/CSV dataset exactly at the same
        timestamp (volume correlation 0.996); the rest sit around contract rolls.
    es, gc  UTC. The Sunday open is 23:00 in January and 22:00 in July (= 18:00 ET), and ES volume peaks at the
        15:00-16:00 ET close.

--extend-nq (default on): the NQ export ends 2025-03-14. Bars after its last timestamp are taken from our
research dataset (research/data/nq_1min_2020_2026_*.parquet, raw front-month like the export), so NQ runs to
2026-11-24. Turn off with --no-extend-nq.

Prices are as exported (unadjusted front-month for NQ; ES/GC not verified). Anything that compares prices
across a contract roll (multi-day levels, ATR) should allow for roll gaps.

Also here:
    load_15y(symbols=None)          read the combined file (builds it if missing)
    resample(df, "5min")            1-minute bars -> any bar size, per symbol, 18:00-anchored sessions respected
    ticks_to_1min(ticks)            tick data (timestamp, price, volume[, aggressor]) -> 1-minute OHLCV(+delta)
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "research" / "data" / "raw_15y"
OUT = ROOT / "research" / "data" / "futures_1min_15y.parquet"
NY = "America/New_York"
SOURCE_TZ = {"nq": NY, "es": "UTC", "gc": "UTC"}


def read_raw(sym):
    files = sorted(glob.glob(str(RAW / f"{sym}_1min_*.parquet")))
    if not files:
        raise FileNotFoundError(f"no {sym}_1min_*.parquet in {RAW}")
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    naive = pd.to_datetime(d["Date"] + " " + d["Time"])
    tz = SOURCE_TZ[sym]
    if tz == "UTC":
        ts = naive.dt.tz_localize("UTC").dt.tz_convert(NY)
    else:
        # New York wall-clock: the autumn 01:00-01:59 hour happens twice; the export has it once, so treat it as
        # daylight time (the first pass). The spring 02:00-02:59 hour doesn't exist; shift such rows forward.
        ts = naive.dt.tz_localize(NY, ambiguous=np.ones(len(naive), bool), nonexistent="shift_forward")
    out = pd.DataFrame({"symbol": sym, "ts": ts,
                        "open": d["Open"].astype(float), "high": d["High"].astype(float),
                        "low": d["Low"].astype(float), "close": d["Close"].astype(float),
                        "volume": d["Volume"].astype("int64"), "src": "15y"})
    return out


def nq_extension(after):
    files = sorted(glob.glob(str(ROOT / "research" / "data" / "nq_1min_2020_2026_*.parquet")))
    x = pd.concat([pd.read_parquet(f, columns=["ts", "open", "high", "low", "close", "volume"]) for f in files])
    x = x[x.ts > after].sort_values("ts")
    return x.assign(symbol="nq", src="nt", volume=x.volume.astype("int64"))[
        ["symbol", "ts", "open", "high", "low", "close", "volume", "src"]]


def add_session(df):
    wall = df.ts.dt.tz_localize(None)
    df["sess"] = (wall + pd.Timedelta(hours=6)).dt.normalize()     # 18:00 ET -> next calendar day
    return df


def build(symbols=("nq", "es", "gc"), extend_nq=True, out=OUT):
    parts = []
    for sym in symbols:
        d = read_raw(sym)
        if sym == "nq" and extend_nq:
            d = pd.concat([d, nq_extension(d.ts.max())], ignore_index=True)
        d = d.drop_duplicates(["ts"], keep="first").sort_values("ts")
        bad = (d.high < d[["open", "close"]].max(axis=1)) | (d.low > d[["open", "close"]].min(axis=1))
        print(f"{sym}: {len(d):>10,} bars | {d.ts.iloc[0]} -> {d.ts.iloc[-1]} | "
              f"src {d.src.value_counts().to_dict()} | OHLC inconsistencies {int(bad.sum())}")
        parts.append(d)
    df = add_session(pd.concat(parts, ignore_index=True))
    df["symbol"] = df.symbol.astype("category")
    df["src"] = df.src.astype("category")
    df = df[["symbol", "ts", "open", "high", "low", "close", "volume", "sess", "src"]].reset_index(drop=True)
    df.to_parquet(out, index=False, compression="zstd")
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.0f} MB, {len(df):,} rows)")
    return df


def load_15y(symbols=None):
    if not OUT.exists():
        build()
    df = pd.read_parquet(OUT)
    if symbols is not None:
        symbols = [symbols] if isinstance(symbols, str) else list(symbols)
        df = df[df.symbol.isin(symbols)].reset_index(drop=True)
    return df


def resample(df, rule="5min"):
    """1-minute bars -> `rule` bars (e.g. '5min', '15min', '1h'), per symbol. Bars are aligned to the clock and never
    straddle the 18:00 ET session boundary."""
    out = []
    for sym, g in df.groupby("symbol", observed=True):
        g = g.set_index("ts")
        r = g.groupby("sess").resample(rule, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
        out.append(r.reset_index().assign(symbol=sym))
    return pd.concat(out, ignore_index=True)[["symbol", "ts", "open", "high", "low", "close", "volume", "sess"]]


def ticks_to_1min(ticks, tz=NY):
    """Ticks (columns timestamp, price, volume, optional aggressor 'Buy'/'Sell') -> 1-minute OHLCV, with
    buy_vol / sell_vol / delta / trades when the aggressor is present."""
    t = ticks.copy()
    ts = pd.to_datetime(t.pop("timestamp") if "timestamp" in t.columns else t.index, utc=True)
    t.index = pd.DatetimeIndex(ts).tz_convert(tz)
    agg = {"price": ["first", "max", "min", "last", "count"], "volume": "sum"}
    if "aggressor" in t.columns:
        side = t["aggressor"].astype(str).str.strip().str.lower()
        t["buy_vol"] = np.where(side == "buy", t["volume"], 0)
        t["sell_vol"] = np.where(side == "sell", t["volume"], 0)
        agg.update({"buy_vol": "sum", "sell_vol": "sum"})
    b = t.resample("1min", label="left", closed="left").agg(agg)
    b.columns = ["open", "high", "low", "close", "trades", "volume"] + (["buy_vol", "sell_vol"] if "buy_vol" in t else [])
    b = b.dropna(subset=["open"])
    if "buy_vol" in b:
        b["delta"] = b.buy_vol - b.sell_vol
    b.index.name = "ts"
    return b.reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["nq", "es", "gc"])
    ap.add_argument("--no-extend-nq", action="store_true")
    a = ap.parse_args()
    build(tuple(s.lower() for s in a.symbols), extend_nq=not a.no_extend_nq)
