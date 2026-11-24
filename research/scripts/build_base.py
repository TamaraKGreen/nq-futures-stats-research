import glob, sys, pandas as pd, numpy as np
src = sys.argv[1]; out = sys.argv[2]
files = sorted(glob.glob(f"{src}/*NQ_ticks_*_1min.csv"))
dfs = []
for f in files:
    d = pd.read_csv(f)
    d["ts"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("America/New_York")
    dfs.append(d.drop(columns="timestamp"))
df = pd.concat(dfs).sort_values("ts")
print("rows", len(df), "dup ts", df.ts.duplicated().sum())
df = df.drop_duplicates("ts").reset_index(drop=True)
# sanity
bad = (df.high < df[["open","close"]].max(axis=1)) | (df.low > df[["open","close"]].min(axis=1)) | (df.high < df.low)
print("bad ohlc", bad.sum(), "nonpos", (df[["open","high","low","close"]]<=0).any(axis=1).sum())
print("unclassified vol share", 1 - (df.buy_vol+df.sell_vol).sum()/df.volume.sum())
print("delta==buy-sell", (df.delta == df.buy_vol-df.sell_vol).mean())
ticks = ((df[["open","high","low","close"]]*4) % 1 != 0).any(axis=1).sum(); print("off-tick prices", ticks)
# session date (18:00 ET anchor)
df["sess"] = (df.ts - pd.Timedelta(hours=18)).dt.date
spb = df.groupby("sess").size()
print("sessions", len(spb), "median bars", spb.median(), "sessions<1000 bars", (spb<1000).sum())
# gaps between consecutive bars within session > 5 min (excluding 17-18 break)
g = df.ts.diff().dt.total_seconds()/60
big = df[(g>5) & (df.sess==df.sess.shift())]
print("intra-session gaps >5min:", len(big))
print(big.assign(gap=g[big.index])[["ts","gap"]].sort_values("gap").tail(10).to_string())
# roll check: largest abs jumps close->next open across consecutive minutes
j = (df.open - df.close.shift()).abs()
jj = df.assign(jump=j, prev_ts=df.ts.shift())[(g<=2)]
print(jj.nlargest(15,"jump")[["prev_ts","ts","jump"]].to_string())
df.to_parquet(out)
