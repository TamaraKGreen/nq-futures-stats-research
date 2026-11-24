"""
Build the unified NQ 1-minute dataset (Jan 2020 -> today) and the 80/20 dev/OOT split.

Sources
  csv : user's NQ_ticks_YYYY_1min.csv files (2020-01-01 18:00 ET -> 2026-04-29 23:59 ET)
        unadjusted front-month continuous; rolls happen at the 18:00 session open and
        are marked by a single 17:00 ET print of the new contract.
  nt  : NinjaTrader market_history 1-min pulls, NQM6 -> NQU6 -> NQZ6
        (2026-04-30 00:00 ET -> latest pull). buy/sell volume here is up/down-tick
        volume, not aggressor-classified (corr of delta vs csv delta = 0.97 on overlap).

Output columns
  ts (tz-aware ET), open, high, low, close, volume, buy_vol, sell_vol, delta, trades,
  src, contract, sess (18:00-anchored trading date), sess_bars, partial_sess,
  roll_adj (add to raw prices for a back-adjusted series anchored on the latest contract),
  split ('dev' | 'oot')
"""
import glob, json, sys
import numpy as np, pandas as pd

ET = "America/New_York"
RAW_NT = "research/data/raw_nt"   # extract raw_nt_2026.tar.gz into research/data first
BASE = "research/data/nq_base_2020_2026.parquet"
OUT = "research/data/nq_1min_2020_2026.parquet"
CSV_END = pd.Timestamp("2026-04-30 00:00", tz=ET)          # NT takes over here
NT_WINDOWS = {   # contract: [start, end) in ET, roll at session open after volume crossover
    "NQM6": (pd.Timestamp("2026-04-30 00:00", tz=ET), pd.Timestamp("2026-06-12 17:00", tz=ET)),
    "NQU6": (pd.Timestamp("2026-06-14 18:00", tz=ET), pd.Timestamp("2026-09-11 17:00", tz=ET)),
    "NQZ6": (pd.Timestamp("2026-09-13 18:00", tz=ET), pd.Timestamp("2099-01-01", tz=ET)),
}
# NT roll gaps from daily closes on the last session before the roll (new - old)
NT_ROLL_GAPS = {"2026-06-15": 29967.25 - 29677.75, "2026-09-14": 29691.00 - 29391.75}

# ---------------------------------------------------------------- csv part
base = pd.read_parquet(BASE)
base = base[base.ts < CSV_END].copy()
h17 = base.ts.dt.hour == 17
marks = base[h17 & (base.ts.dt.minute == 0)]                 # roll marker prints
prev_close = base.close.shift(1)
roll_rows = []
for i, r in marks.iterrows():
    # marker prints at 17:00 on day X; the new contract trades from the session labelled X+1
    roll_rows.append(dict(roll_sess=(r.ts.tz_localize(None).normalize() + pd.Timedelta(days=1)),
                          gap=r.open - prev_close[i], how="marker"))
base = base[~h17].copy()                                     # drop all 17:xx prints (27 bars)
base["src"] = "csv"

# ---------------------------------------------------------------- nt part
frames = []
for f in sorted(glob.glob(f"{RAW_NT}/*.txt")):
    j = json.load(open(f))
    if j.get("barType") != "Minute" or not j.get("bars"):
        continue
    b = pd.DataFrame(j["bars"])
    b["contract"] = j["symbol"]
    frames.append(b)
nt = pd.concat(frames)
nt["ts"] = pd.to_datetime(nt.timestamp, utc=True).dt.tz_convert(ET)
keep = np.zeros(len(nt), bool)
for c, (s, e) in NT_WINDOWS.items():
    keep |= ((nt.contract == c) & (nt.ts >= s) & (nt.ts < e)).to_numpy()
nt = nt[keep].drop_duplicates(["contract", "ts"])
nt = nt[nt.ts.dt.hour != 17]
nt = pd.DataFrame(dict(
    ts=nt.ts, open=nt.open, high=nt.high, low=nt.low, close=nt.close,
    volume=nt.upVolume + nt.downVolume, buy_vol=nt.upVolume, sell_vol=nt.downVolume,
    delta=nt.upVolume - nt.downVolume, trades=nt.upTicks + nt.downTicks,
    src="nt", contract=nt.contract))

df = pd.concat([base, nt], ignore_index=True).sort_values("ts").reset_index(drop=True)
assert not df.ts.duplicated().any(), "duplicate timestamps after merge"
# trading date the session belongs to, from WALL-CLOCK time (absolute arithmetic breaks on DST Sundays)
naive = df.ts.dt.tz_localize(None)
df["sess"] = (naive - pd.Timedelta(hours=18)).dt.normalize() + pd.Timedelta(days=1)

# ---------------------------------------------------------------- contracts + roll adjust
rolls = pd.DataFrame(roll_rows)
rolls["roll_sess"] = pd.to_datetime(rolls.roll_sess)
# 2025 Jun/Sep/Dec rolls have no marker print in the csv; detected as 237-244 pt session-open gaps
NOMARK = ["2025-06-17", "2025-09-16", "2025-12-16"]
tmp = base.assign(sess=(base.ts.dt.tz_localize(None) - pd.Timedelta(hours=18)).dt.normalize() + pd.Timedelta(days=1))
fo = tmp.groupby("sess").open.first(); lc = tmp.groupby("sess").close.last().shift(1)
rolls = pd.concat([
    rolls,
    pd.DataFrame(dict(roll_sess=pd.to_datetime(NOMARK), gap=[fo[d] - lc[d] for d in pd.to_datetime(NOMARK)], how="open_gap")),
    pd.DataFrame(dict(roll_sess=pd.to_datetime(list(NT_ROLL_GAPS)), gap=list(NT_ROLL_GAPS.values()), how="nt_daily_close")),
])
# csv/nt boundary: the csv's last contract (H6->M6 roll was 2026-03-18) is NQM6 = same as nt start. No gap.
rolls = rolls.sort_values("roll_sess").reset_index(drop=True)
code = "HMUZ"
def contract_after(d):                                       # front contract that a roll moves INTO
    m = d.month; y = d.year
    q = [3, 6, 9, 12]; nxt = next((x for x in q if x > m), None)
    if nxt is None: nxt, y = 3, y + 1
    return f"NQ{code[q.index(nxt)]}{y % 10}"
rolls["contract"] = rolls.roll_sess.map(contract_after)
first = contract_after(pd.Timestamp("2019-12-20"))           # NQH0 at start of data
idx = np.searchsorted(rolls.roll_sess.values, df.sess.values, side="right")
labels = np.array([first] + rolls.contract.tolist())
df.loc[df.src == "csv", "contract"] = labels[idx][df.src.values == "csv"]
mism = df[(df.src == "nt")].contract.ne(pd.Series(labels[idx], index=df.index)[df.src == "nt"]).sum()
print("nt contract label mismatches vs roll table:", mism)
# Panama back-adjustment: add the sum of all LATER roll gaps
gaps = rolls.gap.to_numpy()
later = np.concatenate([np.cumsum(gaps[::-1])[::-1], [0.0]])  # later[k] = sum gaps[k:]
df["roll_adj"] = later[idx]

# ---------------------------------------------------------------- session quality
sb = df.groupby("sess").size()
df["sess_bars"] = df.sess.map(sb).astype(int)
df["partial_sess"] = df.sess_bars < 1200                     # holidays / early closes / outages
sessions = sb.index.sort_values()
full = sessions[sb.loc[sessions].values >= 1200]

# ---------------------------------------------------------------- 80/20 chronological split
cut = sessions[int(len(sessions) * 0.8)]
df["split"] = np.where(df.sess < cut, "dev", "oot")

df.to_parquet(OUT, index=False)
for s in ("dev", "oot"):
    df[df.split == s].to_parquet(OUT.replace(".parquet", f"_{s}.parquet"), index=False)
rolls.to_csv("research/data/roll_table.csv", index=False)

# ---------------------------------------------------------------- report
print(f"bars {len(df):,}  sessions {len(sessions)}  full {len(full)}  partial {len(sessions)-len(full)}")
print(f"first {df.ts.iloc[0]}  last {df.ts.iloc[-1]}")
print(df.groupby("src").agg(bars=("ts", "size"), first=("ts", "min"), last=("ts", "max")).to_string())
for s in ("dev", "oot"):
    d = df[df.split == s]
    print(f"{s}: {d.sess.nunique()} sessions  {d.sess.min().date()} -> {d.sess.max().date()}  bars {len(d):,}")
print("rolls:", len(rolls)); print(rolls.to_string())
g = df.ts.diff().dt.total_seconds() / 60
gap_in = df[(g > 5) & (df.sess == df.sess.shift())]
print("intra-session gaps > 5 min:", len(gap_in), " by src:", gap_in.src.value_counts().to_dict())
print("partial sessions:\n", sb[sb < 1200].to_string())
