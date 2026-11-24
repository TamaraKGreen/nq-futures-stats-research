"""Summarise round-2 continuation trades (DEV). Primary metric: average net R per trade."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "output"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 600); pd.set_option("display.max_columns", 40)
tr = pd.read_parquet(OUT / "continuation2_dev.parquet")
f = tr[tr.filled].copy()
YEARS = f.year.nunique()
TG = ["R2", "EXT", "NEXT"]


def st(d, tgs=TG):
    o = {}
    for t in tgs:
        x = d[d[f"{t}_R"].notna()]
        o[f"{t} n"] = len(x)
        o[f"{t} win%"] = 100 * x[f"{t}_win"].mean() if len(x) else np.nan
        o[f"{t} R"] = x[f"{t}_R"].mean() if len(x) else np.nan
        o[f"{t} Rnet"] = x[f"{t}_Rnet"].mean() if len(x) else np.nan
        if t != "R2":
            o[f"{t} med rr"] = x[f"{t}_rr"].median() if len(x) else np.nan
    return pd.Series(o)


def tbl(d, by, tgs=TG):
    return d.groupby(by, observed=True).apply(lambda x: st(x, tgs), include_groups=False).round(2)


print("ROUND 2 CONTINUATION (dev).  R / Rnet = average result per trade in units of planned risk.")
print("Stop = level -/+ 0.5 range width for every entry. 'med rr' = median reward:risk of that target.\n")

print("=" * 130, "\n0. FILL / CONFIRMATION RATES (share of breaks where the entry triggered)\n")
print((tr.groupby(["dur", "level", "entry"]).filled.mean().unstack() * 100).round(1).to_string())

print("\n" + "=" * 130, "\n1. ENTRY x TARGET, by duration and level (all contexts)\n")
print(tbl(f, ["dur", "level", "entry"]).to_string())

print("\n" + "=" * 130, "\n2. HIGHER-TIMEFRAME AGREEMENT (60m, 240m, session open agreeing with the break: 0-3), levels pooled\n")
print(tbl(f, ["dur", "entry", "htf_agree"]).to_string())

print("\n" + "=" * 130, "\n2b. vs 09:30 OPEN (RTH ranges only; +1 = break direction agrees)\n")
print(tbl(f[f.vs_rth_open.notna()], ["dur", "entry", "vs_rth_open"]).to_string())

print("\n" + "=" * 130, "\n3. TIME OF DAY (block of the range end), levels pooled\n")
print(tbl(f, ["dur", "block", "entry"]).to_string())

print("\n" + "=" * 130, "\n4. STACKED FILTER: htf_agree = 3 AND momentum agree >= 5, by block (levels pooled)\n")
g = f[(f.htf_agree == 3) & (f.agree >= 5)]
print(tbl(g, ["dur", "block", "entry"]).to_string())

print("\n" + "=" * 130, "\n5. BEST CELLS (dur x level x entry x target x block, htf_agree = 3) with n >= 200 - per-year Rnet\n")
rows = []
h = f[f.htf_agree == 3]
for key, d in h.groupby(["dur", "level", "entry", "block"], observed=True):
    for t in TG:
        x = d[d[f"{t}_R"].notna()]
        if len(x) < 200:
            continue
        yr = x.groupby("year")[f"{t}_Rnet"].mean()
        rows.append(dict(zip(["dur", "level", "entry", "block"], key), target=t, n=len(x),
                         per_year=round(len(x) / YEARS), win=100 * x[f"{t}_win"].mean(),
                         R=x[f"{t}_R"].mean(), Rnet=x[f"{t}_Rnet"].mean(),
                         yrs_pos=f"{int((yr > 0).sum())}/{len(yr)}", worst_yr=yr.min(),
                         med_risk_pts=x.risk.median()))
best = pd.DataFrame(rows).sort_values("Rnet", ascending=False)
print(f"cells tested: {len(best)}\n")
print(best.head(25).round(2).to_string(index=False))
print("\n...bottom 5:\n")
print(best.tail(5).round(2).to_string(index=False))
