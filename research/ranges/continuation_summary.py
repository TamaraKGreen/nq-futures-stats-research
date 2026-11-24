"""Summarise continuation pullback trades (DEV)."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "output"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)
tr = pd.read_parquet(OUT / "continuation_dev.parquet")
CF = ["s.25_1R", "s.25_2R", "s.5_1R", "s.5_2R", "s.5_ext"]
LV = ["edge", "q75", "vwap", "mid", "open", "close"]
tr["level"] = pd.Categorical(tr.level, LV, ordered=True)


def stats(d, cfs=CF):
    f = d[d.filled]
    o = {"breaks": len(d), "fill%": 100 * d.filled.mean(), "med_min_fill": f.min_to_fill.median()}
    for c in cfs:
        x = f[f[f"{c}_R"].notna()]
        o[f"{c} win%"] = 100 * x[f"{c}_win"].mean()
        o[f"{c} R"] = x[f"{c}_R"].mean()
        o[f"{c} Rnet"] = x[f"{c}_Rnet"].mean()
    return pd.Series(o)


def tbl(d, by, cfs=CF):
    return d.groupby(by, observed=True).apply(lambda x: stats(x, cfs), include_groups=False).round(2)


print("CONTINUATION PULLBACK ENTRIES AFTER THE FIRST BREAK (dev, conservative fills)\n")
print("R = average result in multiples of planned risk (gross); Rnet after $4.50 + 1 tick slippage on stops.")
print("Break-even win% for 1R = 50, for 2R = 33.3 (gross).\n")
print("=" * 120, "\n1. BY LEVEL - all breaks\n")
print(tbl(tr, ["dur", "level"]).to_string())

print("\n" + "=" * 120, "\n2. BY MOMENTUM AGREEMENT (0-7 signals agreeing with the break) - stop 0.5w configs\n")
tr["agree_b"] = pd.cut(tr.agree, [-1, 1, 3, 5, 7], labels=["0-1", "2-3", "4-5", "6-7"])
print(tbl(tr, ["dur", "agree_b", "level"], ["s.5_1R", "s.5_2R", "s.5_ext"]).to_string())

print("\n" + "=" * 120, "\n3. BY YEAR - agree 6-7, stop 0.5w / 1R and 2R\n")
print(tbl(tr[tr.agree >= 6], ["dur", "level", "year"], ["s.5_1R", "s.5_2R"]).to_string())

print("\n" + "=" * 120, "\n4. RISK SIZE: median planned risk in points (stop 0.5w) by dur\n")
print(tr[tr.filled].groupby("dur")["s.5_1R_risk"].describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
