"""Descriptive break -> retrace statistics on the DEV range events, with the offset-window null."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "output"
ev = pd.read_parquet(OUT / "range_events_dev.parquet")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)

base = ev[ev.full_fwd & ~ev.partial_sess.astype(bool)].copy()
brk = base[base.broke].copy()
brk["year"] = brk.sess.dt.year
h = brk.slot_min // 60
brk["block"] = np.select([(h >= 18) | (h < 2), h < 8, h < 12], ["Asia 18-02", "London 02-08", "NY AM 08-12"], "NY PM 12-17")
T = ["q75", "mid", "open", "close", "q25", "far"]
STOPS = (0.25, 0.5, 1.0)          # stop = this many range-widths beyond the broken edge


def summary(d):
    o = {"breaks": len(d)}
    for t in T:
        o[f"{t}%"] = 100 * d[f"hit_{t}"].mean()
    o["close_back%"] = 100 * d.close_back.mean()
    o["med_min_mid"] = d.min_mid.median()
    for s in STOPS:   # P(mid before extension reaches s widths)
        o[f"mid<{s}w%"] = 100 * ((d.hit_mid == 1) & (d.mae_mid < s * d.width)).mean()
    o["mid<0.5w_opt%"] = 100 * ((d.hit_mid == 1) & (d.maeo_mid < 0.5 * d.width)).mean()
    o["med_w_pts"] = d.width.median()
    return pd.Series(o)


def table(d, by):
    return d.groupby(by, observed=True).apply(summary, include_groups=False).round(1)


print("=" * 100, "\n1. BREAK RATES (all valid ranges, full forward horizon)\n")
br = base.groupby(["dur", "offset"]).apply(lambda d: pd.Series({
    "ranges": len(d), "broke%": 100 * d.broke.mean(), "both_same_bar%": 100 * d.ambiguous.mean(),
    "up_share%": 100 * (d.side == "up").sum() / d.broke.sum(), "med_brk_min": d.brk_min.median()}),
    include_groups=False).round(1)
print(br.to_string())

print("\n", "=" * 100, "\n2. RETRACE RATES AFTER A BREAK - clock ranges (offset 0) vs 7-min offset null\n")
t2 = table(brk, ["dur", "offset"])
print(t2.to_string())

clock = brk[brk.offset == 0]
null = brk[brk.offset == 7]

print("\n", "=" * 100, "\n3. BY SESSION BLOCK (clock ranges)\n")
print(table(clock, ["dur", "block"]).to_string())
print("\n   null (offset 7)\n")
print(table(null, ["dur", "block"]).to_string())

print("\n", "=" * 100, "\n4. BY WIDTH vs SAME-SLOT NORM (w_slot quintile, clock ranges)\n")
clock = clock.assign(wq=clock.groupby("dur").w_slot.transform(lambda s: pd.qcut(s, 5, labels=["Q1 tight", "Q2", "Q3", "Q4", "Q5 wide"])))
print(table(clock, ["dur", "wq"]).to_string())

print("\n", "=" * 100, "\n5. BY MINUTES TO BREAK (clock ranges)\n")
clock["brk_b"] = pd.cut(clock.brk_min / clock.dur, [0, .1, .25, .5, 1, 2, 4], labels=["<0.1d", ".1-.25d", ".25-.5d", ".5-1d", "1-2d", "2-4d"])
print(table(clock, ["dur", "brk_b"]).to_string())

print("\n", "=" * 100, "\n6. BY CLOSE POSITION toward the broken edge (clock ranges)\n")
cp = np.where(clock.side == "up", clock.close_pos, 1 - clock.close_pos)       # 1 = closed AT the edge that later broke
clock["close_near_edge"] = pd.cut(cp, [-.01, .2, .4, .6, .8, 1.01], labels=["0-.2 far", ".2-.4", ".4-.6", ".6-.8", ".8-1 at edge"])
print(table(clock, ["dur", "close_near_edge"]).to_string())

print("\n", "=" * 100, "\n7. BY YEAR (clock ranges)\n")
print(table(clock, ["dur", "year"]).to_string())

print("\n", "=" * 100, "\n8. BY SLOT - 60m (clock)\n")
print(table(clock[clock.dur == 60], ["slot"]).to_string())

clock.to_parquet(OUT / "range_breaks_dev_clock.parquet", index=False)
null.to_parquet(OUT / "range_breaks_dev_null.parquet", index=False)
