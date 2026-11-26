"""
Statmap projected levels (Statmap_5 logic, 60-candle lookback) for two settings:
  12H "6pm-6am"   the 18:00-06:00 ET candle (one per session)
  4H              18-22, 22-02, 02-06, 06-10, 10-14, 14-17 ET candles (anchored at 18:00)
Per candle (from Statmap): manipulation wick = against the candle's direction (bull: open-low, bear: high-open),
distribution = with it (bull: high-open, bear: open-low). Levels for a candle = its OPEN +/- the median of
the prior 60 candles of the SAME slot:
  M+ / M-   open +/- median manipulation         D+ / D-   open +/- median distribution
CURRENT  hit rate of a candle's own levels inside that candle.
PRIOR    the previous candle's levels (previous 4H candle; previous session's 18:00-06:00 candle for 12H),
         split into "already hit in their own candle" and "untested", hit inside the CURRENT candle.
Distance unit: ATR = mean of the prior 14 full-session (18:00-17:00) ranges.
EXPECTED = the hit rate distance alone gives: the share of candles of the same setting/slot whose excursion
from the open (up or down, pooled - a symmetric, real-path benchmark) reaches that distance, evaluated at
each level's own distance and averaged.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from research.scripts.load import load

OUT = ROOT / "research" / "output"
LOOK = 60
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)


def candles(d, setting):
    sm = (d["mod"] - 1080) % 1440
    if setting == "12H 18-06":
        d = d[sm < 720].assign(slot=0)
    else:
        d = d.assign(slot=np.minimum(sm // 240, 5))
    g = d.groupby(["sess", "slot"], sort=True)
    c = g.agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"), n=("close", "size"),
              adj=("roll_adj", "first")).reset_index()
    full = {0: 240, 1: 240, 2: 240, 3: 240, 4: 240, 5: 180} if setting != "12H 18-06" else {0: 720}
    c = c[c.n >= 0.8 * c.slot.map(full)].reset_index(drop=True)
    bull = c.c > c.o
    c["manip"] = np.where(bull, c.o - c.l, c.h - c.o)
    c["distr"] = np.where(bull, c.h - c.o, c.o - c.l)
    for k in ("manip", "distr"):
        c[f"med_{k}"] = c.groupby("slot")[k].transform(lambda s: s.shift(1).rolling(LOOK, min_periods=30).median())
    c["Mu"], c["Md"] = c.o + c.med_manip, c.o - c.med_manip
    c["Du"], c["Dd"] = c.o + c.med_distr, c.o - c.med_distr
    return c


def main():
    d = load("all")
    naive = d.ts.dt.tz_localize(None)
    d["mod"] = naive.dt.hour * 60 + naive.dt.minute
    sess_rng = d.groupby("sess").agg(h=("high", "max"), l=("low", "min"), n=("close", "size"))
    atr = (sess_rng.h - sess_rng.l).where(sess_rng.n >= 1200).rolling(14, min_periods=10).mean().shift(1)
    out = {}
    for setting in ("12H 18-06", "4H"):
        c = candles(d, setting)
        c["atr"] = c.sess.map(atr)
        c = c.dropna(subset=["med_manip", "atr"]).reset_index(drop=True)
        c["xu"], c["xd"] = (c.h - c.o) / c.atr, (c.o - c.l) / c.atr          # excursions in ATR
        # distance curve per slot: P(excursion >= x), pooled up/down
        curve = {s: np.sort(np.r_[g.xu.to_numpy(), g.xd.to_numpy()]) for s, g in c.groupby("slot")}

        def expected(slot, dist_atr):
            cv = curve[slot]
            return 1 - np.searchsorted(cv, dist_atr, side="left") / len(cv)
        rows = []
        for _, r in c.iterrows():
            rec = {"sess": r.sess, "slot": r.slot, "atr": r.atr}
            for lv, up in (("Mu", True), ("Md", False), ("Du", True), ("Dd", False)):
                dist = abs(r[lv] - r.o) / r.atr
                rec[f"{lv}_dist"] = dist
                rec[f"{lv}_hit"] = float(r.h >= r[lv]) if up else float(r.l <= r[lv])
                rec[f"{lv}_exp"] = expected(r.slot, dist)
            rows.append(rec)
        cur = pd.DataFrame(rows)
        # PRIOR candle levels, inside the current candle (roll-shifted into the current contract)
        prev = c.shift(1)
        if setting == "4H":
            adjacent = ((c.sess == prev.sess) & (c.slot == prev.slot + 1)) | ((c.slot == 0) & (prev.slot == 5))
        else:
            adjacent = pd.Series(True, index=c.index)
        pr = []
        adjacent = adjacent.fillna(False) & (c.index > 0)
        for k in c.index[adjacent]:
            r, p = c.loc[k], c.loc[k - 1]
            shift = p.adj - r.adj
            for lv, up in (("Mu", True), ("Md", False), ("Du", True), ("Dd", False)):
                L = p[lv] + shift
                own_hit = (p.h >= p[lv]) if up else (p.l <= p[lv])
                above = L > r.o
                dist = abs(L - r.o) / r.atr
                hit = float(r.h >= L) if above else float(r.l <= L)
                pr.append({"sess": r.sess, "slot": r.slot, "level": lv, "status": "already hit in prior candle" if own_hit else "untested",
                           "side_now": "above open" if above else "below open", "dist": dist, "hit": hit,
                           "exp": expected(r.slot, dist)})
        out[setting] = (cur, pd.DataFrame(pr))
    return out


def report(out):
    bins = [-.001, .1, .25, .5, .75, 1, 1.5, np.inf]
    labs = ["<0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75-1", "1-1.5", ">1.5"]
    for setting, (cur, pr) in out.items():
        cur["year"] = cur.sess.dt.year
        print("=" * 110, f"\n{setting}  (candles {len(cur)}, {cur.sess.min().date()} -> {cur.sess.max().date()})\n")
        print("CURRENT candle levels - hit inside the candle:")
        rows = []
        for slot, g in cur.groupby("slot"):
            rec = {"slot": slot, "n": len(g)}
            for lv in ("Mu", "Md", "Du", "Dd"):
                rec[f"{lv} hit %"] = 100 * g[f"{lv}_hit"].mean(); rec[f"{lv} exp %"] = 100 * g[f"{lv}_exp"].mean()
                rec[f"{lv} dist ATR"] = g[f"{lv}_dist"].median()
            rec["either M %"] = 100 * np.maximum(g.Mu_hit, g.Md_hit).mean()
            rec["both M %"] = 100 * (g.Mu_hit * g.Md_hit).mean()
            rows.append(rec)
        T = pd.DataFrame(rows).set_index("slot")
        if setting == "4H":
            T.index = ["18-22", "22-02", "02-06", "06-10", "10-14", "14-17"][: len(T)]
        print(T.round(2).to_string())
        y = cur.groupby("year").apply(lambda g: pd.Series({
            "M hit % (avg both sides)": 50 * (g.Mu_hit.mean() + g.Md_hit.mean()), "M exp %": 50 * (g.Mu_exp.mean() + g.Md_exp.mean()),
            "D hit %": 50 * (g.Du_hit.mean() + g.Dd_hit.mean()), "D exp %": 50 * (g.Du_exp.mean() + g.Dd_exp.mean())}),
            include_groups=False)
        print("\nby year (all slots):"); print(y.round(1).to_string())
        print("\nPRIOR candle levels - hit inside the CURRENT candle, by distance from the current open (ATR):")
        pr["dist_b"] = pd.cut(pr.dist, bins, labels=labs)
        g = pr.groupby(["status", "dist_b"], observed=True).agg(n=("hit", "size"), hit=("hit", "mean"), exp=("exp", "mean"))
        g["hit %"], g["expected %"] = 100 * g.hit, 100 * g.exp
        g["edge pp"] = g["hit %"] - g["expected %"]
        print(g[["n", "hit %", "expected %", "edge pp"]].round(1).to_string())
        print("\n  overall:", pr.groupby("status").apply(lambda x: f"n {len(x)}  hit {100*x.hit.mean():.1f}%  expected {100*x.exp.mean():.1f}%",
                                                         include_groups=False).to_dict())
        print()


if __name__ == "__main__":
    out = main()
    for s, (cur, pr) in out.items():
        cur.to_parquet(OUT / f"statmap_{s.split()[0]}_current.parquet", index=False)
        pr.to_parquet(OUT / f"statmap_{s.split()[0]}_prior.parquet", index=False)
    report(out)
