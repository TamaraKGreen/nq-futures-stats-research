"""
Hourly stats refined by how much of the DAY / WEEK range is already filled.

    py -m oos.rangefill

Range fill follows "NQ Stats Price Distributions" (Range fill, Full tank =
Percentile 0-100):
  * day  = 18:00-17:00 ET session from its 18:00 open
  * week = Sunday 18:00 open to Friday 17:00
  * excursion so far, each side: up = high_so_far - open, down = open - low_so_far
  * scaled by volatility: z = excursion% / sigma%, sigma = trailing std of the
    anchor's open->close % change (252 days for day, 52 weeks for week), frozen
    at the anchor's open
  * fill % = percentile rank of z among FULL-anchor excursions (all days / weeks)
    -> "up fill 70%" = the day has already gone further up than 70% of days do.
Measured at each hour's open, from bars before the hour starts. Full sample,
no out-of-sample split (as requested).
"""
import numpy as np
import pandas as pd

from .data import load_nq
from . import families as F

OUT = F.__file__.replace("families.py", "results")
BUCKETS = [-1, 25, 50, 75, 101]
BLAB = ["fill <25%", "fill 25-50%", "fill 50-75%", "fill >75%"]


def build():
    b = F.prep(load_nq())
    b["hr"] = (b["mod"] // 60).astype("int8")
    b["wk"] = b["rth_date"] - pd.to_timedelta(b["rth_date"].dt.day_of_week, unit="D")
    # running extremes, day and week
    for key, p in (("rth_date", "d"), ("wk", "w")):
        g = b.groupby(key, sort=False)
        b[f"{p}_open"] = g["Open"].transform("first")
        b[f"{p}_hi"] = g["High"].cummax()
        b[f"{p}_lo"] = g["Low"].cummin()

    def anchor_stats(key, p, look):
        a = b.groupby(key).agg(o=("Open", "first"), h=("High", "max"), l=("Low", "min"), c=("Close", "last"))
        ret = 100 * (a["c"] - a["o"]) / a["o"]
        a["sig"] = ret.rolling(look, min_periods=look // 4).std().shift(1)
        a["zu"] = 100 * (a["h"] - a["o"]) / a["o"] / a["sig"]
        a["zd"] = 100 * (a["o"] - a["l"]) / a["o"] / a["sig"]
        return a

    D = anchor_stats("rth_date", "d", 252)
    W = anchor_stats("wk", "w", 52)
    distU_d, distD_d = np.sort(D["zu"].dropna()), np.sort(D["zd"].dropna())
    distU_w, distD_w = np.sort(W["zu"].dropna()), np.sort(W["zd"].dropna())
    rank = lambda dist, z: 100 * np.searchsorted(dist, z) / len(dist)

    # state just BEFORE each hour starts = last bar of the previous minute
    first = b.groupby(["rth_date", "hr"]).head(1).index
    first = first[first > 0]
    prev = first - 1
    s = b.loc[prev, ["d_open", "d_hi", "d_lo", "w_open", "w_hi", "w_lo", "rth_date", "wk"]].copy()
    s.index = pd.MultiIndex.from_arrays([b.loc[first, "rth_date"].to_numpy(), b.loc[first, "hr"].to_numpy()],
                                        names=["rth_date", "hr"])
    same_day = (s["rth_date"].to_numpy() == s.index.get_level_values(0).to_numpy())
    same_wk = (s["wk"].to_numpy() == (s.index.get_level_values(0) -
               pd.to_timedelta(s.index.get_level_values(0).weekday, unit="D")).to_numpy())
    dsig = s.index.get_level_values(0).map(D["sig"]).to_numpy()
    wk_of = s.index.get_level_values(0) - pd.to_timedelta(s.index.get_level_values(0).weekday, unit="D")
    wsig = wk_of.map(W["sig"]).to_numpy()
    f = pd.DataFrame(index=s.index)
    f["day_up_fill"] = np.where(same_day, rank(distU_d, 100 * (s["d_hi"] - s["d_open"]) / s["d_open"] / dsig), np.nan)
    f["day_dn_fill"] = np.where(same_day, rank(distD_d, 100 * (s["d_open"] - s["d_lo"]) / s["d_open"] / dsig), np.nan)
    f["wk_up_fill"] = np.where(same_wk, rank(distU_w, 100 * (s["w_hi"] - s["w_open"]) / s["w_open"] / wsig), np.nan)
    f["wk_dn_fill"] = np.where(same_wk, rank(distD_w, 100 * (s["w_open"] - s["w_lo"]) / s["w_open"] / wsig), np.nan)
    f.loc[~np.isfinite(dsig), ["day_up_fill", "day_dn_fill"]] = np.nan
    f.loc[~np.isfinite(wsig), ["wk_up_fill", "wk_dn_fill"]] = np.nan
    return f


def run():
    fills = build()
    h = pd.read_parquet(F.__file__.replace("families.py", "cache") + "/table_hourly.parquet")
    h["hr"] = h["hour"].str[:2].astype(int)
    h = h.join(fills, on=["sess", "hr"])
    for c in ("day_up_fill", "day_dn_fill", "wk_up_fill", "wk_dn_fill"):
        h[c + "_b"] = pd.cut(h[c], BUCKETS, labels=BLAB).astype(str)
    # fill on the side each outcome cares about
    up_mid = h["open_pos"].isin(["open lower half", "open below prev L"])   # mid is above
    h["d_side_mid"] = np.where(up_mid, h["day_up_fill_b"], h["day_dn_fill_b"])
    h["w_side_mid"] = np.where(up_mid, h["wk_up_fill_b"], h["wk_dn_fill_b"])
    sw_up = h["swept_side"] == "swept H only"
    h["d_side_sweep"] = np.where(sw_up, h["day_up_fill_b"], h["day_dn_fill_b"])
    h["w_side_sweep"] = np.where(sw_up, h["wk_up_fill_b"], h["wk_dn_fill_b"])
    h.to_parquet(OUT + "/hourly_with_fill.parquet")

    specs = [
        ("sweep_prev_high", ["hour", "open_pos"], "day_up_fill_b", "wk_up_fill_b"),
        ("sweep_prev_low", ["hour", "open_pos"], "day_dn_fill_b", "wk_dn_fill_b"),
        ("hits_prev_mid", ["hour", "open_pos"], "d_side_mid", "w_side_mid"),
        ("sweep_closes_inside", ["hour", "swept_side"], "d_side_sweep", "w_side_sweep"),
    ]
    rows = []
    for o, by, dcol, wcol in specs:
        p = h[h[o].notna()]
        for anchor, col in (("day", dcol), ("week", wcol)):
            g = p[p[col] != "nan"].groupby(by + [col])[o].agg(["mean", "size"]).reset_index()
            base = p.groupby(by)[o].agg(["mean", "size"]).rename(columns={"mean": "cell_rate", "size": "cell_n"})
            g = g.join(base, on=by)
            g.columns = by + ["fill", "rate", "n", "cell_rate", "cell_n"]
            g.insert(0, "anchor", anchor); g.insert(0, "outcome", o)
            g["cond"] = g[by].astype(str).agg(" | ".join, axis=1)
            rows.append(g.drop(columns=by))
    r = pd.concat(rows, ignore_index=True)
    r["lift_pp"] = 100 * (r["rate"] - r["cell_rate"])
    r.to_csv(OUT + "/hourly_by_fill.csv", index=False)
    return r


if __name__ == "__main__":
    r = run()
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
    show = r[r["cond"].str.match(r"(09|10|11):00") & (r["n"] >= 80)]
    piv = show.pivot_table(index=["outcome", "cond", "anchor"], columns="fill", values="rate").mul(100).round(1)
    piv["base"] = show.groupby(["outcome", "cond", "anchor"])["cell_rate"].first().mul(100).round(1)
    nn = show.pivot_table(index=["outcome", "cond", "anchor"], columns="fill", values="n")
    print(piv[["base"] + BLAB].to_string())
    print("\nn per bucket:"); print(nn[BLAB].to_string())
