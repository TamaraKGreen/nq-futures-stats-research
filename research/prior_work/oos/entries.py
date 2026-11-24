"""
Entry methods, with and without the plan's context filters.

    py -m oos.entries

PRE-REGISTERED 19 Nov 2026, before any of these were run:

  A  1h candle continuation  - C2 closes beyond C1's high (low): enter at C3's
     open in that direction. Target C2 extreme + x * C2 range, x in {0, 0.5}.
     Stop C2 mid or C2 far extreme. Exit at C3 close. C3 hours 09:00-15:00.
     Context: none / day's IB narrow (C3 >= 11:00) / IB wide (C3 >= 11:00).
  B  IB breakout             - stop entry at the first IB break (10:30-15:00).
     Stop IB mid or opposite IB edge. Target 1x IB width or hold to 15:59.
     Context: all / narrow IB / wide IB.
  C  ORB breakout            - same on the 09:30-09:45 range, breaks 09:45-15:00.
     Context: all / narrow ORB / wide ORB.
  D  IB second side          - after the first IB break, the first return to
     the IB mid: limit entry there toward the opposite IB edge (target).
     Stop at the extreme made since 10:30 on the broken side. Exit 15:59.
     Context: all / narrow IB / wide IB.

Protocol: for every (method, context) the grid variant is chosen on DISCOVERY by
average net R; that single variant is then reported on the holdout. Every
variant is written to results/entries_all.csv. Same fill rules and 1 pt cost as
tradability.py. Results are reported in R (multiples of initial risk) by era,
because a fixed 1 pt cost meant something very different at NQ 2,000 vs 25,000.
"""
import numpy as np
import pandas as pd

from .data import load_nq, period_of
from . import families as F
from .tradability import Days, build_levels, COST, OUT

ERA_EDGES = ["2015-01-01", "2020-01-01"]


def _era(day):
    d = pd.Timestamp(day)
    if period_of([d])[0] == "holdout":
        return "unseen"
    return "2010-14" if d < pd.Timestamp(ERA_EDGES[0]) else ("2015-19" if d < pd.Timestamp(ERA_EDGES[1]) else "2020-Mar25")


# ------------------------------------------------------------------- A ------

def method_a(D, h, ibq, x, stop_kind, context):
    rows = []
    hh = h[h["n"] >= 45]
    idx = hh.index
    for (d, hr) in idx:
        if hr < 9 or hr > 15:
            continue
        if context != "none" and hr < 11:
            continue
        if context == "narrow IB" and ibq.get(d) != "Q1 narrow":
            continue
        if context == "wide IB" and ibq.get(d) != "Q5 wide":
            continue
        try:
            c1, c2, c3 = hh.loc[(d, hr - 2)], hh.loc[(d, hr - 1)], hh.loc[(d, hr)]
        except KeyError:
            continue
        rng = c2["hi"] - c2["lo"]
        if rng <= 0:
            continue
        if c2["c"] > c1["hi"]:
            dirn, tgt = 1, c2["hi"] + x * rng
            stop = (c2["hi"] + c2["lo"]) / 2 if stop_kind == "mid" else c2["lo"]
        elif c2["c"] < c1["lo"]:
            dirn, tgt = -1, c2["lo"] - x * rng
            stop = (c2["hi"] + c2["lo"]) / 2 if stop_kind == "mid" else c2["hi"]
        else:
            continue
        e = c3["o"]
        if dirn * (tgt - e) <= 0 or dirn * (e - stop) <= 0:
            continue
        r = D.sim(d, F._sm(f"{hr:02d}:00"), e, dirn, tgt, stop, F._sm(f"{hr:02d}:59"), entry_bar_is_open=True)
        if r:
            rows.append({"day": d, **r})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- B and C ---

def method_breakout(D, b, W, q, start, stop_kind, target_kind, context):
    rows = []
    w = W["h"] - W["l"]
    tu = F.first_touch(b, start, "15:00", W["h"], "up")
    td = F.first_touch(b, start, "15:00", W["l"], "dn")
    want = {"narrow": "Q1 narrow", "wide": "Q5 wide"}.get(context)
    for d in W.index:
        if want and q.get(d) != want:
            continue
        u, dn = tu.get(d), td.get(d)
        if pd.isna(u) and pd.isna(dn):
            continue
        hi, lo, ww = W.at[d, "h"], W.at[d, "l"], w.get(d)
        if not np.isfinite(ww) or ww <= 0:
            continue
        mid = (hi + lo) / 2
        if pd.notna(u) and (pd.isna(dn) or u < dn):
            dirn, e, t0 = 1, hi, u
            stop = mid if stop_kind == "mid" else lo
        else:
            dirn, e, t0 = -1, lo, dn
            stop = mid if stop_kind == "mid" else hi
        tgt = e + dirn * ww if target_kind == "1x" else e + dirn * 1e9
        r = D.sim(d, t0, e, dirn, tgt, stop, F._sm("15:59"))
        if r:
            rows.append({"day": d, **r})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- D ------

def method_second_side(D, b, ib, q, context):
    rows = []
    want = {"narrow": "Q1 narrow", "wide": "Q5 wide"}.get(context)
    tu = F.first_touch(b, "10:30", "16:00", ib["h"], "up")
    td = F.first_touch(b, "10:30", "16:00", ib["l"], "dn")
    t_ib, t_end = F._sm("10:30"), F._sm("15:59")
    for d in ib.index:
        if want and q.get(d) != want:
            continue
        u, dn = tu.get(d), td.get(d)
        if (pd.isna(u) and pd.isna(dn)) or d not in D.span:
            continue
        up = pd.notna(u) and (pd.isna(dn) or u < dn)
        tb = u if up else dn
        hi, lo = ib.at[d, "h"], ib.at[d, "l"]
        mid = (hi + lo) / 2
        s, e = D.span[d]
        sm = D.sm[s:e]
        i_b = s + np.searchsorted(sm, tb)
        i_end = s + np.searchsorted(sm, t_end, side="right")
        Hs, Ls = D.H[i_b + 1:i_end], D.L[i_b + 1:i_end]
        back = np.flatnonzero((Ls <= mid) & (Hs >= mid))
        if not len(back):
            continue
        j = i_b + 1 + back[0]
        i_ib = s + np.searchsorted(sm, t_ib)
        if up:
            dirn, tgt, stop = -1, lo, D.H[i_ib:j + 1].max()
        else:
            dirn, tgt, stop = 1, hi, D.L[i_ib:j + 1].min()
        r = D.sim(d, int(D.sm[j]), mid, dirn, tgt, stop, t_end)
        if r:
            rows.append({"day": d, **r})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- run --

def _stats(t):
    if not len(t):
        return {"trades": 0}
    t = t.assign(gR=t["gross"] / t["risk"], nR=t["net"] / t["risk"])
    se = t["nR"].std(ddof=1) / np.sqrt(len(t)) if len(t) > 1 else np.nan
    return {"trades": len(t), "win_%": round(100 * (t["net"] > 0).mean(), 1),
            "gross_R": round(t["gR"].mean(), 3), "net_R": round(t["nR"].mean(), 3),
            "net_R_se": round(se, 3), "net_pts": round(t["net"].mean(), 2),
            "risk_pts_med": round(t["risk"].median(), 1)}


def run():
    bars = load_nq()
    b, days, atr, ib, pre8, h = build_levels(bars)
    D = Days(b)
    orb = F.window(b, "09:30", "09:45", 0.8).reindex(days)
    ibq = F.disc_quintiles((ib["h"] - ib["l"]) / atr, days)
    orbq = F.disc_quintiles((orb["h"] - orb["l"]) / atr, days)

    specs = []
    for ctx in ("none", "narrow IB", "wide IB"):
        for x in (0, 0.5):
            for sk in ("mid", "far"):
                specs.append(("A 1h candle continuation", ctx, f"target +{x}x C2, stop C2 {sk}",
                              lambda x=x, sk=sk, ctx=ctx: method_a(D, h, ibq, x, sk, ctx)))
    for ctx in ("all", "narrow", "wide"):
        for sk in ("mid", "opp"):
            for tk in ("1x", "close"):
                specs.append(("B IB breakout", ctx, f"stop {sk}, target {tk}",
                              lambda sk=sk, tk=tk, ctx=ctx: method_breakout(D, b, ib, ibq, "10:30", sk, tk, ctx)))
                specs.append(("C ORB breakout", ctx, f"stop {sk}, target {tk}",
                              lambda sk=sk, tk=tk, ctx=ctx: method_breakout(D, b, orb, orbq, "09:45", sk, tk, ctx)))
        specs.append(("D IB second side", ctx, "limit at mid, stop at extreme",
                      lambda ctx=ctx: method_second_side(D, b, ib, ibq, ctx)))

    rows, kept = [], {}
    for meth, ctx, var, fn in specs:
        t = fn()
        if not len(t):
            continue
        t["period"] = period_of(t["day"])
        t["era"] = [_era(d) for d in t["day"]]
        kept[(meth, ctx, var)] = t
        rows.append({"method": meth, "context": ctx, "variant": var, "period": "discovery",
                     **_stats(t[t["period"] == "discovery"])})
        print(f"{meth:26s} {ctx:10s} {var:32s} disc net_R {rows[-1].get('net_R')}", flush=True)
    allv = pd.DataFrame(rows)
    allv.to_csv(f"{OUT}/entries_all.csv", index=False)

    # choose on discovery only, then open the holdout for the chosen variants
    pick = allv.loc[allv.groupby(["method", "context"])["net_R"].idxmax()]
    out = []
    for _, p in pick.iterrows():
        t = kept[(p["method"], p["context"], p["variant"])]
        for era in ("2010-14", "2015-19", "2020-Mar25", "unseen"):
            out.append({"method": p["method"], "context": p["context"], "variant": p["variant"], "era": era,
                        **_stats(t[t["era"] == era])})
    res = pd.DataFrame(out)
    res.to_csv(f"{OUT}/entries_chosen.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== variant chosen on discovery; by era (unseen = holdout) ===")
    print(res.to_string(index=False))


if __name__ == "__main__":
    run()
