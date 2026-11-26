"""
The 8 AM range (08:00-08:59 ET) sweep -> midpoint / opposite side claim (Edgeful: 79% / 63%).

Stat     range = high/low of 08:00-08:59. Sweep = first 1-min bar from 09:00 that trades beyond the high or
         low (>= 1 tick). Bars that pierce both sides at once are dropped. After the sweep bar, by 16:00:
         reached the mid? reached the opposite side?
RW       random-walk benchmark: from the swept edge, same distances, path mirrored with probability 1/2.
Trades   (1 pt round-trip cost = MNQ $0.50/side + 1 tick slippage/side)
  T1 fade at the edge   limit at the swept edge (filled 1 tick through it), target the mid,
                        stop k x range width beyond the edge, k in {0.5, 1.0}; exit 16:00 at the latest
  T2 fade the reclaim   first 1-min CLOSE back inside the range after the sweep -> enter at that close,
                        target the mid, stop 1 tick beyond the sweep's extreme so far; exit 16:00
  Each trade scored conservatively (stop wins any ambiguous bar, incl. the entry bar) and optimistically
  (target wins ambiguous bars after entry). The truth is in between.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from research.scripts.load import load

TICK, COST = 0.25, 1.0
OUT = ROOT / "research" / "output"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)


def sim(H, L, C, e0, entry, dirn, tgt, stp, opt):
    """From bar e0 (inclusive) to the end of arrays. Returns points."""
    hs = (L[e0:] <= stp) if dirn > 0 else (H[e0:] >= stp)
    ht = (H[e0:] >= tgt) if dirn > 0 else (L[e0:] <= tgt)
    if opt:
        hs = hs.copy(); hs[0] = False
    ht = ht.copy(); ht[0] = False if not opt else ht[0]
    a = int(np.argmax(hs)) if hs.any() else 10**9
    b = int(np.argmax(ht)) if ht.any() else 10**9
    if a == b == 10**9:
        return dirn * (C[-1] - entry)
    if a < b or (a == b and not opt):
        return dirn * (stp - entry)
    return dirn * (tgt - entry)


def main():
    d = load("all", drop_partial=True)
    naive = d.ts.dt.tz_localize(None)
    d["mod"] = naive.dt.hour * 60 + naive.dt.minute
    d["day"] = naive.dt.normalize()
    rows = []
    for day, g in d[(d["mod"] >= 480) & (d["mod"] < 960)].groupby("day"):
        m = g["mod"].to_numpy(); H = g.high.to_numpy(); L = g.low.to_numpy(); C = g.close.to_numpy()
        r = m < 540
        if r.sum() < 50 or (~r).sum() < 380:
            continue
        hi, lo = H[r].max(), L[r].min(); w = hi - lo; mid = (hi + lo) / 2
        if w <= 0:
            continue
        H2, L2, C2 = H[~r], L[~r], C[~r]
        up = np.flatnonzero(H2 > hi); dn = np.flatnonzero(L2 < lo)
        iu = up[0] if len(up) else 10**9; idn = dn[0] if len(dn) else 10**9
        rec = {"day": day, "w": w}
        if iu == idn == 10**9:
            rec["side"] = "no sweep"; rows.append(rec); continue
        if iu == idn:
            rec["side"] = "both same bar"; rows.append(rec); continue
        side = "high" if iu < idn else "low"; s = min(iu, idn)
        sg = -1 if side == "high" else 1                      # fade direction
        edge, opp = (hi, lo) if side == "high" else (lo, hi)
        post_h, post_l = H2[s + 1:], L2[s + 1:]
        if len(post_h) == 0:
            continue
        hit_mid = (post_l <= mid).any() if side == "high" else (post_h >= mid).any()
        hit_opp = (post_l <= opp).any() if side == "high" else (post_h >= opp).any()
        # random walk from the swept edge over the post-sweep path
        upx, dnx = post_h.max() - edge, edge - post_l.min()
        rw = lambda dist: 0.5 * float(upx >= dist) + 0.5 * float(dnx >= dist)
        # adverse extension beyond the edge before the mid (incl. sweep bar)
        ext = (H2[s:] - edge) if side == "high" else (edge - L2[s:])
        hm = np.flatnonzero((L2[s + 1:] <= mid) if side == "high" else (H2[s + 1:] >= mid))
        mae = np.maximum.accumulate(ext)[hm[0] + 1] if len(hm) else ext.max()
        rec.update(side=side, sweep_min=int(m[~r][s]), hit_mid=float(hit_mid), hit_opp=float(hit_opp),
                   rw_mid=rw(w / 2), rw_opp=rw(w), mae_w=mae / w)
        # T1: limit fade at the edge, filled 1 tick through
        thr = np.flatnonzero((H2[s:] >= edge + TICK) if side == "high" else (L2[s:] <= edge - TICK))
        if len(thr):
            e0 = s + thr[0]
            for k in (0.5, 1.0):
                stp = edge - sg * k * w
                for opt in (False, True):
                    rec[f"T1_k{k}_{'opt' if opt else 'cons'}"] = sim(H2, L2, C2, e0, edge, sg, mid, stp, opt) - COST
                rec[f"T1_k{k}_risk"] = k * w
        # T2: first close back inside after the sweep
        back = np.flatnonzero((C2[s:] < edge) if side == "high" else (C2[s:] > edge))
        if len(back):
            j = s + back[0]
            ext_px = H2[s:j + 1].max() if side == "high" else L2[s:j + 1].min()
            entry = C2[j]; stp = ext_px - sg * TICK
            if sg * (mid - entry) > 0 and j + 1 < len(C2):
                for opt in (False, True):
                    rec[f"T2_{'opt' if opt else 'cons'}"] = sim(H2, L2, C2, j + 1, entry, sg, mid, stp, opt) - COST
                rec["T2_risk"] = abs(entry - stp); rec["T2_reward"] = abs(mid - entry)
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t = main()
    t.to_parquet(OUT / "eight_am.parquet", index=False)
    t["year"] = t.day.dt.year
    sw = t[t.side.isin(["high", "low"])]
    print(f"days {len(t)}  sweep {len(sw)} ({100*len(sw)/len(t):.0f}%)  no sweep {(t.side=='no sweep').sum()}  "
          f"both sides in one bar {(t.side=='both same bar').sum()}\n")
    last2 = sw[sw.day >= sw.day.max() - pd.Timedelta(days=730)]

    def stat(x):
        return pd.Series({"n": len(x), "mid %": 100 * x.hit_mid.mean(), "mid RW %": 100 * x.rw_mid.mean(),
                          "opp side %": 100 * x.hit_opp.mean(), "opp RW %": 100 * x.rw_opp.mean(),
                          "med range pts": x.w.median(), "med adverse before mid (x range)": x.mae_w.median()})
    print("=" * 100, "\n1. THE STAT (sweep 09:00-16:00, outcome by 16:00)\n")
    print(pd.concat({"all 2020-26": stat(sw), "last 2 years": stat(last2), "swept HIGH": stat(sw[sw.side == "high"]),
                     "swept LOW": stat(sw[sw.side == "low"])}, axis=1).T.round(2).to_string())
    print("\nby year:"); print(sw.groupby("year").apply(stat, include_groups=False).round(1).to_string())
    print("\nadverse move beyond the swept edge BEFORE the mid is reached (x range width), percentiles:")
    print(sw.mae_w.quantile([.25, .5, .75, .9]).round(2).to_string())
    print("\n" + "=" * 100, "\n2. AS A TRADE (net of 1 pt cost; cons / opt = conservative / optimistic fills)\n")
    out = []
    for name, base, rk in (("T1 fade at edge, stop 0.5x range", "T1_k0.5", "T1_k0.5_risk"),
                           ("T1 fade at edge, stop 1.0x range", "T1_k1.0", "T1_k1.0_risk"),
                           ("T2 fade the reclaim, stop beyond sweep extreme", "T2", "T2_risk")):
        for lab, x in (("all", sw), ("last 2y", last2)):
            x = x.dropna(subset=[f"{base}_cons"])
            c, o, rsk = x[f"{base}_cons"], x[f"{base}_opt"], x[rk]
            out.append({"trade": name, "period": lab, "n": len(x), "win % cons": 100 * (c > 0).mean(),
                        "win % opt": 100 * (o > 0).mean(), "avg pts cons": c.mean(), "avg pts opt": o.mean(),
                        "avg R cons": (c / rsk).mean(), "avg R opt": (o / rsk).mean(),
                        "med risk pts": rsk.median(),
                        "med reward pts": (x.w / 2).median() if base.startswith("T1") else x.T2_reward.median()})
    print(pd.DataFrame(out).round(2).to_string(index=False))
    print("\nT1 by year (avg pts, conservative / optimistic):")
    for k in ("0.5", "1.0"):
        g = sw.groupby("year").agg(c=(f"T1_k{k}_cons", "mean"), o=(f"T1_k{k}_opt", "mean"))
        print(f"  stop {k}x:", {y: f"{a:+.1f} / {b:+.1f}" for y, (a, b) in g.round(1).iterrows()})
