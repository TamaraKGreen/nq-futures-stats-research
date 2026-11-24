"""
Re-measure the hourly stats from NQ_best_stats.md / NQ_stat_shortlist.md on the DEV split,
then show each one three ways:
  rate      how often it happens (your original framing)
  flip      the same question on a random walk with the SAME distances (each hour's path is
            mirrored with probability 1/2): what geometry alone would give
  trade     the stat turned into a trade: enter at the hour open toward the level, target = level,
            stop = same distance the other way (1:1), exit at the hour close otherwise.
            Entry 1 tick slippage, $4.50 RT, 1 tick on stop/time exits. 50% win = no edge.

Hour H is a clock hour (H:00-H:59 ET); P is the previous clock hour of the same session.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from research.scripts.load import load

TICK, COMM = 0.25, 0.225
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)

b = load("dev", drop_partial=True)
naive = b.ts.dt.tz_localize(None)
b["hour"] = naive.dt.hour
b["pos"] = np.arange(len(b))
g = b.groupby(["sess", "hour"], sort=True)
h = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
          n=("close", "size"), i0=("pos", "first"), i1=("pos", "last")).reset_index()
h = h[h.n >= 50]
p = h.copy(); p["hour"] = (p.hour + 1) % 24
h = h.merge(p[["sess", "hour", "open", "high", "low", "close"]], on=["sess", "hour"], suffixes=("", "_p"))
h["mid_p"] = (h.high_p + h.low_p) / 2
h["year"] = h.sess.dt.year

H, L, C = b.high.to_numpy(), b.low.to_numpy(), b.close.to_numpy()


def trade(row, up, dist):
    """Market entry at the hour open toward the level `dist` away; symmetric stop; exit by hour end."""
    i0, i1 = int(row.i0), int(row.i1)
    sg = 1 if up else -1
    entry = row.open + sg * TICK
    tgt = row.open + sg * dist
    stp = entry - sg * abs(tgt - entry)
    hh, ll = H[i0:i1 + 1], L[i0:i1 + 1]
    ht = (hh >= tgt) if up else (ll <= tgt)
    hs = (ll <= stp) if up else (hh >= stp)
    it = np.argmax(ht) if ht.any() else 10**9
    is_ = np.argmax(hs) if hs.any() else 10**9
    risk = abs(tgt - entry)
    if it == 10**9 and is_ == 10**9:
        pts, how = sg * (C[i1] - entry), "time"
    elif is_ <= it:
        pts, how = -risk, "stop"
    else:
        pts, how = risk, "target"
    pts -= COMM + (TICK if how != "target" else 0)
    return pts, how == "target", risk


rows = []
for _, r in h.iterrows():
    up_half = r.open > r.mid_p
    inside = r.low_p <= r.open <= r.high_p
    mx_up, mx_dn = r.high - r.open, r.open - r.low
    rec = {"sess": r.sess, "year": r.year, "hour": r.hour}
    if inside and r.open != r.mid_p:
        near_up = up_half
        d_near = (r.high_p - r.open) if near_up else (r.open - r.low_p)
        d_far = (r.open - r.low_p) if near_up else (r.high_p - r.open)
        rec.update(case="opens " + ("upper" if up_half else "lower") + " half of prior hour",
                   near_hit=(mx_up >= d_near) if near_up else (mx_dn >= d_near),
                   near_flip=0.5 * (mx_up >= d_near) + 0.5 * (mx_dn >= d_near),
                   far_hit=(mx_dn >= d_far) if near_up else (mx_up >= d_far),
                   far_flip=0.5 * (mx_up >= d_far) + 0.5 * (mx_dn >= d_far),
                   d_near=d_near)
        if d_near >= 2 * TICK:
            pts, win, risk = trade(r, near_up, d_near)
            rec.update(t_pts=pts, t_win=win, t_risk=risk)
    # sweep-only events: took P high but not P low (or vice versa) -> closes back inside?
    took_hi, took_lo = r.high > r.high_p, r.low < r.low_p
    if took_hi and not took_lo:
        rec["sweep"] = "swept prior high only"; rec["closed_inside"] = r.close <= r.high_p
    elif took_lo and not took_hi:
        rec["sweep"] = "swept prior low only"; rec["closed_inside"] = r.close >= r.low_p
    rows.append(rec)
ev = pd.DataFrame(rows)


def near_tbl(d):
    t = d.dropna(subset=["t_pts"])
    return pd.Series({"n": len(d), "near%": 100 * d.near_hit.astype(float).mean(),
                      "near_flip%": 100 * d.near_flip.mean(),
                      "far%": 100 * d.far_hit.astype(float).mean(), "far_flip%": 100 * d.far_flip.mean(),
                      "trade win%": 100 * t.t_win.astype(float).mean(), "net pts/trade": t.t_pts.mean(),
                      "med target pts": t.t_risk.median()})


sel = ev[ev.case.notna() & ev.hour.isin([8, 9, 10, 11, 12, 13, 14, 15])]
print("HOURLY: hour H opens in one half of hour H-1 (dev 2020 - Jul 2025, full sessions)\n")
print("near / far = touches the prior hour's near / far extreme during hour H.")
print("flip = same distances on a mirrored (random-walk) path.  trade = 1:1 toward the near extreme.\n")
print(sel.groupby(["hour", "case"]).apply(near_tbl, include_groups=False).round(1).to_string())

print("\nSame, pooled 09:00-12:00, by year:\n")
print(sel[sel.hour.isin([9, 10, 11, 12])].groupby(["year"]).apply(near_tbl, include_groups=False).round(1).to_string())

sw = ev[ev.sweep.notna() & ev.hour.isin([9, 10, 11, 12, 13, 14, 15])]
print("\nSWEEPS: hour H takes only one side of hour H-1 -> closes back inside H-1's range\n")
print(sw.groupby(["hour", "sweep"]).closed_inside.agg(n="size", pct=lambda s: 100 * s.astype(float).mean()).round(1).to_string())
