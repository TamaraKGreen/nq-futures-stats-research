"""
iFVG scalps, 09:30-11:00 ET, filtered by the hourly / overnight stats.

    py -m oos.ifvg

FVG / IFVG engine ported from FVG_IFVG_KeyLevels (10).pine:
  * gaps on the 5m and 15m bars (TF1 = 5, keep 12; TF2 = 15, keep 10)
  * bull gap: low(C) > high(A)  -> zone [high(A), low(C)]
    bear gap: high(C) < low(A)  -> zone [high(C), low(A)]
  * on each new HTF close, gaps formed on EARLIER bars are checked first:
      intact bull gap closed below its bottom -> inverts (bearish IFVG)
      intact bear gap closed above its top    -> inverts (bullish IFVG)
      an IFVG closed back through the other side is retired
    then a new gap on the bar that just closed is added; oldest dropped past N
  * displacement = how far the inverting close went past the edge / ATR14 (RMA)
    on that timeframe. >= 1.0 = "strong" (the script's default).

PRE-REGISTERED before running (the script has no entry rules, so these are
the two standard ways to trade an IFVG):
  signal   an inversion whose HTF bar closes between 09:30 and 10:40 ET
  entry A  at the inverting bar's close, in the IFVG's direction
  entry B  limit at the zone's near edge if price retests it within 30 min
           (before 10:40, while the zone has not been closed back through)
  stop     "zone" = far edge of the zone (trades with a zone stop < 4 or > 40
           today-equivalent pts are skipped)  |  "fixed 15" today-equiv pts
  target   20 today-equivalent pts; max hold 20 min; flat by 11:00
  cost     1 pt round trip; points scaled to today as in scalps.py
  filters  magnet: "with" = trade points at an untaken stat target, "against"
           = points away from one, "none" = no active target or conflicting.
             09:30-09:59  near 8:00-hour extreme (by where 9:00 opened)
             10:00-10:59  near 9:00-hour extreme (by where 10:00 opened)
             any time     ONH if 8am in the high third, ONL if low third
           displacement: strong (>= 1 ATR) / weak
  baseline random entries, same day, same direction, same stop and target,
           random minute 09:30-10:40, same fill rules; edge = setup - random,
           under both fill scorings, est net = -COST/S + mean edge
Data 2020-01 .. 2026-04 (order-flow file), shown 2020-Mar25 and unseen
separately.

AMENDED after the engine check, before any trade result: one HTF close often
inverts several stacked gaps at once, so signals are de-duplicated to one per
(inverting bar, direction), keeping the zone nearest to price.
"""
import numpy as np
import pandas as pd

from .data import period_of
from . import families as F
from . import scalps as SC
from .orderflow import load_flow, sim
from .tradability import Days, OUT

COST = 1.0
TF_KEEP = {5: 12, 15: 10}
T_PTS = 20
import os as _os
FILL_THRU = float(_os.environ.get("IFVG_FILL_THRU", "0"))   # 0.25 = must trade 1 tick through the edge


def htf_bars(b, tf):
    t = b["Datetime"].dt.tz_localize(None)
    k = t.dt.floor(f"{tf}min")
    g = b.assign(k=k).groupby("k", sort=True)
    h = g.agg(o=("Open", "first"), h=("High", "max"), l=("Low", "min"), c=("Close", "last"))
    h["end"] = h.index + pd.Timedelta(minutes=tf)
    tr = pd.concat([h["h"] - h["l"], (h["h"] - h["c"].shift()).abs(), (h["l"] - h["c"].shift()).abs()], axis=1).max(axis=1)
    h["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return h.reset_index()


def engine(h, keep):
    """Return inversion events and each zone's retirement time."""
    O, H, L, C, A = (h[c].to_numpy() for c in ("o", "h", "l", "c", "atr"))
    T = h["k"].to_numpy(); E = h["end"].to_numpy()
    stack = []          # dicts: top, bot, bull, inv, born, id
    events = []
    nid = 0
    for k in range(len(h)):
        cC = C[k]
        survivors = []
        for g in stack:
            if g["born"] < k:
                if not g["inv"]:
                    through = cC < g["bot"] if g["bull"] else cC > g["top"]
                    if through:
                        g["inv"] = True
                        disp = ((g["bot"] - cC) if g["bull"] else (cC - g["top"])) / A[k] if A[k] > 0 else np.nan
                        events.append({"id": g["id"], "t_inv": E[k], "dir": -1 if g["bull"] else 1,
                                       "top": g["top"], "bot": g["bot"], "close": cC, "disp": disp,
                                       "born": T[g["born"]]})
                else:
                    spent = cC > g["top"] if g["bull"] else cC < g["bot"]
                    if spent:
                        g["retired"] = E[k]
                        continue
            survivors.append(g)
        stack = survivors
        if k >= 2:
            if L[k] > H[k - 2]:
                stack.append({"top": L[k], "bot": H[k - 2], "bull": True, "inv": False, "born": k, "id": nid}); nid += 1
            elif H[k] < L[k - 2]:
                stack.append({"top": L[k - 2], "bot": H[k], "bull": False, "inv": False, "born": k, "id": nid}); nid += 1
        while len(stack) > keep:
            stack.pop(0)
    return pd.DataFrame(events)


def magnet_dir(D, day, L, i_sig, sm_sig):
    """+1 / -1 for each active untaken stat target at the signal bar."""
    s, e = D.span[day]
    sm = D.sm[s:e]
    at = lambda hhmm: s + np.searchsorted(sm, F._sm(hhmm))
    mags = []
    r = L.loc[day]

    def seg(a, z):
        if z < a:
            return -np.inf, np.inf          # nothing traded yet -> untaken
        return D.H[a:z + 1].max(), D.L[a:z + 1].min()
    # near 8:00 extreme, 09:30-09:59
    if sm_sig < F._sm("10:00") and np.isfinite(r["H8H"]):
        i9 = at("09:00")
        o9 = D.O[i9]
        tgt, d = (r["H8H"], 1) if o9 >= r["H8M"] else (r["H8L"], -1)
        seg_h, seg_l = seg(i9, i_sig)
        if (d > 0 and seg_h < tgt) or (d < 0 and seg_l > tgt):
            mags.append(d)
    # near 9:00 extreme, 10:00-10:59
    if sm_sig >= F._sm("10:00") and np.isfinite(r["H9H"]):
        i10 = at("10:00")
        o10 = D.O[i10]
        tgt, d = (r["H9H"], 1) if o10 >= r["H9M"] else (r["H9L"], -1)
        seg_h, seg_l = seg(i10, i_sig)
        if (d > 0 and seg_h < tgt) or (d < 0 and seg_l > tgt):
            mags.append(d)
    # overnight extreme by 8am third
    i930 = at("09:30")
    if r["ctx_8am"] == "8am high third" and seg(i930, i_sig)[0] < r["ONH"]:
        mags.append(1)
    if r["ctx_8am"] == "8am low third" and seg(i930, i_sig)[1] > r["ONL"]:
        mags.append(-1)
    return mags


def run():
    b = load_flow()
    L, ref = SC.levels_table(b)
    D = Days(b)
    rng = np.random.default_rng(11)
    t_lo, t_hi, end_sm = F._sm("09:30"), F._sm(SC.LAST_ENTRY), F._sm("10:59")
    rows = []
    for tf, keep in TF_KEEP.items():
        h = htf_bars(b, tf)
        ev = engine(h, keep)
        ev["t_inv"] = pd.to_datetime(ev["t_inv"])
        ev["day"] = (ev["t_inv"] + pd.Timedelta(hours=6)).dt.normalize()
        ev["sm"] = ((ev["t_inv"].dt.hour * 60 + ev["t_inv"].dt.minute - 1080) % 1440)
        ev = ev[(ev["sm"] >= t_lo) & (ev["sm"] <= t_hi) & ev["day"].isin(L.index)]
        # one close often inverts several stacked gaps: one trade per bar and
        # direction, using the zone nearest to price (the one retested first)
        ev["dist"] = np.where(ev["dir"] > 0, ev["close"] - ev["top"], ev["bot"] - ev["close"])
        ev = ev.sort_values("dist").drop_duplicates(["t_inv", "dir"]).sort_values("t_inv")
        print(f"{tf}m: {len(ev):,} inversions in the window")
        for _, x in ev.iterrows():
            day = x["day"]
            if day not in D.span:
                continue
            scale = L.at[day, "scale"]
            if not np.isfinite(scale):
                continue
            s, e = D.span[day]
            sm = D.sm[s:e]
            day_end = s + np.searchsorted(sm, end_sm, side="right") - 1
            i_sig = s + np.searchsorted(sm, x["sm"])          # first 1m bar after the HTF close
            if i_sig > day_end:
                continue
            dirn = int(x["dir"])
            mags = magnet_dir(D, day, L, i_sig - 1, x["sm"])
            mag = "with" if (dirn in mags and -dirn not in mags) else (
                  "against" if (-dirn in mags and dirn not in mags) else "none")
            strong = "strong" if x["disp"] >= 1.0 else "weak"
            near, far = (x["top"], x["bot"]) if dirn > 0 else (x["bot"], x["top"])
            entries = {"A inversion close": (i_sig, x["close"], False)}
            # B: retest of the near edge within 30 min, zone not closed back through
            j_end = min(i_sig + 30, s + np.searchsorted(sm, t_hi, side="right") - 1)
            for j in range(i_sig, j_end + 1):
                # a touch fills FIRST - a bar that touches and then closes back
                # through the zone is a filled (usually losing) trade, not a skip
                if (dirn > 0 and D.L[j] <= near - FILL_THRU) or (dirn < 0 and D.H[j] >= near + FILL_THRU):
                    entries["B retest"] = (j, near, True)
                    break
                if (dirn > 0 and D.C[j] < far) or (dirn < 0 and D.C[j] > far):
                    break
            for ename, (ie, px, amb) in entries.items():
                zone_s = abs(px - far) / scale
                for sname, S in (("zone", zone_s), ("fixed 15", 15.0)):
                    if sname == "zone" and not (4 <= zone_s <= 40):
                        continue
                    res = sim(D, ie, px, dirn, T_PTS, S, scale, day_end, amb)
                    if res is None:
                        continue
                    # matched random entry: same day, same direction / stop / target
                    i0 = s + np.searchsorted(sm, t_lo)
                    i1 = s + np.searchsorted(sm, t_hi, side="right") - 1
                    ir = int(rng.integers(i0, i1 + 1))
                    rr = sim(D, ir, D.O[ir], dirn, T_PTS, S, scale, day_end, amb)
                    rows.append({"day": day, "tf": f"{tf}m", "entry": ename, "stop": sname, "S": S,
                                 "magnet": mag, "disp": strong, "disp_atr": x["disp"],
                                 "cons": res[0], "opt": res[1],
                                 "r_cons": rr[0] if rr else np.nan, "r_opt": rr[1] if rr else np.nan})
    t = pd.DataFrame(rows)
    t["period"] = np.where(period_of(t["day"]) == "holdout", "unseen", "2020-Mar25")
    t.to_parquet(f"{OUT}/ifvg_trades.parquet")
    return t


def summarise(t):
    def one(g):
        S = g["S"]
        c, o = g["cons"] / S, g["opt"] / S
        rc, ro = g["r_cons"] / S, g["r_opt"] / S
        ec, eo = (c - rc).mean(), (o - ro).mean()
        sec = (c - rc).std(ddof=1) / np.sqrt(len(g)); seo = (o - ro).std(ddof=1) / np.sqrt(len(g))
        est = (-COST / S).mean() + (ec + eo) / 2
        return pd.Series({"n": len(g), "per_month": round(len(g) / max(g["day"].dt.to_period("M").nunique(), 1), 1),
                          "win_cons_%": round(100 * (g["cons"] > 0).mean(), 1),
                          "win_opt_%": round(100 * (g["opt"] > 0).mean(), 1),
                          "rand_win_cons_%": round(100 * (g["r_cons"] > 0).mean(), 1),
                          "edge_R": round((ec + eo) / 2, 3), "t_cons": round(ec / sec, 2), "t_opt": round(eo / seo, 2),
                          "est_net_R": round(est, 3), "est_net_pts": round(est * S.mean(), 2)})
    keys = ["tf", "entry", "stop"]
    out = []
    for label, sub in [("all", t), ("magnet: with", t[t["magnet"] == "with"]),
                       ("magnet: against", t[t["magnet"] == "against"]), ("magnet: none", t[t["magnet"] == "none"]),
                       ("disp strong", t[t["disp"] == "strong"]), ("disp weak", t[t["disp"] == "weak"]),
                       ("with + strong", t[(t["magnet"] == "with") & (t["disp"] == "strong")])]:
        for per in ("2020-Mar25", "unseen"):
            g = sub[sub["period"] == per].groupby(keys).apply(one, include_groups=False).reset_index()
            g.insert(0, "period", per); g.insert(0, "filter", label)
            out.append(g)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    import os, sys
    f = f"{OUT}/ifvg_trades.parquet"
    t = pd.read_parquet(f) if os.path.exists(f) and "--rerun" not in sys.argv else run()
    s = summarise(t)
    s.to_csv(f"{OUT}/ifvg_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(s.to_string(index=False))
