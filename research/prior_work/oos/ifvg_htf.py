"""
1-minute iFVG inside a 15-minute FVG, 09:30-11:00 ET.

    py -m oos.ifvg_htf

PRE-REGISTERED 20 Nov 2026, before running. Same FVG/iFVG engine as
oos/ifvg.py (ported from FVG_IFVG_KeyLevels), run on 1m and 15m bars.

  15m FVG "active" = formed (its bar C closed) and still INTACT: not yet closed
      through (inverted) and not yet pushed out of the 10-gap stack.
  1m signal = a 1m FVG inverts (a 1m bar closes through it) between 09:30 and
      10:40. One signal per (bar, direction), zone nearest to price.
  context of each 1m signal
      "inside, aligned"  - the 1m iFVG zone overlaps an active 15m FVG of the
                           SAME direction as the trade (e.g. price dips into a
                           bullish 15m FVG, a 1m bearish FVG inverts -> long)
      "inside, opposite" - overlaps an active 15m FVG of the other direction
      "not inside"       - no overlap with any active 15m FVG
  entries   A at the inverting 1m bar's close
            B limit at the 1m iFVG's near edge, retest within 15 min (before
              10:40, zone not closed back through)
  stops     "1m zone"  far edge of the 1m iFVG
            "15m zone" far edge of the 15m FVG (inside-aligned signals only)
            "fixed 15"
            zone stops < 4 or > 40 today-equivalent pts are skipped
  target 20 today-equivalent pts; max hold 20 min; flat by 11:00; 1 pt cost.
  baseline random entries, same day / direction / stop / target / fill rules.
Data 2020-01 .. 2026-04, reported 2020-Mar25 and unseen separately.
"""
import numpy as np
import pandas as pd

from .data import period_of
from . import families as F
from . import scalps as SC
from .orderflow import load_flow, sim
from .tradability import Days, OUT
from .ifvg import htf_bars

COST, T_PTS = 1.0, 20
RETEST_MIN = 15


def engine_full(h, keep):
    """All gaps with their lifecycle, plus inversion events."""
    H, L, C, A = (h[c].to_numpy() for c in ("h", "l", "c", "atr"))
    E = h["end"].to_numpy()
    stack, gaps, events = [], [], []
    for k in range(len(h)):
        cC = C[k]
        surv = []
        for g in stack:
            if g["born"] < k:
                if not g["inv"]:
                    if (cC < g["bot"]) if g["bull"] else (cC > g["top"]):
                        g["inv"] = True
                        g["t_inv"] = E[k]
                        events.append({"t_inv": E[k], "dir": -1 if g["bull"] else 1, "top": g["top"],
                                       "bot": g["bot"], "close": cC,
                                       "disp": ((g["bot"] - cC) if g["bull"] else (cC - g["top"])) / A[k] if A[k] > 0 else np.nan})
                else:
                    if (cC > g["top"]) if g["bull"] else (cC < g["bot"]):
                        g["t_end"] = E[k]
                        continue
            surv.append(g)
        stack = surv
        if k >= 2:
            g = None
            if L[k] > H[k - 2]:
                g = {"top": L[k], "bot": H[k - 2], "bull": True}
            elif H[k] < L[k - 2]:
                g = {"top": L[k - 2], "bot": H[k], "bull": False}
            if g:
                g.update(inv=False, born=k, t_born=E[k], t_inv=None, t_end=None)
                stack.append(g); gaps.append(g)
        while len(stack) > keep:
            g0 = stack.pop(0)
            if g0["t_end"] is None:
                g0["t_end"] = E[k]
    G = pd.DataFrame([{k: g[k] for k in ("top", "bot", "bull", "t_born", "t_inv", "t_end")} for g in gaps])
    return G, pd.DataFrame(events)


def run():
    b = load_flow()
    L, ref = SC.levels_table(b)
    D = Days(b)
    rng = np.random.default_rng(23)
    G15, _ = engine_full(htf_bars(b, 15), 10)
    # intact interval of each 15m FVG: [t_born, first of t_inv / t_end)
    G15["t_off"] = G15["t_inv"].where(G15["t_inv"].notna(), G15["t_end"])
    G15["t_off"] = pd.to_datetime(G15["t_off"]).fillna(pd.Timestamp("2100-01-01"))
    G15["t_born"] = pd.to_datetime(G15["t_born"])
    G15 = G15.sort_values("t_born").reset_index(drop=True)
    born = G15["t_born"].to_numpy()

    _, ev = engine_full(htf_bars(b, 1), 12)
    ev["t_inv"] = pd.to_datetime(ev["t_inv"])
    ev["day"] = (ev["t_inv"] + pd.Timedelta(hours=6)).dt.normalize()
    ev["sm"] = (ev["t_inv"].dt.hour * 60 + ev["t_inv"].dt.minute - 1080) % 1440
    t_lo, t_hi, end_sm = F._sm("09:30"), F._sm(SC.LAST_ENTRY), F._sm("10:59")
    ev = ev[(ev["sm"] >= t_lo) & (ev["sm"] <= t_hi) & ev["day"].isin(L.index)].copy()
    ev["dist"] = np.where(ev["dir"] > 0, ev["close"] - ev["top"], ev["bot"] - ev["close"])
    ev = ev.sort_values("dist").drop_duplicates(["t_inv", "dir"]).sort_values("t_inv")
    print(f"1m inversions in window: {len(ev):,}; 15m FVGs: {len(G15):,}")

    rows = []
    for _, x in ev.iterrows():
        day = x["day"]
        if day not in D.span or not np.isfinite(L.at[day, "scale"]):
            continue
        scale = L.at[day, "scale"]
        dirn = int(x["dir"])
        # active 15m gaps at the signal time that overlap the 1m zone
        hi = np.searchsorted(born, np.datetime64(x["t_inv"]), side="right")
        cand = G15.iloc[max(0, hi - 60):hi]
        act = cand[(cand["t_off"] > x["t_inv"]) & (cand["bot"] <= x["top"]) & (cand["top"] >= x["bot"])]
        al = act[act["bull"] == (dirn > 0)]
        ctx = "inside, aligned" if len(al) else ("inside, opposite" if len(act) else "not inside")
        z15_far = (al["bot"].min() if dirn > 0 else al["top"].max()) if len(al) else np.nan

        s, e = D.span[day]
        sm = D.sm[s:e]
        day_end = s + np.searchsorted(sm, end_sm, side="right") - 1
        i_sig = s + np.searchsorted(sm, x["sm"])
        if i_sig > day_end:
            continue
        near, far = (x["top"], x["bot"]) if dirn > 0 else (x["bot"], x["top"])
        entries = {"A inversion close": (i_sig, x["close"], False)}
        j_end = min(i_sig + RETEST_MIN, s + np.searchsorted(sm, t_hi, side="right") - 1)
        for j in range(i_sig, j_end + 1):
            # a touch fills FIRST (see oos/ifvg.py)
            if (dirn > 0 and D.L[j] <= near) or (dirn < 0 and D.H[j] >= near):
                entries["B retest"] = (j, near, True)
                break
            if (dirn > 0 and D.C[j] < far) or (dirn < 0 and D.C[j] > far):
                break
        i0 = s + np.searchsorted(sm, t_lo)
        i1 = s + np.searchsorted(sm, t_hi, side="right") - 1
        for ename, (ie, px, amb) in entries.items():
            stops = {"1m zone": abs(px - far) / scale, "fixed 15": 15.0}
            if np.isfinite(z15_far):
                stops["15m zone"] = abs(px - z15_far) / scale
            for sname, S in stops.items():
                if sname != "fixed 15" and not (4 <= S <= 40):
                    continue
                res = sim(D, ie, px, dirn, T_PTS, S, scale, day_end, amb)
                if res is None:
                    continue
                ir = int(rng.integers(i0, i1 + 1))
                rr = sim(D, ir, D.O[ir], dirn, T_PTS, S, scale, day_end, amb)
                rows.append({"day": day, "t_inv": x["t_inv"], "dirn": dirn, "ztop": x["top"], "zbot": x["bot"],
                             "scale": scale, "ctx": ctx, "entry": ename, "stop": sname, "S": S,
                             "disp_atr": x["disp"], "cons": res[0], "opt": res[1],
                             "r_cons": rr[0] if rr else np.nan, "r_opt": rr[1] if rr else np.nan})
    t = pd.DataFrame(rows)
    t["period"] = np.where(period_of(t["day"]) == "holdout", "unseen", "2020-Mar25")
    t.to_parquet(f"{OUT}/ifvg_htf_trades.parquet")
    return t


def summarise(t):
    def one(g):
        S = g["S"]
        c, o, rc, ro = g["cons"] / S, g["opt"] / S, g["r_cons"] / S, g["r_opt"] / S
        dc, do = (c - rc), (o - ro)
        est = (-COST / S).mean() + (dc.mean() + do.mean()) / 2
        return pd.Series({"n": len(g), "per_month": round(len(g) / max(g["day"].dt.to_period("M").nunique(), 1), 1),
                          "win_cons_%": round(100 * (g["cons"] > 0).mean(), 1),
                          "win_opt_%": round(100 * (g["opt"] > 0).mean(), 1),
                          "rand_win_%": round(100 * (g["r_cons"] > 0).mean(), 1),
                          "edge_R": round((dc.mean() + do.mean()) / 2, 3),
                          "t_cons": round(dc.mean() / (dc.std(ddof=1) / np.sqrt(len(g))), 2),
                          "t_opt": round(do.mean() / (do.std(ddof=1) / np.sqrt(len(g))), 2),
                          "est_net_pts": round(est * S.mean(), 2)})
    return t.groupby(["ctx", "entry", "stop", "period"]).apply(one, include_groups=False).reset_index()


if __name__ == "__main__":
    import os, sys
    f = f"{OUT}/ifvg_htf_trades.parquet"
    t = pd.read_parquet(f) if os.path.exists(f) and "--rerun" not in sys.argv else run()
    s = summarise(t)
    s.to_csv(f"{OUT}/ifvg_htf_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(s.to_string(index=False))
