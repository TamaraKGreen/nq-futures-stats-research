"""
The iFVG candidate (1m iFVG inside an ALIGNED 15m FVG, NEAR a key level, B retest entry) with KEY-LEVEL
targets instead of the fixed 20 pts.

Entries, stops, 'near key level', costs, scoring and the random baseline follow oos/ifvg_htf.py and
oos/ifvg_levels.py unchanged (entry window 09:30-10:40). Only the exit changes:
  targets   FIX20   20 today-equivalent pts (the original, for reference)
            K1R     nearest key level in the trade direction at least 1.0 x risk from the entry
            K05R    nearest key level at least 0.5 x risk away
            K2ND    second key level in the trade direction at least 0.5 x risk away
            Key levels = research/stats/keylevels.py set known at the signal time (PDH/PDL/PDC, ONH/ONL, Asia,
            London, 8am H/L/M, DO, WO, prior RTH/ETH VAH/VAL/POC, Statmap 4H 06-10; IB / SM 10-14 after 10:30).
            No qualifying level -> no trade for that target.
  holding   H11 = flat by 11:00 (original window)      H16 = flat by 16:00
            (the 20-minute max hold of the original applies only to FIX20)
  baseline  random entry minute 09:30-10:40 same day, same direction, same stop distance and SAME TARGET
            DISTANCE, same fill rules. edge = setup R - random R; est net = -COST/S + mean edge (both scorings).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from oos import families as F
from oos import scalps as SC
from oos.orderflow import load_flow
from oos.tradability import Days, OUT
from oos.data import fresh_of
from research.stats.keylevels import bars as kl_bars, level_table, AVAIL

COST = 1.0


def sim_px(D, i, entry, dirn, tgt, stp, j_end, amb):
    """(cons, opt) in POINTS (not scaled), exits by bar j_end."""
    if i > j_end:
        return None
    H, L, C = D.H[i:j_end + 1], D.L[i:j_end + 1], D.C[i:j_end + 1]
    st, tg = (L <= stp, H >= tgt) if dirn > 0 else (H >= stp, L <= tgt)
    out = []
    for opt in (False, True):
        s2, t2 = st.copy(), tg.copy()
        if amb:
            t2[0] = False
            if opt:
                s2[0] = False
        a = np.argmax(s2) if s2.any() else 10**9
        b = np.argmax(t2) if t2.any() else 10**9
        if a == b == 10**9:
            g = dirn * (C[-1] - entry)
        elif (a < b) or (a == b and not opt):
            g = dirn * (stp - entry)
        else:
            g = dirn * (tgt - entry)
        out.append(g - COST)
    return out


def run():
    t = pd.read_parquet(f"{OUT}/ifvg_levels_trades_fresh.parquet")
    t = t[(t.ctx == "inside, aligned") & (t.entry == "B retest") & (t.near_level == "near key level") &
          t["stop"].isin(["15m zone", "fixed 15"])].copy()
    b = load_flow()
    L_, _ = SC.levels_table(b)
    D = Days(b)
    # key levels in RAW prices of each day's contract
    kb = kl_bars()
    KL, _ = level_table(kb)
    adj = kb.groupby("day").roll_adj.first()
    KLraw = KL[list(AVAIL)].sub(adj.reindex(KL.index), axis=0)
    rng = np.random.default_rng(31)
    t_lo, t_hi = F._sm("09:30"), F._sm(SC.LAST_ENTRY)
    rows = []
    for _, x in t.iterrows():
        day = x["day"]
        if day not in D.span or day not in KLraw.index:
            continue
        s, e = D.span[day]; sm = D.sm[s:e]
        dirn = int(x["dirn"]); scale = x["scale"]
        sig_sm = (x["t_inv"].hour * 60 + x["t_inv"].minute - 1080) % 1440
        near, far = (x["ztop"], x["zbot"]) if dirn > 0 else (x["zbot"], x["ztop"])
        # rebuild the B-retest fill exactly as ifvg_htf.py
        i_sig = s + np.searchsorted(sm, sig_sm)
        j_end = min(i_sig + 15, s + np.searchsorted(sm, t_hi, side="right") - 1)
        ie = None
        for j in range(i_sig, j_end + 1):
            if (dirn > 0 and D.L[j] <= near) or (dirn < 0 and D.H[j] >= near):
                ie = j; break
            if (dirn > 0 and D.C[j] < far) or (dirn < 0 and D.C[j] > far):
                break
        if ie is None:
            continue
        px = near
        S_raw = x["S"] * scale                                 # stop distance in raw points
        stp = px - dirn * S_raw
        cur_sm = int(D.sm[ie])
        lv = KLraw.loc[day]
        cands = sorted([v for k, v in lv.items() if np.isfinite(v) and (F._sm(f"{AVAIL[k] // 60:02d}:{AVAIL[k] % 60:02d}") <= cur_sm)
                        and dirn * (v - px) > 0], key=lambda v: dirn * (v - px))
        tgts = {"FIX20": px + dirn * 20 * scale}
        r1 = [v for v in cands if dirn * (v - px) >= 1.0 * S_raw]
        r05 = [v for v in cands if dirn * (v - px) >= 0.5 * S_raw]
        if r1: tgts["K1R"] = r1[0]
        if r05: tgts["K05R"] = r05[0]
        if len(r05) > 1: tgts["K2ND"] = r05[1]
        ends = {"H11": s + np.searchsorted(sm, F._sm("10:59"), side="right") - 1,
                "H16": s + np.searchsorted(sm, F._sm("15:59"), side="right") - 1}
        i0 = s + np.searchsorted(sm, t_lo); i1 = s + np.searchsorted(sm, t_hi, side="right") - 1
        ir = int(rng.integers(i0, i1 + 1))
        for tn, tv in tgts.items():
            for hn, je in ends.items():
                if tn == "FIX20" and hn == "H16":
                    continue
                jj = min(ie + SC.MAX_HOLD, je) if tn == "FIX20" else je
                res = sim_px(D, ie, px, dirn, tv, stp, jj, True)
                if res is None:
                    continue
                rd = abs(tv - px)
                jr = min(ir + SC.MAX_HOLD, je) if tn == "FIX20" else je
                rr = sim_px(D, ir, D.O[ir], dirn, D.O[ir] + dirn * rd, D.O[ir] - dirn * S_raw, jr, True)
                rows.append({"day": day, "stop": x["stop"], "target": tn, "hold": hn, "S": S_raw, "rr": rd / S_raw,
                             "cons": res[0], "opt": res[1], "r_cons": rr[0] if rr else np.nan, "r_opt": rr[1] if rr else np.nan,
                             "scale": scale})
    T = pd.DataFrame(rows)
    T["period"] = pd.Series(fresh_of(T.day)).map({"discovery": "1 2020-Mar25", "holdout": "2 Mar25-Apr26", "fresh": "3 May-Nov26"}).to_numpy()
    return T


def summarise(T):
    def one(g):
        S = g["S"]
        c, o, rc, ro = g.cons / S, g.opt / S, g.r_cons / S, g.r_opt / S
        dc, do = c - rc, o - ro
        return pd.Series({"n": len(g), "med R:R": g.rr.median(), "win cons %": 100 * (g.cons > 0).mean(),
                          "win opt %": 100 * (g.opt > 0).mean(), "R cons": c.mean(), "R opt": o.mean(),
                          "rand R (avg)": ((rc + ro) / 2).mean(), "edge R": (dc.mean() + do.mean()) / 2,
                          "t cons": dc.mean() / (dc.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 else np.nan,
                          "t opt": do.mean() / (do.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 else np.nan,
                          "net pts/trade (avg of bounds)": ((g.cons + g.opt) / 2 / g.scale).mean()})
    a = T.groupby(["stop", "target", "hold", "period"]).apply(one, include_groups=False)
    b = T.groupby(["stop", "target", "hold"]).apply(one, include_groups=False)
    return a, b


if __name__ == "__main__":
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
    T = run()
    T.to_parquet(f"{OUT}/ifvg_targets_trades.parquet")
    a, b = summarise(T)
    print("iFVG CANDIDATE WITH KEY-LEVEL TARGETS (R = net of 1 pt cost, per unit of stop distance)\n")
    print("ALL PERIODS POOLED:\n"); print(b.round(3).to_string())
    print("\nBY PERIOD:\n"); print(a.round(3).to_string())
