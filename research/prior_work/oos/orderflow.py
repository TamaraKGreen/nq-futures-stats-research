"""
Order flow at the level touch, 09:30-11:00 scalps.

    py -m oos.orderflow

PRE-REGISTERED 19 Nov 2026, before running.

Same levels, windows, first-touch rule, max hold (20 min), today-equivalent
scaling and 1 pt cost as oos/scalps.py. Prices AND order flow both come from
nq_combined_1min_bars_2020_2026.parquet (buy/sell aggressor volume per minute).

Two entry styles
  E1 at the level   - FADE: limit at the level; BREAK: stop 2 ticks through.
                      Only order flow from BEFORE the touch bar may be used.
  E2 confirmation   - wait for the touch bar to close, enter at its close
                      (FADE = against the approach, BREAK = with it). The touch
                      bar's own order flow may be used.

Order-flow features (signed so that + = aggression IN the approach direction,
e.g. net buying while price rises into a level from below)
  app_delta    sum(delta)/sum(volume), 5 bars before the touch          E1, E2
  vol_surge    volume of those 5 bars / mean volume of the 30 before    E1, E2
  size_surge   avg trade size (vol/trades) of those 5 bars vs the 30    E1, E2
               before (a proxy for big lots - your BigTrades idea)
  touch_delta  touch bar delta/volume                                   E2
  size_touch   touch bar avg trade size vs the 30 bars before           E2
  rejected     touch bar closes back on the approach side (yes/no)      E2
  Continuous features are cut in terciles (edges from discovery touches).

Exits: (T, S) in {(10,10), (20,15), (20,20)} today-equivalent points.

Scoring: every trade scored conservative and optimistic, as in scalps.py.
  baseline   = PLACEBO-level touches, all feature bins pooled, same entry,
               mode, T, S, period (pure noise with the same fill bias)
  est_net_R  = -COST/S + mean over both scorings of (cell R - baseline R)
  Cells: entry x mode x {real levels pooled, placebo levels pooled} x feature x
  bin x (T,S). Order flow is allowed to work at ANY price, so placebo-level
  cells are tested too - that answers "is it the level or the flow?".
Selection: discovery 2020-01-01..2025-03-14, est_net_R > 0 and t > 2 under
both scorings. Only those cells are scored on the holdout (2025-03-17 on).
"""
import numpy as np
import pandas as pd

from .data import period_of, DATA, NY
from . import families as F
from . import scalps as SC
from .tradability import Days, OUT

COST = SC.COST
TS = [(10, 10), (20, 15), (20, 20)]
REAL = [k for k in SC.WINDOWS if not k.startswith("P")]


def load_flow():
    d = pd.read_parquet(DATA / "nq_combined_1min_bars_2020_2026.parquet")
    d.index = d.index.tz_convert(NY)
    b = pd.DataFrame({"Datetime": d.index, "Open": d["open"].to_numpy("float64"),
                      "High": d["high"].to_numpy("float64"), "Low": d["low"].to_numpy("float64"),
                      "Close": d["close"].to_numpy("float64"), "Volume": d["volume"].to_numpy("float64"),
                      "Delta": d["delta"].to_numpy("float64"), "Trades": d["trades"].to_numpy("float64")})
    b = b[~b["Datetime"].duplicated()].sort_values("Datetime").reset_index(drop=True)
    return F.prep(b)


def sim(D, i, entry, dirn, T, S, scale, day_end, entry_bar_ambiguous):
    """(cons_net, opt_net) in today points. If not entry_bar_ambiguous, bar i is
    entirely after the entry (confirmation entries start at i = touch + 1)."""
    Tr, Sr = T * scale, S * scale
    tgt, stp = entry + dirn * Tr, entry - dirn * Sr
    j = min(i + SC.MAX_HOLD, day_end)
    if i > j:
        return None
    H, L, C = D.H[i:j + 1], D.L[i:j + 1], D.C[i:j + 1]
    st, tg = (L <= stp, H >= tgt) if dirn > 0 else (H >= stp, L <= tgt)
    out = []
    for opt in (False, True):
        s2, t2 = st.copy(), tg.copy()
        if entry_bar_ambiguous:
            t2[0] = False
            if opt:
                s2[0] = False
        a = np.argmax(s2) if s2.any() else 10**9
        b_ = np.argmax(t2) if t2.any() else 10**9
        if a == b_ == 10**9:
            g = dirn * (C[-1] - entry)
        elif (a < b_) or (a == b_ and not opt):
            g = -Sr
        else:
            g = Tr
        out.append(g / scale - COST)
    return out


def features(V, DL, TR, C, i, side, lvl):
    a, z = i - 5, i
    p0, p1 = i - 35, i - 5
    if p0 < 0:
        return None
    v5, v30 = V[a:z].sum(), V[p0:p1].mean()
    sz = lambda s, e: V[s:e].sum() / max(TR[s:e].sum(), 1)
    f = {"app_delta": side * DL[a:z].sum() / max(v5, 1),
         "vol_surge": (v5 / 5) / max(v30, 1),
         "size_surge": sz(a, z) / max(sz(p0, p1), 1e-9),
         "touch_delta": side * DL[i] / max(V[i], 1),
         "size_touch": (V[i] / max(TR[i], 1)) / max(sz(p0, p1), 1e-9),
         "rejected": "rejected" if side * (C[i] - lvl) < 0 else "closed through"}
    return f


def run_trades(b, L):
    D = Days(b)
    V, DL, TR = b["Volume"].to_numpy(), b["Delta"].to_numpy(), b["Trades"].to_numpy()
    rows = []
    end_sm = F._sm("10:59")
    for day, r in L.iterrows():
        if day not in D.span or not np.isfinite(r["scale"]):
            continue
        s, e = D.span[day]
        day_end = s + np.searchsorted(D.sm[s:e], end_sm, side="right") - 1
        for lv, (a, z) in SC.WINDOWS.items():
            lvl = r[lv]
            if not np.isfinite(lvl):
                continue
            ft = SC.first_touch(D, day, lvl, a, z)
            if ft is None:
                continue
            i, side = ft
            f = features(V, DL, TR, D.C, i, side, lvl)
            if f is None:
                continue
            grp = "placebo" if lv.startswith("P") else "real"
            for mode in ("fade", "break"):
                dirn = -side if mode == "fade" else side
                # E1 at the level
                if mode == "fade":
                    e1 = (i, lvl)
                else:
                    ent = lvl + side * 0.5
                    k = np.flatnonzero((D.H[i:i + 6] >= ent) if side > 0 else (D.L[i:i + 6] <= ent))
                    e1 = (i + int(k[0]), ent) if len(k) and i + k[0] <= day_end else None
                # E2 confirmation at the touch bar close
                e2 = (i + 1, D.C[i]) if i + 1 <= day_end else None
                for style, ee, amb in (("E1 at level", e1, True), ("E2 confirm", e2, False)):
                    if ee is None:
                        continue
                    for T, S in TS:
                        res = sim(D, ee[0], ee[1], dirn, T, S, r["scale"], day_end, amb)
                        if res is None:
                            continue
                        rows.append({"day": day, "level": lv, "grp": grp, "style": style, "mode": mode,
                                     "T": T, "S": S, "cons": res[0], "opt": res[1], **f})
    t = pd.DataFrame(rows)
    t["period"] = period_of(t["day"])
    return t


CONT = ["app_delta", "vol_surge", "size_surge", "touch_delta", "size_touch"]
E1_FEATS = ["app_delta", "vol_surge", "size_surge"]
E2_FEATS = E1_FEATS + ["touch_delta", "size_touch", "rejected"]


def add_bins(t):
    disc = t[t["period"] == "discovery"]
    for c in CONT:
        # edges per entry style and mode would leak nothing; one set per feature
        base = disc.drop_duplicates(["day", "level"])[c]
        e = np.nanquantile(base, [1/3, 2/3])
        t[c + "_b"] = pd.cut(t[c], [-np.inf, *e, np.inf], labels=["low", "mid", "high"]).astype(str)
    t["rejected_b"] = t["rejected"]
    return t


def score(t, min_n=60):
    rows = []
    keys = ["style", "mode", "T", "S"]
    for k, g in t.groupby(keys):
        style, mode, T, S = k
        base = g[g["grp"] == "placebo"]
        bc, bo = (base["cons"] / S), (base["opt"] / S)
        feats = E1_FEATS if style.startswith("E1") else E2_FEATS
        for grp in ("real", "placebo"):
            gg = g[g["grp"] == grp]
            for f in feats:
                for binv, c in gg.groupby(f + "_b"):
                    n = len(c)
                    if n < min_n:
                        continue
                    rc, ro = c["cons"] / S, c["opt"] / S
                    ec, eo = rc.mean() - bc.mean(), ro.mean() - bo.mean()
                    tc = ec / np.hypot(rc.std(ddof=1) / np.sqrt(n), bc.std(ddof=1) / np.sqrt(len(bc)))
                    to = eo / np.hypot(ro.std(ddof=1) / np.sqrt(n), bo.std(ddof=1) / np.sqrt(len(bo)))
                    est = -COST / S + (ec + eo) / 2
                    rows.append({"style": style, "mode": mode, "T": T, "S": S, "levels": grp, "feature": f,
                                 "bin": binv, "n": n,
                                 "per_month": round(n / max(c["day"].dt.to_period("M").nunique(), 1), 1),
                                 "win_cons_%": round(100 * (c["cons"] > 0).mean(), 1),
                                 "win_opt_%": round(100 * (c["opt"] > 0).mean(), 1),
                                 "edge_R_cons": round(ec, 3), "edge_R_opt": round(eo, 3),
                                 "t_cons": round(tc, 2), "t_opt": round(to, 2),
                                 "est_net_R": round(est, 3), "est_net_pts": round(est * S, 2)})
    return pd.DataFrame(rows)


def run():
    b = load_flow()
    L, ref = SC.levels_table(b)
    print(f"ATR14 ref {ref:.0f}; days {len(L)}")
    import os, sys
    f = f"{OUT}/orderflow_trades.parquet"
    if os.path.exists(f) and "--rerun" not in sys.argv:
        t = pd.read_parquet(f)
    else:
        t = run_trades(b, L)
        t.to_parquet(f)
    t = add_bins(t)
    disc = score(t[t["period"] == "discovery"])
    disc.to_csv(f"{OUT}/orderflow_discovery.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400)
    print(f"\n=== discovery: {len(disc)} cells; best 40 by est_net_R ===")
    print(disc.sort_values("est_net_R", ascending=False).head(40).to_string(index=False))
    keys = ["style", "mode", "T", "S", "levels", "feature", "bin"]
    keep = disc[(disc["est_net_R"] > 0) & (disc["t_cons"] > 2) & (disc["t_opt"] > 2)]
    print(f"\n{len(keep)} cells pass the discovery bar")
    if len(keep):
        hold = score(t[t["period"] == "holdout"], min_n=10).merge(keep[keys], on=keys, how="right")
        hold.to_csv(f"{OUT}/orderflow_holdout.csv", index=False)
        print("\n=== holdout (unseen) for passing cells ===")
        print(hold.to_string(index=False))


if __name__ == "__main__":
    run()
