"""
Level-reaction scalps, 09:30-11:00 ET.

    py -m oos.scalps

PRE-REGISTERED 19 Nov 2026, before running:

Question: at the FIRST touch of a known level inside the window, does a
fixed-point scalp work better in one direction than the other?

  levels (available from)
    ONH / ONL         overnight 18:00-09:29 high / low         09:30
    PDH / PDL         prior RTH high / low                     09:30
    PDC               prior RTH close (gap-fill level)          09:30
    M69               06:00-09:00 range mid                     09:30
    H8H / H8L / H8M   08:00 hour high / low / mid               09:30-09:59 only
    H9H / H9L / H9M   09:00 hour high / low / mid               10:00-10:40
    ORH / ORL         09:30-09:45 high / low                    09:45-10:40
  (IB levels only exist from 10:30, so they are outside a 9:30-11:00 window.)

  entry   first touch of the level between its start time and 10:40.
          FADE  = limit at the level, against the approach.
          BREAK = stop 2 ticks through the level, with the approach.
  exits   target T in {10, 20}, stop S in {10, 15, 20} points; max hold 20 min
          (flat by 11:00). Cost 1 point per round trip.
  scaling all point values are TODAY-EQUIVALENT: on each day they are scaled by
          ATR14_day / ATR14_ref, ATR14_ref = median ATR14 over the holdout
          (Mar 2025 - Apr 2026). 20 pts today = ~10 pts in early 2020.
  data    discovery 2020-01-01 .. 2025-03-14 (earlier years make 20 pts
          today-equivalent smaller than the 1-minute bars can resolve);
          holdout 2025-03-17 .. 2026-04-29.
  fills   1-minute bars cannot tell whether target or stop came first inside
          one bar, so every trade is scored twice:
            conservative - the stop wins any ambiguous bar, incl. the entry bar
            optimistic   - the target wins ambiguous bars after the entry bar
          The truth is in between. A setup only counts if the CONSERVATIVE
          net result is positive.
  contexts (second pass, only for levels that look interesting):
          ORB width tercile, 8am third of the overnight range, gap size.

AMENDED before any real-level result was seen: a random-entry check showed the
two fill bounds are ~13 pp apart on 10/10 scalps (38% vs 51.5% wins, true
~50%), so "conservative net > 0" is unreachable by construction. Instead:

  placebo levels  4 per day per entry-window start (09:30 / 09:45 / 10:00),
                  uniform within +-0.15 ATR of the price just before that start,
                  run through the identical touch/fill code. They carry the same
                  fill bias but no information.
  edge            setup R minus placebo R (same mode, T, S, period), computed
                  under BOTH scorings - the fill bias cancels.
  estimated net   -COST/S + mean(edge_cons, edge_opt)   (placebo gross ~ 0)

Selection: on discovery, keep (level, mode, T, S) with estimated net R > 0 and
edge t > 2 under both scorings. Only those are scored on the holdout.
"""
import numpy as np
import pandas as pd

from .data import load_nq, period_of
from . import families as F
from .tradability import Days, OUT

START = pd.Timestamp("2020-01-01")
COST = 1.0
TARGETS, STOPS = (10, 20), (10, 15, 20)
MAX_HOLD = 20
LAST_ENTRY = "10:40"


def levels_table(b):
    full = b[b["rth_date"] >= START - pd.Timedelta(days=30)]
    W = lambda a, z, f=0.6: F.window(full, a, z, f)
    rth = W("09:30", "16:00", 0.9)
    days = rth.dropna(subset=["o"]).index
    days = days[days >= START]
    atr = (rth["h"] - rth["l"]).dropna().rolling(14, min_periods=10).mean().shift(1)
    prev = rth.dropna(subset=["o"]).shift(1)
    on, r69 = W("18:00", "09:30", 0.5), W("06:00", "09:00")
    h8, h9, orb = W("08:00", "09:00", .75), W("09:00", "10:00", .75), W("09:30", "09:45", .8)
    pre8 = W("18:00", "08:00", 0.5)
    L = pd.DataFrame(index=days)
    L["atr"] = atr
    L["ONH"], L["ONL"] = on["h"], on["l"]
    L["PDH"], L["PDL"], L["PDC"] = prev["h"], prev["l"], prev["c"]
    L["M69"] = (r69["h"] + r69["l"]) / 2
    L["H8H"], L["H8L"], L["H8M"] = h8["h"], h8["l"], (h8["h"] + h8["l"]) / 2
    L["H9H"], L["H9L"], L["H9M"] = h9["h"], h9["l"], (h9["h"] + h9["l"]) / 2
    L["ORH"], L["ORL"] = orb["h"], orb["l"]
    # contexts
    ow = (orb["h"] - orb["l"]).reindex(days) / L["atr"]
    edges = np.nanquantile(ow[days <= pd.Timestamp("2025-03-14")], [1/3, 2/3])
    L["ctx_orb"] = pd.cut(ow, [-np.inf, *edges, np.inf], labels=["ORB narrow", "ORB mid", "ORB wide"]).astype(object)
    p8 = ((pre8["c"] - pre8["l"]) / (pre8["h"] - pre8["l"])).reindex(days)
    L["ctx_8am"] = pd.cut(p8, [-np.inf, 1/3, 2/3, np.inf], labels=["8am low third", "8am mid third", "8am high third"]).astype(object)
    gap = ((rth["o"] - prev["c"]) / atr).reindex(days)
    L["ctx_gap"] = pd.cut(gap.abs(), [-np.inf, .15, .5, np.inf], labels=["gap small", "gap mid", "gap large"]).astype(object)
    rng = np.random.default_rng(7)
    for tag, (a, z) in {"A": ("09:25", "09:30"), "B": ("09:40", "09:45"), "C": ("09:55", "10:00")}.items():
        px = W(a, z, 0.2)["c"].reindex(days)
        for k in range(1, 5):
            L[f"P{tag}{k}"] = px + rng.uniform(-0.15, 0.15, len(days)) * L["atr"].to_numpy()
    # ADAPTED: reference frozen to the original holdout window (17 Mar 2025 - 29 Apr 2026)
    ref = np.nanmedian(L.loc[(L.index >= pd.Timestamp("2025-03-17")) & (L.index <= pd.Timestamp("2026-04-29")), "atr"])
    L["scale"] = L["atr"] / ref                     # raw pts = today pts * scale
    return L, ref


WINDOWS = {  # level -> (from, until) entry window
    **{k: ("09:30", LAST_ENTRY) for k in ("ONH", "ONL", "PDH", "PDL", "PDC", "M69")},
    **{k: ("09:30", "09:59") for k in ("H8H", "H8L", "H8M")},
    **{k: ("10:00", LAST_ENTRY) for k in ("H9H", "H9L", "H9M")},
    **{k: ("09:45", LAST_ENTRY) for k in ("ORH", "ORL")},
    **{f"PA{k}": ("09:30", LAST_ENTRY) for k in range(1, 5)},
    **{f"PB{k}": ("09:45", LAST_ENTRY) for k in range(1, 5)},
    **{f"PC{k}": ("10:00", LAST_ENTRY) for k in range(1, 5)},
}
# which placebo set each real level is compared with (same entry-window start)
PLACEBO_OF = {**{k: "PA" for k in ("ONH", "ONL", "PDH", "PDL", "PDC", "M69", "H8H", "H8L", "H8M")},
              **{k: "PB" for k in ("ORH", "ORL")},
              **{k: "PC" for k in ("H9H", "H9L", "H9M")}}


def first_touch(D, day, lvl, a, z):
    """Index of first bar in [a, z] trading through lvl, and the approach side
    (+1 = came from below). Requires price to start on one side of the level."""
    s, e = D.span[day]
    sm = D.sm[s:e]
    i0 = s + np.searchsorted(sm, F._sm(a))
    i1 = s + np.searchsorted(sm, F._sm(z), side="right")
    if i0 >= i1 or i0 == 0:
        return None
    ref = D.C[i0 - 1]                       # price just before the window opens
    if ref == lvl:
        return None
    side = 1 if ref < lvl else -1
    hit = (D.H[i0:i1] >= lvl) if side > 0 else (D.L[i0:i1] <= lvl)
    if not hit.any():
        return None
    return i0 + int(np.argmax(hit)), side


def scalp(D, i, entry, dirn, tgt_pts, stp_pts, scale, day_end):
    """Both bounds. Returns (cons_net, opt_net, minutes) in TODAY points."""
    T, S = tgt_pts * scale, stp_pts * scale
    tgt, stp = entry + dirn * T, entry - dirn * S
    j = min(i + MAX_HOLD, day_end)
    H, L, C = D.H[i:j + 1], D.L[i:j + 1], D.C[i:j + 1]
    if dirn > 0:
        st, tg = L <= stp, H >= tgt
    else:
        st, tg = H >= stp, L <= tgt
    res = []
    for optimistic in (False, True):
        s2, t2 = st.copy(), tg.copy()
        t2[0] = False
        if optimistic:
            s2[0] = False                      # entry-bar order unknown -> ignore it
        a =np.argmax(s2) if s2.any() else 10**9
        b_ = np.argmax(t2) if t2.any() else 10**9
        if a == 10**9 and b_ == 10**9:
            g = dirn * (C[-1] - entry); k = len(C) - 1
        elif (a <= b_) if not optimistic else (a < b_):
            g = -S; k = a
        else:
            g = T; k = b_
        res.append((g / scale - COST, k + 1))
    return res[0][0], res[1][0], res[0][1]


def run_trades(D, L):
    rows = []
    end_sm = F._sm("10:59")
    for day, r in L.iterrows():
        if day not in D.span or not np.isfinite(r["scale"]):
            continue
        s, e = D.span[day]
        day_end = s + np.searchsorted(D.sm[s:e], end_sm, side="right") - 1
        for lv, (a, z) in WINDOWS.items():
            lvl = r[lv]
            if not np.isfinite(lvl):
                continue
            ft = first_touch(D, day, lvl, a, z)
            if ft is None:
                continue
            i, side = ft
            for mode in ("fade", "break"):
                dirn = -side if mode == "fade" else side
                entry = lvl if mode == "fade" else lvl + side * 0.5
                if mode == "break" and not ((D.H[i] >= entry) if side > 0 else (D.L[i] <= entry)):
                    # break entry not triggered on the touch bar: look ahead up to 5 bars
                    k = np.flatnonzero((D.H[i:i + 6] >= entry) if side > 0 else (D.L[i:i + 6] <= entry))
                    if not len(k) or i + k[0] > day_end:
                        continue
                    ii = i + int(k[0])
                else:
                    ii = i
                for T in TARGETS:
                    for S in STOPS:
                        c, o, m = scalp(D, ii, entry, dirn, T, S, r["scale"], day_end)
                        rows.append({"day": day, "level": lv, "mode": mode, "T": T, "S": S,
                                     "cons": c, "opt": o, "mins": m,
                                     "ctx_orb": r["ctx_orb"], "ctx_8am": r["ctx_8am"], "ctx_gap": r["ctx_gap"]})
    t = pd.DataFrame(rows)
    t["period"] = period_of(t["day"])
    return t


def agg(g):
    S, T = g["Sv"].iloc[0], g["Tv"].iloc[0]
    n = len(g)
    c, o = g["cons"] / S, g["opt"] / S
    return pd.Series({
        "n": n, "per_month": round(n / max(g["day"].dt.to_period("M").nunique(), 1), 1),
        "win_cons_%": round(100 * (g["cons"] > 0).mean(), 1),
        "win_opt_%": round(100 * (g["opt"] > 0).mean(), 1),
        "breakeven_%": round(100 * (S + COST) / (T + S), 1),
        "net_pts_cons": round(g["cons"].mean(), 2), "net_pts_opt": round(g["opt"].mean(), 2),
        "R_cons": round(c.mean(), 3), "t_cons": round(c.mean() / (c.std(ddof=1) / np.sqrt(n)), 2) if n > 2 else np.nan,
        "R_opt": round(o.mean(), 3), "mins_med": g["mins"].median(),
        "se_cons": c.std(ddof=1) / np.sqrt(n) if n > 2 else np.nan,
        "se_opt": o.std(ddof=1) / np.sqrt(n) if n > 2 else np.nan})


def vs_placebo(t):
    """Per (level, mode, T, S): edge over the matching placebo set, both scorings."""
    t = t.copy()
    t["Tv"], t["Sv"] = t["T"], t["S"]
    t["grp"] = np.where(t["level"].str.match(r"P[ABC]\d"), t["level"].str[:2], t["level"])
    keys = ["grp", "mode", "T", "S"]
    a = t.groupby(keys).apply(agg, include_groups=False).reset_index()
    real = a[~a["grp"].isin(["PA", "PB", "PC"])].rename(columns={"grp": "level"})
    plac = a[a["grp"].isin(["PA", "PB", "PC"])].set_index(["grp", "mode", "T", "S"])
    out = []
    for _, r in real.iterrows():
        p = plac.loc[(PLACEBO_OF[r["level"]], r["mode"], r["T"], r["S"])]
        ec, eo = r["R_cons"] - p["R_cons"], r["R_opt"] - p["R_opt"]
        tc = ec / np.hypot(r["se_cons"], p["se_cons"])
        to = eo / np.hypot(r["se_opt"], p["se_opt"])
        est = -COST / r["S"] + (ec + eo) / 2
        out.append({**{k: r[k] for k in ("level", "mode", "T", "S", "n", "per_month", "win_cons_%", "win_opt_%",
                                         "breakeven_%", "mins_med")},
                    "placebo_win_cons_%": p["win_cons_%"], "placebo_win_opt_%": p["win_opt_%"],
                    "edge_R_cons": round(ec, 3), "edge_R_opt": round(eo, 3),
                    "t_cons": round(tc, 2), "t_opt": round(to, 2),
                    "est_net_R": round(est, 3), "est_net_pts": round(est * r["S"], 2)})
    return pd.DataFrame(out)


def run():
    bars = load_nq()
    b = F.prep(bars[bars["Datetime"] >= pd.Timestamp("2019-11-01", tz="America/New_York")])
    L, ref = levels_table(b)
    print(f"ATR14 reference (holdout median) = {ref:.0f} pts; scale range "
          f"{L['scale'].min():.2f}-{L['scale'].max():.2f}")
    import os
    f = f"{OUT}/scalp_trades.parquet"
    if os.path.exists(f) and "--rerun" not in __import__("sys").argv:
        t = pd.read_parquet(f)
    else:
        D = Days(b)
        t = run_trades(D, L)
        t.to_parquet(f)
    keys = ["level", "mode", "T", "S"]
    disc = vs_placebo(t[t["period"] == "discovery"])
    disc.to_csv(f"{OUT}/scalp_discovery.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
    cols = ["level", "mode", "T", "S", "n", "per_month", "win_cons_%", "win_opt_%", "placebo_win_cons_%",
            "placebo_win_opt_%", "edge_R_cons", "edge_R_opt", "t_cons", "t_opt", "est_net_R", "est_net_pts", "mins_med"]
    print("\n=== discovery 2020 - Mar 2025 (best 40 by estimated net R) ===")
    print(disc.sort_values("est_net_R", ascending=False)[cols].head(40).to_string(index=False))
    keep = disc[(disc["est_net_R"] > 0) & (disc["t_cons"] > 2) & (disc["t_opt"] > 2)]
    print(f"\n{len(keep)} of {len(disc)} cells pass the discovery bar")
    if len(keep):
        hold = vs_placebo(t[t["period"] == "holdout"]).merge(keep[keys], on=keys)
        hold.to_csv(f"{OUT}/scalp_holdout.csv", index=False)
        print("\n=== holdout (unseen) for the passing cells ===")
        print(hold[cols].to_string(index=False))


if __name__ == "__main__":
    run()
