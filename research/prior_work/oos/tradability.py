"""
Tradability check for the plan's section-3 rules.

    py -m oos.tradability

Each rule becomes a mechanical trade: entry, target, stop, time exit. Stops come
from a small fixed grid; the grid value is chosen on DISCOVERY ONLY (best net
points per trade), then that one version is run on the holdout. All grid rows
are written out too, so nothing is hidden.

Fills (conservative):
  * limit entries at a level fill when price trades through it
  * in the bar where entry happens, only the stop is checked (unknown order)
  * if stop and target are both inside one later bar, the stop wins
  * COST points per round trip are deducted from every trade
Entries/exit times are ET clock; RTH trades exit at the 15:59 close.
"""
import numpy as np
import pandas as pd

from .data import load_nq, DISCOVERY_END, HOLDOUT_START, period_of
from . import families as F

COST = 1.0          # points per round trip (commission + slippage), NQ = $20/pt
OUT = F.__file__.replace("families.py", "results")


# ------------------------------------------------------------------ engine --

class Days:
    """Fast per-day access to 1-minute bars."""
    def __init__(self, b):
        b = b.sort_values("Datetime").reset_index(drop=True)
        self.sm = b["sm"].to_numpy("int32")
        self.O, self.H = b["Open"].to_numpy(), b["High"].to_numpy()
        self.L, self.C = b["Low"].to_numpy(), b["Close"].to_numpy()
        d = b["rth_date"].to_numpy()
        starts = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
        ends = np.r_[starts[1:], len(d)]
        self.span = dict(zip(pd.to_datetime(d[starts]), zip(starts, ends)))

    def sim(self, day, t0, entry, dirn, target, stop, t1, entry_bar_is_open=False):
        """Trade from the bar at minute t0 (since 18:00) to the bar at t1."""
        if day not in self.span or not np.isfinite([entry, target, stop]).all():
            return None
        s, e = self.span[day]
        sm = self.sm[s:e]
        i0 = s + np.searchsorted(sm, t0)
        i1 = s + np.searchsorted(sm, t1, side="right") - 1
        if i0 > i1 or i0 >= e:
            return None
        H, L = self.H[i0:i1 + 1], self.L[i0:i1 + 1]
        if dirn > 0:
            st, tg = L <= stop, H >= target
        else:
            st, tg = H >= stop, L <= target
        if not entry_bar_is_open:
            tg = tg.copy(); tg[0] = False
        ist = np.argmax(st) if st.any() else 10**9
        itg = np.argmax(tg) if tg.any() else 10**9
        if ist == 10**9 and itg == 10**9:
            k, px, why = len(H) - 1, self.C[i1], "time"
        elif ist <= itg:
            k, px, why = ist, stop, "stop"
        else:
            k, px, why = itg, target, "target"
        adverse = (entry - L[:k + 1].min()) if dirn > 0 else (H[:k + 1].max() - entry)
        gross = dirn * (px - entry)
        return {"gross": gross, "net": gross - COST, "risk": abs(entry - stop),
                "exit": why, "mae": max(adverse, 0.0), "bars": k + 1}


def summarise(t: pd.DataFrame) -> dict:
    if not len(t):
        return {"trades": 0}
    n = t["net"]
    eq = n.cumsum()
    yr = t.groupby(pd.to_datetime(t["day"]).dt.year)["net"].sum()
    return {"trades": len(t), "win_%": round(100 * (n > 0).mean(), 1),
            "target_%": round(100 * (t["exit"] == "target").mean(), 1),
            "avg_net_pts": round(n.mean(), 2), "median_net": round(n.median(), 2),
            "avg_R": round((n / t["risk"]).mean(), 3),
            "profit_factor": round(n[n > 0].sum() / -n[n < 0].sum(), 2) if (n < 0).any() else np.inf,
            "total_pts": round(n.sum(), 0), "max_dd_pts": round((eq.cummax() - eq).max(), 0),
            "avg_risk_pts": round(t["risk"].mean(), 1),
            "years_positive": f"{int((yr > 0).sum())}/{len(yr)}"}


# ------------------------------------------------------------------- setups --

def build_levels(bars):
    b = F.prep(bars)
    rth = F.window(b, "09:30", "16:00", 0.9)
    days = rth.dropna(subset=["o"]).index
    atr = (rth["h"] - rth["l"]).dropna().rolling(14, min_periods=10).mean().shift(1).reindex(days)
    ib = F.window(b, "09:30", "10:30", 0.8).reindex(days)
    pre8 = F.window(b, "18:00", "08:00", 0.5).reindex(days)
    b["hr"] = (b["mod"] // 60).astype("int8")
    h = (b.groupby(["rth_date", "hr"]).agg(o=("Open", "first"), hi=("High", "max"), lo=("Low", "min"),
                                           c=("Close", "last"), n=("Close", "size")))
    return b, days, atr, ib, pre8, h


def setup_ib_fade(D, b, days, atr, ib, narrow_only, k):
    """Narrow IB: fade the first IB break back to the IB mid."""
    w = ib["h"] - ib["l"]
    q = F.disc_quintiles(w / atr, days)
    tu = F.first_touch(b, "10:30", "16:00", ib["h"], "up")
    td = F.first_touch(b, "10:30", "16:00", ib["l"], "dn")
    rows = []
    for d in days:
        if narrow_only and q.get(d) != "Q1 narrow":
            continue
        u, dn = tu.get(d), td.get(d)
        if pd.isna(u) and pd.isna(dn):
            continue
        up = pd.notna(u) and (pd.isna(dn) or u < dn)
        hi, lo, ww = ib.at[d, "h"], ib.at[d, "l"], w.get(d)
        mid = (hi + lo) / 2
        if up:
            r = D.sim(d, u, hi, -1, mid, hi + k * ww, F._sm("15:59"))
        else:
            r = D.sim(d, dn, lo, +1, mid, lo - k * ww, F._sm("15:59"))
        if r:
            rows.append({"day": d, **r})
    return pd.DataFrame(rows)


def setup_on_extreme(D, days, pre8, m):
    """8am in top third -> long to the overnight high (bottom third: short to low).
    Stop = m x the distance to target."""
    rows = []
    pos = (pre8["c"] - pre8["l"]) / (pre8["h"] - pre8["l"])
    for d in days:
        p = pos.get(d)
        if pd.isna(p) or 1/3 <= p <= 2/3:
            continue
        e = pre8.at[d, "c"]
        if p > 2/3:
            tgt = pre8.at[d, "h"]; dist = tgt - e; dirn = 1
        else:
            tgt = pre8.at[d, "l"]; dist = e - tgt; dirn = -1
        if dist < 2:                       # target closer than costs make sense
            continue
        r = D.sim(d, F._sm("08:00"), e, dirn, tgt, e - dirn * m * dist, F._sm("15:59"), entry_bar_is_open=True)
        if r:
            rows.append({"day": d, **r})
    return pd.DataFrame(rows)


def setup_nine_mid(D, h, days, m):
    """9:00 hour: from the 9:00 open, trade toward the 8:00 hour's mid; exit 9:59."""
    rows = []
    for d in days:
        try:
            p, c = h.loc[(d, 8)], h.loc[(d, 9)]
        except KeyError:
            continue
        if p["n"] < 45 or c["n"] < 45:
            continue
        mid = (p["hi"] + p["lo"]) / 2
        e = c["o"]
        dist = abs(mid - e)
        if dist < 2:
            continue
        dirn = 1 if mid > e else -1
        r = D.sim(d, F._sm("09:00"), e, dirn, mid, e - dirn * m * dist, F._sm("09:59"), entry_bar_is_open=True)
        if r:
            rows.append({"day": d, **r})
    return pd.DataFrame(rows)


def _sweep_trades(D, b, h, days, hour, k, mode, side_filter=None):
    """Sweep of the previous hour's extreme during `hour`.
    mode 'cont': stop-entry through the level, ride to the hour close.
    mode 'fade': limit entry at the level against the sweep, exit at the hour close."""
    rows = []
    t_start, t_end = F._sm(f"{hour:02d}:00"), F._sm(f"{hour:02d}:59")
    for d in days:
        try:
            p, c = h.loc[(d, hour - 1)], h.loc[(d, hour)]
        except KeyError:
            continue
        if p["n"] < 45 or c["n"] < 45 or not (p["lo"] <= c["o"] <= p["hi"]):
            continue
        s, e = D.span.get(d, (0, 0))
        sm = D.sm[s:e]
        i0, i1 = s + np.searchsorted(sm, t_start), s + np.searchsorted(sm, t_end, side="right")
        Hh, Ll = D.H[i0:i1], D.L[i0:i1]
        if not len(Hh):
            continue
        a = np.flatnonzero(Hh > p["hi"]); z = np.flatnonzero(Ll < p["lo"])
        ia = a[0] if len(a) else 10**9; iz = z[0] if len(z) else 10**9
        if ia == iz == 10**9:
            continue
        side = "H" if ia < iz else "L"
        if side_filter and side != side_filter:
            continue
        rng = p["hi"] - p["lo"]
        t0 = int(D.sm[i0 + min(ia, iz)])
        lvl = p["hi"] if side == "H" else p["lo"]
        if mode == "cont":
            dirn = 1 if side == "H" else -1
            entry = lvl
            stop = lvl - dirn * k * rng
            far = 1e9 * dirn                       # no target: time exit
        else:
            dirn = -1 if side == "H" else 1
            entry = lvl
            stop = lvl - dirn * k * rng
            far = 1e9 * dirn
        r = D.sim(d, t0, entry, dirn, entry + far, stop, t_end)
        if r:
            rows.append({"day": d, "side": side, **r})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- run --

def run():
    bars = load_nq()
    b, days, atr, ib, pre8, h = build_levels(bars)
    D = Days(b)
    specs = []
    for k in (0.25, 0.5, 0.75, 1.0):
        specs.append(("IB fade to mid - narrow IB (3.1)", f"stop {k}x IB beyond", lambda k=k: setup_ib_fade(D, b, days, atr, ib, True, k)))
        specs.append(("IB fade to mid - all days (control)", f"stop {k}x IB beyond", lambda k=k: setup_ib_fade(D, b, days, atr, ib, False, k)))
    for m in (1, 2, 3, 5):
        specs.append(("8am third -> overnight extreme (3.2)", f"stop {m}x target dist", lambda m=m: setup_on_extreme(D, days, pre8, m)))
        specs.append(("9:00 hour -> 8:00 mid (3.5)", f"stop {m}x target dist", lambda m=m: setup_nine_mid(D, h, days, m)))
    for k in (0.25, 0.5, 1.0):
        specs.append(("9:00 sweep continuation (3.5)", f"stop {k}x prev-hr range", lambda k=k: _sweep_trades(D, b, h, days, 9, k, "cont")))
        specs.append(("11-13 sweep of low, fade (3.5)", f"stop {k}x prev-hr range",
                      lambda k=k: pd.concat([_sweep_trades(D, b, h, days, H_, k, "fade", "L") for H_ in (11, 12, 13)],
                                            ignore_index=True)))
    rows, trades = [], {}
    for name, var, fn in specs:
        t = fn()
        if not len(t):
            continue
        t["period"] = period_of(t["day"])
        trades[(name, var)] = t
        for per in ("discovery", "holdout"):
            rows.append({"setup": name, "variant": var, "period": per, **summarise(t[t["period"] == per])})
        print(f"{name:40s} {var:24s} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/tradability_all.csv", index=False)

    # choose each setup's variant on DISCOVERY only
    disc = res[res["period"] == "discovery"]
    pick = disc.loc[disc.groupby("setup")["avg_net_pts"].idxmax(), ["setup", "variant"]]
    chosen = res.merge(pick, on=["setup", "variant"])
    chosen.to_csv(f"{OUT}/tradability_chosen.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== all variants ===")
    print(res.to_string(index=False))
    print("\n=== variant chosen on discovery, then holdout ===")
    print(chosen.to_string(index=False))
    # adverse excursion of winners for the chosen variants (discovery)
    print("\n=== chosen variants: adverse excursion before exit (discovery) ===")
    for _, r in pick.iterrows():
        t = trades[(r["setup"], r["variant"])]
        t = t[t["period"] == "discovery"]
        w = t[t["exit"] == "target"]
        print(f"{r['setup']:40s} winners' MAE median {w['mae'].median():6.1f} pts  p75 {w['mae'].quantile(.75):6.1f}  "
              f"| risk median {t['risk'].median():6.1f} pts | median bars {t['bars'].median():.0f}")


if __name__ == "__main__":
    run()
