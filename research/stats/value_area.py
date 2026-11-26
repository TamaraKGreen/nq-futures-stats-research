"""
Where NQ opens relative to the PRIOR day's value area, and whether it then trades to VAH / VAL.

Profile    prior session 18:00-17:00 ET (ETH), built from 1-min bars: each bar's volume is spread
           evenly over every tick from its low to its high. POC = highest-volume tick.
Value area 70% of volume, grown from the POC one tick at a time toward the side with more volume.
           Prior-day levels are shifted by the roll difference when a contract roll sits between
           the two sessions, so they're in the current contract's prices.
Opens      18:00 (session open), 08:00 (NY futures open), 09:30 (RTH open) - open of that 1-min bar.
Window     18:00 and 08:00 opens: to the session end (16:59 bar). 09:30 open: to the 15:59 bar (RTH close).
           Also reported: hit by 11:00.
Hit        price trades at or through the level (low <= level for a level below, high >= level above).
Location   above VAH / below VAL / inside, inside split into quarters of the VA:
           Q1 = VAL to 25%, Q2 = 25-50%, Q3 = 50-75%, Q4 = 75% to VAH.
RW         random-walk benchmark: same distances, path mirrored with probability 1/2
           (= 0.5 x [reached that distance up] + 0.5 x [reached it down]). Rate well above RW = more
           than just "the level was close".
Sessions   full sessions only (>= 1200 bars) for both the prior and the current session.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from research.scripts.load import load

TICK = 0.25
OUT = ROOT / "research" / "output"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)


def value_area(lo, hi, vol, pct=0.70):
    lo_t = np.round(lo / TICK).astype(np.int64); hi_t = np.round(hi / TICK).astype(np.int64)
    base = lo_t.min(); n = hi_t.max() - base + 1
    per = vol / (hi_t - lo_t + 1)
    diff = np.zeros(n + 1)
    np.add.at(diff, lo_t - base, per)
    np.add.at(diff, hi_t - base + 1, -per)
    prof = np.cumsum(diff)[:n]
    poc = int(np.argmax(prof)); tot = prof.sum()
    a = b = poc; acc = prof[poc]
    while acc < pct * tot:
        up = prof[b + 1] if b + 1 < n else -1
        dn = prof[a - 1] if a - 1 >= 0 else -1
        if up >= dn:
            b += 1; acc += up
        else:
            a -= 1; acc += dn
    return (base + a) * TICK, (base + b) * TICK, (base + poc) * TICK


def build():
    d = load("all")
    d = d[d.sess_bars >= 1200].reset_index(drop=True)
    naive = d.ts.dt.tz_localize(None)
    d["mod"] = naive.dt.hour * 60 + naive.dt.minute
    prof = {}
    for s, g in d.groupby("sess", sort=True):
        val, vah, poc = value_area(g.low.to_numpy(), g.high.to_numpy(), g.volume.to_numpy(np.float64))
        prof[s] = (val, vah, poc, g.roll_adj.iloc[0])
    sessions = sorted(prof)
    rows = []
    H, L, M = d.high.to_numpy(), d.low.to_numpy(), d["mod"].to_numpy()
    O = d.open.to_numpy()
    starts = d.groupby("sess").indices
    for k in range(1, len(sessions)):
        s, p = sessions[k], sessions[k - 1]
        if (s - p).days > 4:                     # missing session in between
            continue
        idx = starts[s]
        val, vah, poc, adj_p = prof[p]
        shift = adj_p - d.roll_adj.iloc[idx[0]]   # convert prior contract prices to current contract
        val, vah, poc = val + shift, vah + shift, poc + shift
        mods = M[idx]
        for name, t0, t_end in (("18:00 open", 18 * 60, 16 * 60 + 59), ("08:00 open", 8 * 60, 16 * 60 + 59),
                                ("09:30 open", 9 * 60 + 30, 15 * 60 + 59)):
            # minutes since 18:00 so the overnight wraps correctly
            sm = (mods - 1080) % 1440
            a0, a1, a11 = (t0 - 1080) % 1440, (t_end - 1080) % 1440, (11 * 60 - 1080) % 1440
            j = np.flatnonzero(sm == a0)
            if not len(j):
                continue
            i0 = idx[j[0]]
            w = idx[(sm >= a0) & (sm <= a1)]
            w11 = idx[(sm >= a0) & (sm < a11)]
            op = O[i0]
            hh, ll = H[w].max(), L[w].min()
            h11, l11 = (H[w11].max(), L[w11].min()) if len(w11) else (np.nan, np.nan)
            if op > vah:
                loc = "above VAH"
            elif op < val:
                loc = "below VAL"
            else:
                q = min(int((op - val) / (vah - val) * 4), 3) if vah > val else 0
                loc = f"inside Q{q + 1}"

            def rw(level):
                dist = abs(level - op)
                return 0.5 * float(hh - op >= dist) + 0.5 * float(op - ll >= dist)

            # first touch order for inside opens
            first = np.nan
            if loc.startswith("inside"):
                hv = np.flatnonzero(H[w] >= vah); lv = np.flatnonzero(L[w] <= val)
                iv, il = (hv[0] if len(hv) else 10**9), (lv[0] if len(lv) else 10**9)
                first = np.nan if iv == il == 10**9 else ("VAH first" if iv < il else ("VAL first" if il < iv else "same bar"))
            rows.append({"sess": s, "open_time": name, "loc": loc, "open": op, "vah": vah, "val": val, "poc": poc,
                         "va_width": vah - val,
                         "hit_vah": float(hh >= vah) if op < vah else float(ll <= vah),
                         "hit_val": float(ll <= val) if op > val else float(hh >= val),
                         "hit_poc": float((hh >= poc) and (ll <= poc)),
                         "hit_vah_11": (float(h11 >= vah) if op < vah else float(l11 <= vah)) if len(w11) else np.nan,
                         "hit_val_11": (float(l11 <= val) if op > val else float(h11 >= val)) if len(w11) else np.nan,
                         "rw_vah": rw(vah), "rw_val": rw(val), "first": first})
    return pd.DataFrame(rows)


def table(t, by11):
    def f(x):
        o = {"n": len(x),
             "VAH %": 100 * x.hit_vah.mean(), "VAH rw %": 100 * x.rw_vah.mean(),
             "VAL %": 100 * x.hit_val.mean(), "VAL rw %": 100 * x.rw_val.mean(),
             "POC %": 100 * x.hit_poc.mean()}
        if by11:
            o["VAH by 11 %"] = 100 * x.hit_vah_11.mean(); o["VAL by 11 %"] = 100 * x.hit_val_11.mean()
        return pd.Series(o)
    order = ["above VAH", "inside Q4", "inside Q3", "inside Q2", "inside Q1", "below VAL"]
    r = t.groupby("loc").apply(f, include_groups=False).reindex(order)
    r.insert(1, "share %", 100 * r.n / r.n.sum())
    return r.round(1)


if __name__ == "__main__":
    t = build()
    t.to_parquet(OUT / "value_area_opens.parquet", index=False)
    print(f"sessions: {t.sess.min().date()} -> {t.sess.max().date()}\n")
    for name in ("18:00 open", "08:00 open", "09:30 open"):
        x = t[t.open_time == name]
        print("=" * 120, f"\n{name}  ({len(x)} sessions; hit = by {'16:00' if name.startswith('09') else '17:00'})\n")
        print(table(x, by11=name != "18:00 open").to_string())
        ins = x[x["loc"].str.startswith("inside")]
        fo = pd.crosstab(ins["loc"], ins["first"], normalize="index").mul(100).round(1)
        print("\ninside opens - which edge is touched first (% of days):")
        print(fo.reindex(["inside Q4", "inside Q3", "inside Q2", "inside Q1"]).to_string())
        x = x.assign(year=x.sess.dt.year)
        y = x.groupby(["loc", "year"]).agg(vah=("hit_vah", "mean"), val=("hit_val", "mean"), n=("hit_vah", "size"))
        print("\nby year (VAH % / VAL %, n):")
        print((y.vah * 100).round(0).astype(int).astype(str).str.cat((y.val * 100).round(0).astype(int).astype(str), sep=" / ")
              .str.cat(y.n.astype(str), sep="  n=").unstack("year")
              .reindex(["above VAH", "inside Q4", "inside Q3", "inside Q2", "inside Q1", "below VAL"]).to_string())
        print()
