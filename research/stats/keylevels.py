"""
Key levels by distance (stage 2), the "statistical mapper" session sequences, and key levels as targets
after entry triggers (stage 3). RTH only. Back-adjusted prices throughout (levels across rolls consistent).

LEVELS (per day; minute they become known, ET)
  PDH PDL PDC         prior RTH high / low / close                          09:30
  ONH ONL             overnight 18:00-09:29 high / low                      09:30
  ASH ASL             Asia 18:00-02:00 high / low                           09:30
  LNH LNL             London 02:00-08:00 high / low                         09:30
  H8H H8L H8M         08:00-08:59 high / low / mid                          09:30
  DO WO               18:00 daily open / Sunday 18:00 weekly open           09:30
  rVAH rVAL rPOC      prior RTH value area (70%) / POC                      09:30
  eVAH eVAL ePOC      prior ETH (18:00-17:00) value area / POC              09:30
  SM610 M+ M- D+ D-   Statmap 4H 06-10 candle levels (60-candle lookback)   09:30
  IBH IBL             09:30-10:29 high / low                                10:30
  SM1014 M+ M- D+ D-  Statmap 4H 10-14 candle levels                        10:30
ATR = mean of the prior 14 full RTH ranges.

STAGE 2  decision at 09:30 (IB / SM1014 at 10:30). Hit = traded at the level by 11:00 / by 16:00.
         EXPECTED = share of days whose excursion from the decision price (up or down, pooled) reached that
         distance by the same time -> what distance + time alone give. Edge = hit - expected.
MAPPER   first extreme taken: Asia range during London (02:00-08:00); London range during 08:00-16:00;
         overnight range during RTH. Random-walk benchmark for "high first" = where the session opened
         inside the range ((open - low) / (high - low)).
STAGE 3  triggers 09:45-15:30 with a direction:
           iFVG    1m iFVG (oos engine), its direction
           sweep   price trades through a key level and a 1-min bar closes back on the other side within
                   5 bars -> direction away from the level (first sweep per level per day)
           VWAP    1-min close crosses RTH VWAP (>= 10 min since the last cross) -> direction of the cross
           EMA20 / EMA200   5-min close crosses the EMA of 5-min closes -> direction of the cross
           RANDOM  random minutes, random direction (control, same pipeline)
         Targets = every known key level in the trigger direction 0.05-1.5 ATR away.
         hit16 = target traded by 16:00; expected from the same distance curve at that time of day.
         TRADE   enter at the trigger close, target = level, stop = same distance the other way;
                 first touch by 16:00 (stop wins a bar touching both), else exit at 16:00 close;
                 net R after 1 pt cost.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from research.scripts.load import load
from research.stats.statmap import candles

OUT = ROOT / "research" / "output"
COST = 1.0
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)


def bars():
    d = load("all")
    for c in ("open", "high", "low", "close"):
        d[c] = d[c] + d.roll_adj
    naive = d.ts.dt.tz_localize(None)
    d["mod"] = naive.dt.hour * 60 + naive.dt.minute
    d["day"] = d.sess                                     # RTH date (18:00 anchor)
    return d


def window(d, a, z):
    sm = (d["mod"] - 1080) % 1440
    s, e = (a - 1080) % 1440, (z - 1080) % 1440 or 1440
    w = d[(sm >= s) & (sm < e)]
    return w.groupby("day").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"), n=("close", "size"))


def level_table(d):
    rth = window(d, 570, 960)
    days = rth[rth.n >= 380].index
    R = rth.loc[days]
    L = pd.DataFrame(index=days)
    L["atr"] = (R.h - R.l).rolling(14, min_periods=10).mean().shift(1)
    prev = R.shift(1)
    L["PDH"], L["PDL"], L["PDC"] = prev.h, prev.l, prev.c
    on = window(d, 1080, 570).reindex(days); L["ONH"], L["ONL"] = on.h, on.l
    a = window(d, 1080, 120).reindex(days); L["ASH"], L["ASL"] = a.h, a.l
    ln = window(d, 120, 480).reindex(days); L["LNH"], L["LNL"] = ln.h, ln.l
    h8 = window(d, 480, 540).reindex(days); L["H8H"], L["H8L"], L["H8M"] = h8.h, h8.l, (h8.h + h8.l) / 2
    full = window(d, 1080, 1020)
    L["DO"] = full.o.reindex(days)
    wk = pd.Series(full.index - pd.to_timedelta(full.index.dayofweek, unit="D"), index=full.index)
    L["WO"] = full.o.groupby(wk).transform("first").reindex(days)
    for prof, tag in (("rth", "r"), ("eth", "e")):
        v = pd.read_parquet(OUT / f"value_area_opens_{prof}.parquet")
        v = v[v.open_time == "09:30 open"].set_index("sess")
        adj = d.groupby("day").roll_adj.first()
        for c_, nm in (("vah", "VAH"), ("val", "VAL"), ("poc", "POC")):
            L[f"{tag}{nm}"] = (v[c_] + adj.reindex(v.index)).reindex(days)
    ib = window(d, 570, 630).reindex(days); L["IBH"], L["IBL"] = ib.h, ib.l
    c4 = candles(d.rename(columns={}), "4H")
    for slot, tag in ((3, "SM610"), (4, "SM1014")):
        x = c4[c4.slot == slot].set_index("sess")
        for lv, nm in (("Mu", "M+"), ("Md", "M-"), ("Du", "D+"), ("Dd", "D-")):
            L[f"{tag} {nm}"] = x[lv].reindex(days)
    return L, R


AVAIL = {k: 570 for k in ["PDH", "PDL", "PDC", "ONH", "ONL", "ASH", "ASL", "LNH", "LNL", "H8H", "H8L", "H8M", "DO", "WO",
                          "rVAH", "rVAL", "rPOC", "eVAH", "eVAL", "ePOC", "SM610 M+", "SM610 M-", "SM610 D+", "SM610 D-"]}
AVAIL.update({k: 630 for k in ["IBH", "IBL", "SM1014 M+", "SM1014 M-", "SM1014 D+", "SM1014 D-"]})


def day_arrays(d, days):
    r = d[(d["mod"] >= 570) & (d["mod"] < 960) & d.day.isin(days)]
    out = {}
    for day, g in r.groupby("day"):
        if len(g) >= 380:
            out[day] = (g["mod"].to_numpy(), g.open.to_numpy(), g.high.to_numpy(), g.low.to_numpy(), g.close.to_numpy(),
                        g.volume.to_numpy(np.float64))
    return out


def excursion_curves(A, atr, grid):
    """For each grid start minute and horizon end, pooled up/down excursions (ATR) across days."""
    cur = {}
    for t0 in grid:
        for h in (660, 960):
            if h <= t0:
                continue
            xs = []
            for day, (m, O, H, L_, C, V) in A.items():
                a = atr.get(day)
                if not np.isfinite(a):
                    continue
                j = np.flatnonzero(m == t0)
                if not len(j):
                    continue
                j = j[0]; sel = (m >= t0) & (m < h)
                p = O[j]
                xs += [(H[sel].max() - p) / a, (p - L_[sel].min()) / a]
            cur[(t0, h)] = np.sort(np.array(xs))
    return cur


def expect(cur, t0, h, dist):
    c = cur[(t0, h)]
    return 1 - np.searchsorted(c, dist, side="left") / len(c)


# ---------------------------------------------------------------- stage 2
def stage2(L, A):
    atr = L.atr.to_dict()
    cur = excursion_curves(A, atr, [570, 630])
    rows = []
    for day, (m, O, H, Lw, C, V) in A.items():
        if day not in L.index or not np.isfinite(atr[day]):
            continue
        r = L.loc[day]
        for lv, t0 in AVAIL.items():
            x = r[lv]
            if not np.isfinite(x):
                continue
            j = np.flatnonzero(m == t0)
            if not len(j):
                continue
            j = j[0]; p = O[j]
            dist = abs(x - p) / atr[day]
            rec = {"day": day, "level": lv, "dist": dist, "side": "above" if x > p else "below", "t0": t0}
            for h, tag in ((660, "11"), (960, "16")):
                if h <= t0:
                    continue
                sel = (m >= t0) & (m < h)
                rec[f"hit{tag}"] = float(H[sel].max() >= x) if x > p else float(Lw[sel].min() <= x)
                rec[f"exp{tag}"] = expect(cur, t0, h, dist)
            rows.append(rec)
    return pd.DataFrame(rows)


def report2(S):
    print("=" * 120, "\nSTAGE 2 - KEY LEVELS vs DISTANCE (decision 09:30; IB and SM1014 at 10:30)\n")
    bins = [-.001, .1, .25, .5, .75, 1, 1.5, 99]; labs = ["<0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75-1", "1-1.5", ">1.5"]
    S["db"] = pd.cut(S.dist, bins, labels=labs)
    g = S.groupby("db", observed=True).agg(n=("hit16", "size"), hit16=("hit16", "mean"), exp16=("exp16", "mean"))
    print("ALL LEVELS POOLED, by distance (by 16:00):  hit % vs expected %")
    print((g[["hit16", "exp16"]] * 100).round(1).assign(n=g.n).to_string(), "\n")

    def f(x):
        yr = x.groupby(x.day.dt.year).apply(lambda y: (y.hit16 - y.exp16).mean(), include_groups=False)
        e = (x.hit16 - x.exp16).mean()
        se = (x.hit16 - x.exp16).std() / np.sqrt(len(x))
        o = {"n": len(x), "med dist ATR": x.dist.median(), "hit16 %": 100 * x.hit16.mean(), "exp16 %": 100 * x.exp16.mean(),
             "edge16 pp": 100 * e, "t": e / se if se > 0 else np.nan, "yrs +": f"{int((yr > 0).sum())}/{len(yr)}"}
        if "hit11" in x and x.hit11.notna().any():
            o.update({"hit11 %": 100 * x.hit11.mean(), "exp11 %": 100 * x.exp11.mean(), "edge11 pp": 100 * (x.hit11 - x.exp11).mean()})
        return pd.Series(o)
    T = S.groupby("level").apply(f, include_groups=False).sort_values("edge16 pp")
    print("BY LEVEL (each event's expectation uses its own distance):")
    print(T.round(2).to_string(), "\n")
    print("BY LEVEL AND SIDE (above / below the 09:30 price):")
    T2 = S.groupby(["level", "side"]).apply(f, include_groups=False)
    print(T2[["n", "med dist ATR", "hit16 %", "exp16 %", "edge16 pp", "t", "yrs +"]].round(2).to_string())
    return T


# ---------------------------------------------------------------- mapper
def mapper(d, L):
    print("\n" + "=" * 120, "\nSTATISTICAL MAPPER - which extreme of the prior session is taken first\n")
    rows = []
    specs = [("London takes Asia range", (1080, 120), (120, 480)),
             ("08:00-16:00 takes London range", (120, 480), (480, 960)),
             ("RTH takes overnight range", (1080, 570), (570, 960))]
    sm = (d["mod"] - 1080) % 1440
    for name, (a1, z1), (a2, z2) in specs:
        rng = window(d, a1, z1)
        s2, e2 = (a2 - 1080) % 1440, (z2 - 1080) % 1440 or 1440
        w = d[(sm >= s2) & (sm < e2)]
        for day, g in w.groupby("day"):
            if day not in rng.index or len(g) < 0.8 * (e2 - s2):
                continue
            hi, lo = rng.at[day, "h"], rng.at[day, "l"]
            if not hi > lo:
                continue
            op = g.open.iloc[0]
            H, Lw = g.high.to_numpy(), g.low.to_numpy()
            iu = np.flatnonzero(H > hi); idn = np.flatnonzero(Lw < lo)
            a = iu[0] if len(iu) else 10**9; b = idn[0] if len(idn) else 10**9
            pos = (op - lo) / (hi - lo)
            first = "none" if a == b == 10**9 else ("same bar" if a == b else ("high" if a < b else "low"))
            rows.append({"seq": name, "day": day, "pos": pos, "first": first, "both": float(len(iu) > 0 and len(idn) > 0),
                         "inside_open": 0 <= pos <= 1})
    M = pd.DataFrame(rows)
    M["pos_b"] = pd.cut(M.pos, [-99, 0, 1/3, 2/3, 1, 99], labels=["opened below", "low third", "mid third", "high third", "opened above"])

    def f(x):
        t = x[x["first"].isin(["high", "low"])]
        ins = t[t.inside_open]
        hf = (t["first"] == "high").astype(float)
        rw = ins.pos.clip(0, 1).mean() if len(ins) else np.nan
        yr = t.groupby(t.day.dt.year).apply(lambda y: (y["first"] == "high").mean() - (y.pos.clip(0, 1).mean() if y.inside_open.all() else np.nan), include_groups=False)
        return pd.Series({"n": len(x), "took one %": 100 * len(t) / len(x), "both %": 100 * x.both.mean(),
                          "HIGH first %": 100 * hf.mean(), "RW %": 100 * rw if x.inside_open.all() else np.nan,
                          "excess pp": 100 * (hf.mean() - rw) if x.inside_open.all() else np.nan})
    for name, g in M.groupby("seq", sort=False):
        print(f"--- {name}")
        print(g.groupby("pos_b", observed=True).apply(f, include_groups=False).round(1).to_string(), "\n")
    return M


# ---------------------------------------------------------------- stage 3
def triggers(d, A, L):
    from oos.ifvg import htf_bars
    from oos.ifvg_htf import engine_full
    b = pd.DataFrame({"Datetime": d.ts, "Open": d.open, "High": d.high, "Low": d.low, "Close": d.close}).reset_index(drop=True)
    _, E = engine_full(htf_bars(b, 1), 12)
    E["t_inv"] = pd.to_datetime(E.t_inv)
    E["mod"] = E.t_inv.dt.hour * 60 + E.t_inv.dt.minute - 1           # minute of the inverting bar
    E["day"] = (E.t_inv + pd.Timedelta(hours=6)).dt.normalize()
    E = E[(E["mod"] >= 585) & (E["mod"] <= 930)].sort_values("t_inv").drop_duplicates(["day", "mod", "dir"])
    trig = [pd.DataFrame({"day": E.day, "mod": E["mod"], "dir": E.dir.astype(int), "trigger": "iFVG"})]
    # EMAs on 5-min closes (24h)
    naive = d.ts.dt.tz_localize(None)
    k5 = naive.dt.floor("5min")
    c5 = d.groupby(k5).close.last()
    rng = np.random.default_rng(5)
    lvl_cols = list(AVAIL)
    rows = []
    for name, span in (("EMA20", 20), ("EMA200", 200)):
        e = c5.ewm(span=span, adjust=False).mean()
        x = pd.DataFrame({"c": c5, "e": e})
        x["cross"] = np.sign(x.c - x.e).diff().fillna(0) / 2
        x = x[x.cross != 0]
        x["end"] = x.index + pd.Timedelta(minutes=4)
        x["mod"] = x.end.dt.hour * 60 + x.end.dt.minute
        x["day"] = (x.end + pd.Timedelta(hours=6)).dt.normalize()
        x = x[(x["mod"] >= 585) & (x["mod"] <= 930)]
        trig.append(pd.DataFrame({"day": x.day, "mod": x["mod"], "dir": np.sign(x.cross).astype(int), "trigger": name}))
    for day, (m, O, H, Lw, C, V) in A.items():
        if day not in L.index:
            continue
        tp = (H + Lw + C) / 3
        vw = np.cumsum(tp * V) / np.maximum(np.cumsum(V), 1)
        last = -99
        for k in range(1, len(m)):
            if m[k] < 585 or m[k] > 930:
                continue
            s0, s1 = np.sign(C[k - 1] - vw[k - 1]), np.sign(C[k] - vw[k])
            if s0 != 0 and s1 != 0 and s0 != s1 and m[k] - last >= 10:
                rows.append((day, m[k], int(s1), "VWAP")); last = m[k]
        # sweeps of key levels
        r = L.loc[day]
        for lv in lvl_cols:
            x = r[lv]
            if not np.isfinite(x):
                continue
            t_av = AVAIL[lv]
            j0 = np.flatnonzero(m >= max(t_av, 585))
            if not len(j0):
                continue
            j0 = j0[0]
            above = C[j0 - 1] < x                            # level above price at the start -> sweep of a high
            hit = np.flatnonzero((H[j0:] > x) if above else (Lw[j0:] < x))
            if not len(hit):
                continue
            k = j0 + hit[0]
            back = np.flatnonzero((C[k:k + 6] < x) if above else (C[k:k + 6] > x))
            if len(back) and m[k + back[0]] <= 930:
                rows.append((day, m[k + back[0]], -1 if above else 1, "sweep"))
        for mm in rng.choice(np.arange(585, 931), size=8, replace=False):
            rows.append((day, int(mm), int(rng.choice([-1, 1])), "RANDOM"))
    trig.append(pd.DataFrame(rows, columns=["day", "mod", "dir", "trigger"]))
    return pd.concat(trig, ignore_index=True)


def stage3(T, A, L):
    atr = L.atr.to_dict()
    grid = list(range(585, 931, 15))
    cur = {}
    for t0 in grid:
        xs = []
        for day, (m, O, H, Lw, C, V) in A.items():
            a = atr.get(day)
            j = np.flatnonzero(m == t0)
            if not len(j) or not np.isfinite(a):
                continue
            j = j[0]
            xs += [(H[j + 1:].max() - C[j]) / a, (C[j] - Lw[j + 1:].min()) / a]
        cur[t0] = np.sort(np.array(xs))
    rows = []
    for day, g in T.groupby("day"):
        if day not in A or day not in L.index or not np.isfinite(atr.get(day, np.nan)):
            continue
        m, O, H, Lw, C, V = A[day]; a = atr[day]; r = L.loc[day]
        pos = {mm: k for k, mm in enumerate(m)}
        for _, t in g.iterrows():
            k = pos.get(int(t["mod"]))
            if k is None or k + 1 >= len(m):
                continue
            p, dr = C[k], int(t.dir)
            hh, ll, cc = H[k + 1:], Lw[k + 1:], C[-1]
            cmh, cml = np.maximum.accumulate(hh), np.minimum.accumulate(ll)
            gkey = grid[min(len(grid) - 1, max(0, (int(t["mod"]) - 585) // 15))]
            for lv, t_av in AVAIL.items():
                if t["mod"] < t_av:
                    continue
                x = r[lv]
                if not np.isfinite(x) or dr * (x - p) <= 0:
                    continue
                dist = abs(x - p) / a
                if dist < 0.05 or dist > 1.5:
                    continue
                ht = (cmh >= x) if dr > 0 else (cml <= x)
                stp = p - dr * abs(x - p)
                hs = (cml <= stp) if dr > 0 else (cmh >= stp)
                it = int(np.argmax(ht)) if ht.any() else 10**9
                is_ = int(np.argmax(hs)) if hs.any() else 10**9
                risk = abs(x - p)
                if it == is_ == 10**9:
                    pts = dr * (cc - p)
                elif is_ <= it:
                    pts = -risk
                else:
                    pts = risk
                rows.append({"day": day, "trigger": t.trigger, "level": lv, "dist": dist,
                             "hit16": float(it < 10**9), "exp16": 1 - np.searchsorted(cur[gkey], dist) / len(cur[gkey]),
                             "win": float(pts > 0 and it < is_), "R": (pts - COST) / risk})
    return pd.DataFrame(rows)


def report3(X):
    print("\n" + "=" * 120, "\nSTAGE 3 - KEY LEVELS AS TARGETS AFTER A TRIGGER (09:45-15:30, levels 0.05-1.5 ATR away)\n")
    bins = [0.05, .1, .25, .5, .75, 1, 1.5]; labs = ["0.05-0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75-1", "1-1.5"]
    X["db"] = pd.cut(X.dist, bins, labels=labs)

    def f(x):
        e = (x.hit16 - x.exp16)
        return pd.Series({"n": len(x), "hit16 %": 100 * x.hit16.mean(), "exp16 %": 100 * x.exp16.mean(), "edge pp": 100 * e.mean(),
                          "trade win %": 100 * x.win.mean(), "avg R net": x.R.mean(),
                          "R t": x.R.mean() / (x.R.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan})
    print("BY TRIGGER (all target levels pooled):")
    print(X.groupby("trigger").apply(f, include_groups=False).round(2).to_string(), "\n")
    print("BY TRIGGER AND DISTANCE:")
    print(X.groupby(["trigger", "db"], observed=True).apply(f, include_groups=False).round(2).to_string(), "\n")
    print("BY TRIGGER AND TARGET LEVEL (n >= 300), sorted by trade R:")
    g = X.groupby(["trigger", "level"]).apply(f, include_groups=False)
    g = g[g.n >= 300].sort_values("avg R net", ascending=False)
    print(g.head(25).round(2).to_string())
    print("...\n", g.tail(8).round(2).to_string())
    print(f"\ncells (trigger x level, n>=300): {len(g)}  with |R t| > 2: {(g['R t'].abs() > 2).sum()}")


if __name__ == "__main__":
    d = bars()
    L, R = level_table(d)
    A = day_arrays(d, L.index)
    S = stage2(L, A)
    S.to_parquet(OUT / "keylevels_stage2.parquet", index=False)
    report2(S)
    M = mapper(d, L)
    T = triggers(d, A, L)
    print(f"\ntriggers: {T.trigger.value_counts().to_dict()}")
    X = stage3(T, A, L)
    X.to_parquet(OUT / "keylevels_stage3.parquet", index=False)
    report3(X)


def stage3_only():
    d = bars()
    L, R = level_table(d)
    A = day_arrays(d, L.index)
    T = triggers(d, A, L)
    print(f"triggers: {T.trigger.value_counts().to_dict()}")
    X = stage3(T, A, L)
    X.to_parquet(OUT / "keylevels_stage3.parquet", index=False)
    report3(X)
