"""
Statmap reversion: fade price at an M or D zone back toward the candle OPEN, with or without an iFVG.

Candles (ET, anchored at 18:00): 1H, 4H, D (18:00-17:00), 12H (18:00-06:00).
Levels   open +/- median manipulation (M+/M-) and +/- median distribution (D+/D-) of the prior LB candles of
         the same slot, LB in {20, 60, 120}.
Sets     CURRENT = the candle's own levels; PRIOR = the previous candle's levels that were NOT touched in their
         own candle (roll-consistent: back-adjusted prices).
Trade    fade direction: level above the open -> short, below -> long. Target = candle open. Exit at the
         candle's last bar close at the latest.
  E0 limit   entry at the level when price trades 1 tick through it; stop = same distance as the target
             beyond the level (1:1); the entry bar counts against us if it also reaches the stop.
  E1 iFVG    after the touch, the first 1m iFVG in the fade direction within 15 bars (oos engine); entry at the
             inverting bar's close; stop 1 tick beyond the extreme since the touch; target = open.
Control  for every trade: a random bar of the same candle, same direction, same stop and target distances.
Cost     1 pt per trade. R = net points / stop distance. edge = R - random R.
Split    dev (to 2025-07-09) / later (2025-07-10 on).
"""
import sys
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from research.scripts.load import load

OUT = ROOT / "research" / "output"
COST, TICK = 1.0, 0.25
SPLIT = pd.Timestamp("2025-07-10")
TFS = {"1H": 60, "4H": 240, "D": 1380, "12H": 720}


def get_bars():
    d = load("all")
    for c in ("open", "high", "low", "close"):
        d[c] = d[c] + d.roll_adj
    naive = d.ts.dt.tz_localize(None)
    d["mod"] = naive.dt.hour * 60 + naive.dt.minute
    d["sm"] = (d["mod"] - 1080) % 1440
    return d.reset_index(drop=True)


def ifvg_events(d):
    f = OUT / "ifvg_1m_all.parquet"
    if f.exists():
        return pd.read_parquet(f)
    from oos.ifvg import htf_bars
    from oos.ifvg_htf import engine_full
    b = pd.DataFrame({"Datetime": d.ts, "Open": d.open, "High": d.high, "Low": d.low, "Close": d.close})
    _, E = engine_full(htf_bars(b, 1), 12)
    E["t_inv"] = pd.to_datetime(E.t_inv)
    naive = d.ts.dt.tz_localize(None).to_numpy()
    E["pos"] = np.searchsorted(naive, (E.t_inv - pd.Timedelta(minutes=1)).to_numpy())   # inverting bar index
    E = E[["pos", "dir"]].drop_duplicates()
    E.to_parquet(f, index=False)
    return E


def candle_table(d, tf):
    minutes = TFS[tf]
    if tf == "12H":
        x = d[d.sm < 720].assign(slot=0)
    elif tf == "D":
        x = d.assign(slot=0)
    else:
        x = d.assign(slot=np.minimum(d.sm // minutes, (1380 - 1) // minutes))
    x = x.assign(pos=x.index)
    g = x.groupby(["sess", "slot"], sort=True)
    c = g.agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"), n=("close", "size"),
              p0=("pos", "first"), p1=("pos", "last")).reset_index()
    c = c[c.n >= 0.8 * min(minutes, 1380)].reset_index(drop=True)
    bull = c.c > c.o
    c["manip"] = np.where(bull, c.o - c.l, c.h - c.o)
    c["distr"] = np.where(bull, c.h - c.o, c.o - c.l)
    return c


def work(args):
    tf, LB = args
    d = get_bars()
    E = ifvg_events(d)
    H, L, C = d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy()
    epos, edir = E.pos.to_numpy(), E.dir.to_numpy()
    order = np.argsort(epos); epos, edir = epos[order], edir[order]
    sess_rng = d.groupby("sess").agg(h=("high", "max"), l=("low", "min"), n=("close", "size"))
    atr = (sess_rng.h - sess_rng.l).where(sess_rng.n >= 1200).rolling(14, min_periods=10).mean().shift(1)
    c = candle_table(d, tf)
    for k in ("manip", "distr"):
        c[f"med_{k}"] = c.groupby("slot")[k].transform(lambda s: s.shift(1).rolling(LB, min_periods=max(10, LB // 2)).median())
    c["M+"], c["M-"] = c.o + c.med_manip, c.o - c.med_manip
    c["D+"], c["D-"] = c.o + c.med_distr, c.o - c.med_distr
    c["atr"] = c.sess.map(atr)
    rng = np.random.default_rng(hash((tf, LB)) % 2**32)
    rows = []
    LV = ["M+", "M-", "D+", "D-"]
    for k in range(1, len(c)):
        r = c.iloc[k]
        if not np.isfinite(r["M+"]) or not np.isfinite(r.atr):
            continue
        p0, p1 = int(r.p0), int(r.p1)
        if p1 - p0 < 10:
            continue
        sets = [("current", lv, r[lv]) for lv in LV]
        pr = c.iloc[k - 1]
        if np.isfinite(pr["M+"]) and int(pr.p1) + 1 >= p0 - 61:       # adjacent previous candle
            for lv in LV:
                x = pr[lv]
                touched = (pr.h >= x) if lv.endswith("+") else (pr.l <= x)
                if not touched:
                    sets.append(("prior", lv, x))
        hh, ll, cc = H[p0:p1 + 1], L[p0:p1 + 1], C[p0:p1 + 1]
        for st, lv, x in sets:
            if not np.isfinite(x) or x == r.o:
                continue
            dr = -1 if x > r.o else 1                     # fade direction
            dist = abs(x - r.o)
            thr = (hh >= x + TICK) if dr < 0 else (ll <= x - TICK)
            touch = (hh >= x) if dr < 0 else (ll <= x)
            if not touch.any():
                continue
            it = int(np.argmax(touch))
            base = {"sess": r.sess, "tf": tf, "LB": LB, "set": st, "level": lv, "dist_atr": dist / r.atr,
                    "slot": int(r.slot)}
            # E0 limit at the level
            if thr.any():
                j = int(np.argmax(thr))
                entry, tgt, stp = x, r.o, x - dr * dist
                pts = _sim(hh, ll, cc, j, entry, dr, tgt, stp, amb=True)
                rp = _rand(hh, ll, cc, rng, dr, dist, dist)
                rows.append({**base, "entry": "E0 limit", "R": (pts - COST) / dist, "rR": None if rp is None else (rp - COST) / dist})
            # E1 iFVG after the touch
            a = np.searchsorted(epos, p0 + it); bnd = np.searchsorted(epos, min(p0 + it + 15, p1 - 1), side="right")
            cand = np.flatnonzero(edir[a:bnd] == dr)
            if len(cand):
                kpos = int(epos[a + cand[0]]); j = kpos - p0
                entry = cc[j]
                ext = hh[it:j + 1].max() if dr < 0 else ll[it:j + 1].min()
                stp = ext - dr * TICK
                risk, reward = abs(entry - stp), dr * (r.o - entry)
                if risk > 0 and reward > 0 and j + 1 <= len(cc) - 1:
                    pts = _sim(hh, ll, cc, j + 1, entry, dr, r.o, stp, amb=False)
                    rp = _rand(hh, ll, cc, rng, dr, risk, reward)
                    rows.append({**base, "entry": "E1 iFVG", "R": (pts - COST) / risk,
                                 "rR": None if rp is None else (rp - COST) / risk, "rr": reward / risk})
    return pd.DataFrame(rows)


def _sim(hh, ll, cc, j, entry, dr, tgt, stp, amb):
    H, L = hh[j:], ll[j:]
    hs = (L <= stp) if dr > 0 else (H >= stp)
    ht = (H >= tgt) if dr > 0 else (L <= tgt)
    if amb:
        ht = ht.copy(); ht[0] = False
    a = int(np.argmax(hs)) if hs.any() else 10**9
    b = int(np.argmax(ht)) if ht.any() else 10**9
    if a == b == 10**9:
        return dr * (cc[-1] - entry)
    return dr * (stp - entry) if a <= b else dr * (tgt - entry)


def _rand(hh, ll, cc, rng, dr, risk, reward):
    n = len(cc)
    if n < 12:
        return None
    j = int(rng.integers(5, n - 5))
    e = cc[j]
    return _sim(hh, ll, cc, j + 1, e, dr, e + dr * reward, e - dr * risk, amb=False)


def report(X):
    X["period"] = np.where(X.sess < SPLIT, "dev", "later")
    X["edge"] = X.R - X.rR

    def f(x):
        e = x.edge.dropna()
        dv, lt = x[x.period == "dev"], x[x.period == "later"]
        return pd.Series({"n": len(x), "win %": 100 * (x.R > 0).mean(), "R": x.R.mean(), "rand R": x.rR.mean(),
                          "edge": e.mean(), "t": e.mean() / (e.std() / np.sqrt(len(e))) if len(e) > 2 else np.nan,
                          "dev edge": dv.edge.mean(), "later edge": lt.edge.mean(), "later n": len(lt),
                          "med dist ATR": x.dist_atr.median()})
    print("STATMAP REVERSION - fade M / D zones back to the candle open (R net of 1 pt; edge vs random entry)\n")
    for tf in ["1H", "4H", "12H", "D"]:
        x = X[X.tf == tf]
        if not len(x):
            continue
        print("=" * 130, f"\n{tf}\n")
        t = x.groupby(["entry", "set", "level", "LB"]).apply(f, include_groups=False)
        print(t.round(3).to_string(), "\n")
    print("=" * 130, "\nSUMMARY: pooled over M+/M-/D+/D-, by timeframe x entry x set x lookback\n")
    print(X.groupby(["tf", "entry", "set", "LB"]).apply(f, include_groups=False).round(3).to_string())
    g = X.groupby(["tf", "entry", "set", "level", "LB"]).apply(f, include_groups=False)
    print(f"\ncells: {len(g)}   |t| > 2: {(g.t.abs() > 2).sum()} (chance ~{0.046 * len(g):.0f})   "
          f"edge > 0 in BOTH dev and later with t > 2: {((g.t > 2) & (g['dev edge'] > 0) & (g['later edge'] > 0)).sum()}")
    print(g[(g.t > 2)].sort_values("t", ascending=False).head(20).round(3).to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
    d = get_bars(); ifvg_events(d); del d
    jobs = [(tf, lb) for tf in TFS for lb in (20, 60, 120)]
    with Pool(4) as p:
        parts = p.map(work, jobs)
    X = pd.concat(parts, ignore_index=True)
    X.to_parquet(OUT / "statmap_reversion.parquet", index=False)
    report(X)
