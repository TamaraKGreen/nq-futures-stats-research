"""
RTH ATR breakout (user-supplied rules), MNQ.

Rules as given (CT) -> ET used here:
  reference   RTH opening price, 08:30 CT = 09:30 ET (open of the 09:30 bar)
  checks      every 30 min, 09:00-15:00 CT = closes at 10:00, 10:30, ... 16:00 ET (30-min bars)
  long        30-min close > open + K_ENTRY x ATR ; short: close < open - K_ENTRY x ATR
  filter      yesterday's RTH range < FILTER x ATR5
  stop        K_STOP x ATR from entry, no target
  exit        stop, or forced close at 15:40 CT = 16:40 ET (close of the 16:39 bar)
  re-entry    allowed after a stop-out on a later check; one position at a time
  sizing      contracts = floor(RISK_USD / (stop pts x $2)), >= 1, total risk <= MAX_RISK_USD

Assumptions (not in the rules):
  ATR         daily ATR(14) on RTH bars (true range vs prior RTH close, back-adjusted across rolls),
              simple mean of the 14 PRIOR days. ATR5 the same over 5 days.
  fills       entry at the signal bar's close + 1 tick slippage; stop fills at the stop - 1 tick;
              stop is checked on 1-min bars from the bar after entry.
  costs       $1.50 per contract round trip commission + fees.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from research.scripts.load import load

PT, TICK, COMM = 2.0, 0.25, 1.50          # MNQ $/pt, tick, $ per contract round trip
OUT = ROOT / "research" / "output"

BASE = dict(K_ENTRY=0.2, K_STOP=0.25, FILTER=1.2, ATR_N=14, ATR_F=5, RISK_USD=500, MAX_RISK_USD=1000,
            CHECKS=[600 + 30 * i for i in range(13)], EXIT=16 * 60 + 39, SIDES=("long", "short"), REENTRY=True)


def prep():
    d = load("all")
    naive = d.ts.dt.tz_localize(None)
    d["mod"] = naive.dt.hour * 60 + naive.dt.minute
    rth = d[(d["mod"] >= 570) & (d["mod"] < 960)]
    D = rth.groupby("sess").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"),
                                n=("close", "size"), adj=("roll_adj", "first"))
    D = D[D.n >= 380]                                   # full RTH days only
    ha, la, ca = D.h + D.adj, D.l + D.adj, D.c + D.adj  # back-adjusted for TR across rolls
    pc = ca.shift(1)
    D["tr"] = pd.concat([ha - la, (ha - pc).abs(), (la - pc).abs()], axis=1).max(axis=1)
    D["rng"] = D.h - D.l
    return d, D


def run(d, D, **kw):
    P = {**BASE, **kw}
    D = D.copy()
    D["atr"] = D.tr.rolling(P["ATR_N"]).mean().shift(1)
    D["atr5"] = D.tr.rolling(P["ATR_F"]).mean().shift(1)
    D["prev_rng"] = D.rng.shift(1)
    D["ok"] = (D.prev_rng < P["FILTER"] * D.atr5) if P["FILTER"] is not None else True
    H, L, C, O, M = (d[c].to_numpy() for c in ("high", "low", "close", "open", "mod"))
    idx = d.groupby("sess").indices
    trades = []
    for s, r in D.iterrows():
        if not np.isfinite(r.atr) or not np.isfinite(r.atr5) or s not in idx:
            continue
        if P["FILTER"] is not None and not r.ok:
            continue
        ii = idx[s]; mm = M[ii]
        pos_of = {m: ii[k] for k, m in enumerate(mm)}
        if 570 not in pos_of or P["EXIT"] not in pos_of:
            continue
        ref = O[pos_of[570]]
        i_exit = pos_of[P["EXIT"]]
        busy_until = -1
        for t in P["CHECKS"]:
            j = pos_of.get(t - 1)                       # bar closing at time t
            if j is None or j <= busy_until or j >= i_exit:
                continue
            px = C[j]
            if px > ref + P["K_ENTRY"] * r.atr and "long" in P["SIDES"]:
                dirn = 1
            elif px < ref - P["K_ENTRY"] * r.atr and "short" in P["SIDES"]:
                dirn = -1
            else:
                continue
            entry = px + dirn * TICK
            stop_pts = P["K_STOP"] * r.atr
            stop = entry - dirn * stop_pts
            qty = max(1, int(P["RISK_USD"] // (stop_pts * PT)))
            qty = min(qty, max(1, int(P["MAX_RISK_USD"] // (stop_pts * PT))))
            hh, ll = H[j + 1:i_exit + 1], L[j + 1:i_exit + 1]
            hit = (ll <= stop) if dirn > 0 else (hh >= stop)
            if hit.any():
                k = int(np.argmax(hit)); x = stop - dirn * TICK; how = "stop"; jx = j + 1 + k
            else:
                x = C[i_exit]; how = "time"; jx = i_exit
            pts = dirn * (x - entry)
            pnl = qty * (pts * PT - COMM)
            trades.append({"sess": s, "check": f"{t // 60:02d}:{t % 60:02d}", "dir": "long" if dirn > 0 else "short",
                           "entry": entry, "exit": x, "how": how, "qty": qty, "pts": pts, "R": pts / stop_pts,
                           "pnl": pnl, "risk_usd": qty * stop_pts * PT, "atr": r.atr})
            busy_until = jx
            if not P["REENTRY"] or how == "time":
                break
    return pd.DataFrame(trades)


def stats(t, days):
    if not len(t):
        return pd.Series({"trades": 0})
    daily = t.groupby("sess").pnl.sum().reindex(days, fill_value=0.0)
    eq = daily.cumsum()
    sh = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    return pd.Series({"trades": len(t), "win %": 100 * (t.pnl > 0).mean(), "avg R": t.R.mean(),
                      "net $": t.pnl.sum(), "$/trade": t.pnl.mean(), "PF": t.pnl[t.pnl > 0].sum() / -t.pnl[t.pnl < 0].sum(),
                      "sharpe": sh, "max DD $": (eq.cummax() - eq).max()})


def by_year(t, D):
    days = pd.Series(D.index, index=D.index)
    rows = {}
    for y, ds in days.groupby(days.dt.year):
        rows[y] = stats(t[t.sess.dt.year == y], ds.index)
    rows["ALL"] = stats(t, D.index)
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    d, D = prep()
    t = run(d, D)
    t.to_parquet(OUT / "atr_breakout_base.parquet", index=False)
    print("BASE RULES  (MNQ, costs incl.)\n")
    print(by_year(t, D[D.index >= D.index[20]]).round(2).to_string())
