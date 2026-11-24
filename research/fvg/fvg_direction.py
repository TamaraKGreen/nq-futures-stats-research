"""
Exploratory: do FVGs / iFVGs tell us direction? (FVG/iFVG engine = oos/ifvg_htf.py, unchanged)

FVG   bull: low(C) > high(A); bear: high(C) < low(A). Known when bar C closes (t_born).
iFVG  an intact FVG closed through by a later bar: bull gap closed below -> bearish iFVG,
      bear gap closed above -> bullish iFVG. Known at that bar's close (t_inv).

PART A - RTH session. Signal = the FIRST FVG (all three bars at/after 09:30) or the FIRST
iFVG (inverting bar closes at/after 09:30) of the day, on 1m and 5m bars.
  Outcomes measured FROM THE SIGNAL (price = close of the signal bar):
    close_vs_sig   RTH close (15:59) above the signal price?        baseline: same, all days,
                                                                    at the same clock time
    touch_first    from the signal price, +0.25 x ATR14(RTH range) before -0.25 x ATR?  RW = 50%
    open_vs_close  your original framing (RTH close vs 09:30 open) - shown only to compare

PART B - clock ranges (15/30/60m, first break within 1x duration, dev range events).
  For each range: direction of the FIRST and LAST 1m FVG formed inside the range, net FVG count,
  and the first iFVG inside the range. Outcome = which side breaks first, vs the random-walk
  expectation close_pos (as in research/ranges/direction.py).

Splits: dev (to 2025-07-09) and later (2025-07-10 on) reported separately.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from oos.ifvg import htf_bars
from oos.ifvg_htf import engine_full
from oos.orderflow import load_flow

OUT = ROOT / "research" / "output"
SPLIT = pd.Timestamp("2025-07-10")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)


def events(b, tf, keep):
    G, E = engine_full(htf_bars(b, tf), keep)
    G["t_born"] = pd.to_datetime(G["t_born"])
    E["t_inv"] = pd.to_datetime(E["t_inv"])
    G["dir"] = np.where(G["bull"], 1, -1)
    G["start"] = G["t_born"] - pd.Timedelta(minutes=3 * tf)       # bar A start
    return G, E


def part_a(b):
    b = b.copy()
    b["t"] = b["Datetime"].dt.tz_localize(None)
    b["day"] = b["t"].dt.normalize()
    rth = b[(b["mod"] >= 570) & (b["mod"] < 960)]
    D = rth.groupby("day").agg(o=("Open", "first"), c=("Close", "last"), h=("High", "max"), l=("Low", "min"), n=("Close", "size"))
    D = D[D.n >= 380]
    D["atr"] = (D.h - D.l).rolling(14, min_periods=10).mean().shift(1)
    t_idx = pd.Index(rth["t"])
    C, H, L = rth["Close"].to_numpy(), rth["High"].to_numpy(), rth["Low"].to_numpy()
    T = rth["t"].to_numpy()
    rows = []
    for tf, keep in ((1, 12), (5, 12)):
        G, E = events(b, tf, keep)
        sigs = {
            "first FVG": G[(G.start.dt.hour * 60 + G.start.dt.minute >= 570)].rename(columns={"t_born": "ts"})[["ts", "dir"]],
            "first iFVG": E[(E.t_inv.dt.hour * 60 + E.t_inv.dt.minute >= 570 + tf)].rename(columns={"t_inv": "ts"})[["ts", "dir"]],
        }
        for name, s in sigs.items():
            s = s[(s.ts.dt.hour * 60 + s.ts.dt.minute) < 960].copy()
            s["day"] = s.ts.dt.normalize()
            s = s[s.day.isin(D.index)].sort_values("ts").drop_duplicates("day")
            for _, x in s.iterrows():
                i = t_idx.searchsorted(x.ts) - 1                 # last 1m bar closing at/before the signal
                if i < 0 or T[i].astype("datetime64[D]") != np.datetime64(x.day.date()):
                    continue
                d = D.loc[x.day]
                px = C[i]
                j_end = t_idx.searchsorted(x.day + pd.Timedelta(hours=16))
                k = 0.25 * d.atr
                fw_h, fw_l = H[i + 1:j_end], L[i + 1:j_end]
                up = np.flatnonzero(fw_h >= px + k); dn = np.flatnonzero(fw_l <= px - k)
                iu = up[0] if len(up) else 10**9; idn = dn[0] if len(dn) else 10**9
                tf_ = np.nan if iu == idn else float(iu < idn)
                rows.append({"tf": f"{tf}m", "signal": name, "day": x.day, "sig_min": x.ts.hour * 60 + x.ts.minute,
                             "dir": "bull" if x.dir > 0 else "bear", "close_vs_sig": float(d.c > px),
                             "touch_up_first": tf_, "open_vs_close": float(d.c > d.o)})
    A = pd.DataFrame(rows)
    A["period"] = np.where(A.day < SPLIT, "dev", "later")
    A["when"] = pd.cut(A.sig_min, [569, 600, 660, 960], labels=["09:30-10:00", "10:00-11:00", "after 11:00"])
    # baseline for close_vs_sig: every day, price at the same minute -> RTH close above it?
    base = []
    for day, d in D.iterrows():
        s0 = t_idx.searchsorted(day + pd.Timedelta(minutes=570)); s1 = t_idx.searchsorted(day + pd.Timedelta(hours=16))
        cc = C[s0:s1]
        if len(cc) < 380:
            continue
        base.append({"day": day, "p_up_any_minute": float((d.c > cc[:-1]).mean())})
    B = pd.DataFrame(base)
    return A, B


def summarise_a(A, B):
    print("=" * 110, "\nPART A - RTH SESSION: first FVG / iFVG after 09:30\n")
    print("bull%/bear%: share of days where RTH closes ABOVE the signal price (bull signal should be high,")
    print("bear signal low). touch: +0.25 ATR hit before -0.25 ATR from the signal (RW 50%).")
    print("open->close: your original framing (biased: includes the FVG's own move).\n")
    bl = B.assign(period=np.where(B.day < SPLIT, "dev", "later")).groupby("period").p_up_any_minute.mean() * 100
    print("baseline: RTH close above the price at a random RTH minute:", bl.round(1).to_dict(), "\n")

    def f(x):
        bu, be = x[x.dir == "bull"], x[x.dir == "bear"]
        return pd.Series({"n_bull": len(bu), "n_bear": len(be),
                          "bull: close>sig%": 100 * bu.close_vs_sig.mean(), "bear: close>sig%": 100 * be.close_vs_sig.mean(),
                          "spread_pp": 100 * (bu.close_vs_sig.mean() - be.close_vs_sig.mean()),
                          "bull: touch up%": 100 * bu.touch_up_first.mean(), "bear: touch up%": 100 * be.touch_up_first.mean(),
                          "touch spread_pp": 100 * (bu.touch_up_first.mean() - be.touch_up_first.mean()),
                          "bull: open->close up%": 100 * bu.open_vs_close.mean(), "bear: open->close up%": 100 * be.open_vs_close.mean()})
    print(A.groupby(["tf", "signal", "period"]).apply(f, include_groups=False).round(1).to_string())
    print("\nBy time of the signal (dev):\n")
    print(A[A.period == "dev"].groupby(["tf", "signal", "when"], observed=True).apply(f, include_groups=False).round(1).to_string())
    print("\nBy year (dev + later), 5m first FVG:\n")
    x = A[(A.tf == "5m") & (A.signal == "first FVG")]
    print(x.groupby(x.day.dt.year).apply(f, include_groups=False)[["n_bull", "n_bear", "spread_pp", "touch spread_pp"]].round(1).to_string())


def part_b(b):
    G, E = events(b, 1, 12)
    rows = []
    ev = pd.concat([pd.read_parquet(OUT / "direction_dev_wait1.parquet")])
    ev = ev[ev.side.isin(["up", "down"])].copy()
    ev["s"] = ev.start.dt.tz_localize(None); ev["e"] = ev.end.dt.tz_localize(None) + pd.Timedelta(minutes=1)
    gb = G.sort_values("t_born"); tb = gb.t_born.to_numpy(); gs = gb.start.to_numpy(); gd = gb.dir.to_numpy()
    ei = E.sort_values("t_inv"); ti = ei.t_inv.to_numpy(); idr = ei.dir.to_numpy()
    s_ = ev.s.to_numpy(); e_ = ev.e.to_numpy()
    lo = np.searchsorted(tb, s_, side="left"); hi = np.searchsorted(tb, e_, side="right")
    ilo = np.searchsorted(ti, s_, side="left"); ihi = np.searchsorted(ti, e_, side="right")
    first, last, net, cnt, ifirst = [], [], [], [], []
    for k in range(len(ev)):
        m = gs[lo[k]:hi[k]] >= s_[k]                       # all three bars inside the range
        d = gd[lo[k]:hi[k]][m]
        first.append(d[0] if len(d) else 0); last.append(d[-1] if len(d) else 0)
        net.append(d.sum() if len(d) else 0); cnt.append(len(d))
        di = idr[ilo[k]:ihi[k]]
        ifirst.append(di[0] if len(di) else 0)
    ev["fvg_first"], ev["fvg_last"], ev["fvg_net"], ev["fvg_n"], ev["ifvg_first"] = first, last, net, cnt, ifirst
    ev["up"] = (ev.side == "up").astype(float)
    return ev


def summarise_b(ev):
    print("\n" + "=" * 110, "\nPART B - RANGES: 1m FVG / iFVG inside the range -> which side breaks first (dev, 1x wait)\n")
    print("excess = up-first% minus the random-walk expectation (close position). Same test as direction.py.\n")
    lab = {1: "bull", -1: "bear", 0: "none"}

    def f(x):
        yrs = x.groupby(x.sess.dt.year).apply(lambda y: y.up.mean() - y.close_pos.mean(), include_groups=False)
        e = x.up.mean() - x.close_pos.mean()
        return pd.Series({"n": len(x), "up_first%": 100 * x.up.mean(), "rw%": 100 * x.close_pos.mean(),
                          "excess_pp": 100 * e, "yrs_same_sign": f"{int((np.sign(yrs) == np.sign(e)).sum())}/{len(yrs)}"})
    for col, title in (("fvg_first", "FIRST FVG in the range"), ("fvg_last", "LAST FVG in the range"),
                       ("ifvg_first", "FIRST iFVG in the range")):
        print(f"--- {title}\n")
        print(ev.assign(c=ev[col].map(lab)).groupby(["dur", "c"]).apply(f, include_groups=False).round(1).to_string(), "\n")
    print("--- NET FVG count (bull - bear)\n")
    ev["net_b"] = pd.cut(ev.fvg_net, [-99, -2, -1, 0, 1, 99], labels=["<=-2", "-1", "0", "+1", ">=+2"])
    print(ev.groupby(["dur", "net_b"], observed=True).apply(f, include_groups=False).round(1).to_string(), "\n")
    print("--- Does the LAST FVG add anything beyond the candle direction? (60m)\n")
    x = ev[ev.dur == 60].assign(c=ev.fvg_last.map(lab))
    print(x.groupby(["bull_bar", "c"]).apply(f, include_groups=False).round(1).to_string())


if __name__ == "__main__":
    b = load_flow()
    A, B = part_a(b)
    A.to_parquet(OUT / "fvg_rth_signals.parquet", index=False)
    summarise_a(A, B)
    ev = part_b(b)
    ev.drop(columns=["s", "e"]).to_parquet(OUT / "fvg_range_dev.parquet", index=False)
    summarise_b(ev)
