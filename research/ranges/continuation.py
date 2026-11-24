"""
Exploratory: continuation-trade pullbacks after the first break (DEV split only).

Setup   the range closes; the first break (within 1x duration) defines direction D.
Levels  inside the range, named for an UP break (mirrored for DOWN):
          edge (the broken high), q75, vwap, mid, open, close
Entry   a limit at the level, filled only if price trades 1 tick THROUGH it, within
        PULLBACK_MULT x duration after the break bar (from the bar AFTER the break bar).
Exits   stop s range-widths beyond the level (against D), target k x risk in D, or
        target = the post-break extreme (retest of the high the breakout made).
        Entry bar: if it also reaches the stop -> loss (order inside the bar unknown).
        Same bar hits stop and target -> loss. Otherwise timeout after HOLD_MULT x
        duration (or session end) at that bar's close.
Costs   net figures deduct $4.50 round trip (0.225 pts) and 1 tick slippage on stop /
        timeout exits; limit entries and target exits have no slippage.
R       gross and net results are in multiples of the planned risk (entry - stop).

Every trade is also scored for how many of 7 momentum signals agreed with D at the
range close (candle direction, opposite extreme formed first, delta quintile, close vs
previous range, 4-range trend, close vs own VWAP, close vs previous range's VWAP).
"""
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from research.scripts.load import load

OUT = Path(__file__).resolve().parents[1] / "output"
PULLBACK_MULT, HOLD_MULT = 1.0, 4.0
TICK, COMM = 0.25, 0.225
LEVELS = ("edge", "q75", "vwap", "mid", "open", "close")
CONFIGS = {  # name: (stop in widths, target in R or 'ext')
    "s.25_1R": (0.25, 1.0), "s.25_2R": (0.25, 2.0),
    "s.5_1R": (0.5, 1.0), "s.5_2R": (0.5, 2.0), "s.5_ext": (0.5, "ext"),
}


def score(b):
    up = (b.side == "up").to_numpy()
    def agree(cond_up, cond_dn):
        return np.where(up, cond_up, cond_dn).astype(int)
    s = agree(b.bull_bar == "close>open", b.bull_bar == "close<open")
    s += agree(b.formed_first == "low", b.formed_first == "high")
    s += agree(b.delta_q == "D5 buyers", b.delta_q == "D1 sellers")
    s += agree(b.vs_prev == "closed above prev high", b.vs_prev == "closed below prev low")
    s += agree(b.trend4.isin(["up", "strong up"]), b.trend4.isin(["down", "strong down"]))
    s += agree(b.close_vs_vwap > 0.05, b.close_vs_vwap < -0.05)
    s += agree(b.vs_prev_vwap == "above prev VWAP", b.vs_prev_vwap == "below prev VWAP")
    return s


def simulate(args):
    dur, brk = args
    bars = load("dev").sort_values("ts").reset_index(drop=True)
    H, L, C = bars.high.to_numpy(), bars.low.to_numpy(), bars.close.to_numpy()
    T = bars.ts.dt.tz_localize(None).to_numpy("datetime64[m]").astype("int64")
    S = bars.sess.to_numpy("datetime64[D]").astype("int64")
    nb = len(H)
    brk = brk[brk.dur == dur].reset_index(drop=True)
    # break bar position
    end_t = T[brk.i1.to_numpy()]
    bpos = np.searchsorted(T, end_t + brk.brk_min.to_numpy().astype("int64"))
    pw, hold = int(dur * PULLBACK_MULT), int(dur * HOLD_MULT)
    recs = []
    for k in range(len(brk)):
        r = brk.iloc[k]
        b = int(bpos[k]); up = r.side == "up"; w = r.width; sess = S[b]
        sgn = 1.0 if up else -1.0
        lv = {"edge": r.high if up else r.low,
              "q75": r.low + .75 * w if up else r.low + .25 * w,
              "vwap": r.vwap, "mid": r.low + .5 * w, "open": r.open, "close": r.close}
        # pullback window bars: b+1 .. while T <= T[b] + pw and same session
        e_lim = b + 1
        while e_lim < nb and T[e_lim] <= T[b] + pw and S[e_lim] == sess:
            e_lim += 1
        for name, lvl in lv.items():
            if (up and lvl > r.high) or ((not up) and lvl < r.low):
                continue
            seg_l, seg_h = L[b + 1:e_lim], H[b + 1:e_lim]
            fill = (seg_l <= lvl - TICK) if up else (seg_h >= lvl + TICK)
            rec = {"rid": k, "dur": dur, "year": r.year, "side": r.side, "agree": r.agree,
                   "width": w, "level": name, "dist_w": sgn * (lv["edge"] - lvl) / w, "filled": bool(fill.any())}
            if not fill.any():
                recs.append(rec); continue
            e = b + 1 + int(np.argmax(fill))
            ext = (H[b:e].max() if up else L[b:e].min())          # post-break extreme before entry
            rec["min_to_fill"] = T[e] - T[b]
            x_end = e
            while x_end < nb and T[x_end] <= T[e] + hold and S[x_end] == sess:
                x_end += 1
            for cname, (sw, tgt) in CONFIGS.items():
                risk = max(sw * w, TICK)
                stop = lvl - sgn * risk
                target = (ext + sgn * TICK) if tgt == "ext" else lvl + sgn * tgt * risk
                if tgt == "ext" and sgn * (target - lvl) <= 0:
                    continue
                # entry bar reaching the stop -> loss
                if (L[e] <= stop) if up else (H[e] >= stop):
                    pts = -risk; how = "stop"
                else:
                    hh, ll = H[e + 1:x_end], L[e + 1:x_end]
                    hs = (ll <= stop) if up else (hh >= stop)
                    ht = (hh >= target) if up else (ll <= target)
                    i_s = int(np.argmax(hs)) if hs.any() else 10**9
                    i_t = int(np.argmax(ht)) if ht.any() else 10**9
                    if i_s == 10**9 and i_t == 10**9:
                        pts = sgn * (C[x_end - 1] - lvl); how = "time"
                    elif i_s <= i_t:
                        pts = -risk; how = "stop"
                    else:
                        pts = sgn * (target - lvl); how = "target"
                cost = COMM + (TICK if how in ("stop", "time") else 0.0)
                rec[f"{cname}_R"] = pts / risk
                rec[f"{cname}_Rnet"] = (pts - cost) / risk
                rec[f"{cname}_win"] = how == "target"
                rec[f"{cname}_risk"] = risk
            recs.append(rec)
    return pd.DataFrame(recs)


if __name__ == "__main__":
    brk = pd.read_parquet(OUT / "direction_dev_wait1.parquet")
    brk = brk[brk.side.isin(["up", "down"])].reset_index(drop=True)
    brk["agree"] = score(brk)
    with Pool(3) as p:
        parts = p.map(simulate, [(d, brk) for d in (15, 30, 60)])
    tr = pd.concat(parts, ignore_index=True)
    tr.to_parquet(OUT / "continuation_dev.parquet", index=False)
    print("saved", len(tr))
