"""
Exploratory continuation study, round 2 (DEV split only).

Same setup as continuation.py (first break within 1x duration sets direction D; pullback to a
level inside the range within 1x duration after the break), extended with:

ENTRIES at level lvl (UP break shown; DOWN mirrored)
  E0 limit      buy limit at lvl, filled if low <= lvl - 1 tick. Entry price lvl.
                If the fill bar also reaches the stop -> loss.
  E1 rejection  first bar in the window with low <= lvl AND close > lvl. Buy at that close + 1 tick.
  E2 delta      after the first touch (low <= lvl), first bar within 5 bars that closes > lvl,
                closes > its open and has delta > 0. Buy at that close + 1 tick.
  For E1/E2: if the stop level trades before confirmation, no trade.
  Stop for all entries: lvl - 0.5 x range width.  risk = entry - stop.

TARGETS
  R2    entry + 2 x risk
  EXT   post-break extreme (highest high from the break bar to entry) + 1 tick
  NEXT  nearest structural level above entry + 1R: previous adjacent range high, session high so
        far, prior session's RTH high, overnight high (only once RTH has started).
        If none qualifies the trade is not taken.

CONTEXT (known at the range close)
  htf60 / htf240   close vs the close 60 / 240 minutes earlier (same session)
  vs_sess_open     close vs the 18:00 session open
  vs_rth_open      close vs the 09:30 open (ranges ending in RTH only)
  htf_agree        how many of htf60, htf240, vs_sess_open point in D (0-3)
  block            time-of-day block of the range END

Exits: stop / target / timeout after 4 x duration (or session end). Same bar stop+target -> loss.
Costs: $4.50 RT (0.225 pts); 1 tick slippage on market entries (E1/E2) and stop/timeout exits.
"""
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from research.scripts.load import load
from research.ranges.continuation import score

OUT = Path(__file__).resolve().parents[1] / "output"
TICK, COMM = 0.25, 0.225
LEVELS = ("q75", "vwap", "mid", "open")
ENTRIES = ("E0", "E1", "E2")
TARGETS = ("R2", "EXT", "NEXT")
BIG = 10**9


def block_of(m):          # minute-of-day of the range END
    h = m / 60
    if h >= 18 or h < 2: return "1 Asia 18-02"
    if h < 8: return "2 London 02-08"
    if h < 9.5: return "3 Pre-RTH 08-0930"
    if h < 11: return "4 RTH open 0930-11"
    if h < 14: return "5 RTH mid 11-14"
    if h < 16: return "6 RTH late 14-16"
    return "7 Post 16-17"


def prep_bars():
    b = load("dev").sort_values("ts").reset_index(drop=True)
    naive = b.ts.dt.tz_localize(None)
    b["mod"] = naive.dt.hour * 60 + naive.dt.minute
    b["sess_hi"] = b.groupby("sess").high.cummax()
    b["sess_lo"] = b.groupby("sess").low.cummin()
    b["sess_open"] = b.groupby("sess").open.transform("first")
    rth = (b["mod"] >= 570) & (b["mod"] < 960)
    on = b["mod"] >= 1080
    on = on | (b["mod"] < 570)
    b["on_hi"] = b.high.where(on).groupby(b.sess).cummax().groupby(b.sess).ffill()
    b["on_lo"] = b.low.where(on).groupby(b.sess).cummin().groupby(b.sess).ffill()
    # 09:30 open of the session
    o930 = b[b["mod"] == 570].groupby("sess").open.first()
    b["rth_open"] = b.sess.map(o930)
    # prior session RTH high/low, carried in back-adjusted terms then converted to this bar's contract
    r = b[rth].assign(ah=lambda x: x.high + x.roll_adj, al=lambda x: x.low + x.roll_adj)
    ph = r.groupby("sess").ah.max().shift(1); pl = r.groupby("sess").al.min().shift(1)
    b["prev_rth_hi"] = b.sess.map(ph) - b.roll_adj
    b["prev_rth_lo"] = b.sess.map(pl) - b.roll_adj
    return b


def simulate(args):
    dur, brk = args
    bars = prep_bars()
    H, L, C, O = (bars[c].to_numpy() for c in ("high", "low", "close", "open"))
    D = bars.delta.to_numpy()
    T = bars.ts.dt.tz_localize(None).to_numpy("datetime64[m]").astype("int64")
    S = bars.sess.to_numpy("datetime64[D]").astype("int64")
    MOD = bars["mod"].to_numpy()
    SH, SL = bars.sess_hi.to_numpy(), bars.sess_lo.to_numpy()
    ONH, ONL = bars.on_hi.to_numpy(), bars.on_lo.to_numpy()
    PRH, PRL = bars.prev_rth_hi.to_numpy(), bars.prev_rth_lo.to_numpy()
    SO, RO = bars.sess_open.to_numpy(), bars.rth_open.to_numpy()
    nb = len(H)
    brk = brk[brk.dur == dur].reset_index(drop=True)
    i1 = brk.i1.to_numpy(); i0 = brk.i0.to_numpy()
    end_t = T[i1]
    bpos = np.searchsorted(T, end_t + brk.brk_min.to_numpy().astype("int64"))
    p60 = np.searchsorted(T, end_t - 60, side="right") - 1
    p240 = np.searchsorted(T, end_t - 240, side="right") - 1
    pprev0 = np.searchsorted(T, T[i0] - dur)            # previous adjacent window [start-dur, start)
    pw, hold = dur, 4 * dur
    recs = []

    def run_exit(start, entry, stop, target, up, check_first):
        """returns (pts, how). start = first bar to evaluate; check_first: entry bar may hit stop."""
        sg = 1.0 if up else -1.0
        if check_first and ((L[start] <= stop) if up else (H[start] >= stop)):
            return -(abs(entry - stop)), "stop"
        s0 = start + 1 if check_first else start
        x = s0
        lim = T[start] + hold; ss = S[start]
        while x < nb and T[x] <= lim and S[x] == ss:
            x += 1
        if x == s0:
            return sg * (C[start] - entry), "time"
        hh, ll = H[s0:x], L[s0:x]
        hs = (ll <= stop) if up else (hh >= stop)
        ht = (hh >= target) if up else (ll <= target)
        i_s = int(np.argmax(hs)) if hs.any() else BIG
        i_t = int(np.argmax(ht)) if ht.any() else BIG
        if i_s == BIG and i_t == BIG:
            return sg * (C[x - 1] - entry), "time"
        if i_s <= i_t:
            return -abs(entry - stop), "stop"
        return sg * (target - entry), "target"

    for k in range(len(brk)):
        r = brk.iloc[k]
        b = int(bpos[k]); up = r.side == "up"; sg = 1.0 if up else -1.0; w = r.width; ss = S[b]
        ie = i1[k]; cnow = C[ie]
        def dir_(x):
            return np.nan if np.isnan(x) else np.sign(cnow - x)
        h60 = dir_(C[p60[k]]) if p60[k] >= 0 and S[p60[k]] == S[ie] else np.nan
        h240 = dir_(C[p240[k]]) if p240[k] >= 0 and S[p240[k]] == S[ie] else np.nan
        vso = dir_(SO[ie])
        end_mod = (MOD[ie] + 1) % 1440
        in_rth = 570 <= end_mod <= 960 and MOD[ie] >= 570
        vro = dir_(RO[ie]) if in_rth and not np.isnan(RO[ie]) else np.nan
        htf_agree = int(np.nansum([x == sg for x in (h60, h240, vso)]))
        pp0 = pprev0[k]
        prev_ok = pp0 < i0[k] and S[pp0] == S[i0[k]]
        prev_hi = H[pp0:i0[k]].max() if prev_ok else np.nan
        prev_lo = L[pp0:i0[k]].min() if prev_ok else np.nan
        base = {"rid": k, "dur": dur, "year": r.year, "side": r.side, "agree": r.agree, "width": w,
                "block": block_of(end_mod), "htf60": h60 * sg, "htf240": h240 * sg, "vs_sess_open": vso * sg,
                "vs_rth_open": vro * sg, "htf_agree": htf_agree}
        e_lim = b + 1
        while e_lim < nb and T[e_lim] <= T[b] + pw and S[e_lim] == ss:
            e_lim += 1
        lv = {"q75": r.low + .75 * w if up else r.low + .25 * w, "vwap": r.vwap,
              "mid": r.low + .5 * w, "open": r.open}
        for name, lvl in lv.items():
            if (up and lvl >= r.high) or ((not up) and lvl <= r.low):
                continue
            stop = lvl - sg * 0.5 * w
            seg_h, seg_l = H[b + 1:e_lim], L[b + 1:e_lim]
            touch = (seg_l <= lvl) if up else (seg_h >= lvl)
            stopped = (seg_l <= stop) if up else (seg_h >= stop)
            first_stop = int(np.argmax(stopped)) if stopped.any() else BIG
            entries = {}
            # E0 limit with trade-through
            thr = (seg_l <= lvl - TICK) if up else (seg_h >= lvl + TICK)
            if thr.any():
                j = b + 1 + int(np.argmax(thr)); entries["E0"] = (j, lvl, True)
            # E1 rejection bar: touch and close back on the D side, before any stop
            rej = touch & (((C[b + 1:e_lim] > lvl) if up else (C[b + 1:e_lim] < lvl)))
            if rej.any():
                j = int(np.argmax(rej))
                if j < first_stop:      # the stop level must not have traded before or on the rejection bar
                    entries["E1"] = (b + 1 + j + 1, C[b + 1 + j] + sg * TICK, False)
            # E2 delta confirmation within 5 bars of the first touch
            if touch.any():
                t0 = b + 1 + int(np.argmax(touch))
                for j in range(t0, min(t0 + 6, nb)):
                    if S[j] != ss: break
                    if (L[j] <= stop) if up else (H[j] >= stop): break
                    ok = (C[j] > lvl and C[j] > O[j] and D[j] > 0) if up else (C[j] < lvl and C[j] < O[j] and D[j] < 0)
                    if ok:
                        entries["E2"] = (j + 1, C[j] + sg * TICK, False); break
            rec0 = dict(base, level=name, dist_w=sg * ((r.high if up else r.low) - lvl) / w)
            for en in ENTRIES:
                rec = dict(rec0, entry=en, filled=en in entries)
                if en in entries:
                    start, px, chk = entries[en]
                    if start >= nb or S[min(start, nb - 1)] != ss:
                        rec["filled"] = False; recs.append(rec); continue
                    risk = abs(px - stop)
                    ebar = start if chk else start - 1
                    ext = (H[b:ebar + 1].max() if up else L[b:ebar + 1].min())
                    cands = [prev_hi if up else prev_lo, SH[ebar] if up else SL[ebar],
                             PRH[ebar] if up else PRL[ebar]]
                    if MOD[ebar] >= 570 and MOD[ebar] < 960:
                        cands.append(ONH[ebar] if up else ONL[ebar])
                    cands = [c for c in cands if not np.isnan(c) and sg * (c - px) >= risk]
                    tg = {"R2": px + sg * 2 * risk,
                          "EXT": ext + sg * TICK if sg * (ext + sg * TICK - px) > 0 else np.nan,
                          "NEXT": (min(cands) if up else max(cands)) if cands else np.nan}
                    rec["risk"] = risk
                    for tn, tv in tg.items():
                        if np.isnan(tv):
                            continue
                        pts, how = run_exit(start, px, stop, tv, up, chk)
                        cost = COMM + (TICK if how in ("stop", "time") else 0.0)
                        rec[f"{tn}_R"] = pts / risk
                        rec[f"{tn}_Rnet"] = (pts - cost) / risk
                        rec[f"{tn}_win"] = how == "target"
                        rec[f"{tn}_rr"] = sg * (tv - px) / risk
                recs.append(rec)
    return pd.DataFrame(recs)


if __name__ == "__main__":
    brk = pd.read_parquet(OUT / "direction_dev_wait1.parquet")
    brk = brk[brk.side.isin(["up", "down"])].reset_index(drop=True)
    brk["agree"] = score(brk)
    with Pool(3) as p:
        parts = p.map(simulate, [(d, brk) for d in (15, 30, 60)])
    tr = pd.concat(parts, ignore_index=True)
    tr.to_parquet(OUT / "continuation2_dev.parquet", index=False)
    print("saved", len(tr))
