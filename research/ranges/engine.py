"""
Time-based range break -> retracement engine.

A RANGE is a fixed clock window of `dur` minutes (15 / 30 / 60), aligned to the top of the
hour (optionally shifted by `offset` minutes - used as a null control), never crossing the
17:00-18:00 ET maintenance break. Its levels are known the moment the window closes.

For every range we look FORWARD over `horizon_mult * dur` minutes (same session only) and record:

  break      first 1-min bar whose high > range high (UP) or low < range low (DOWN).
             A bar that pierces both sides is flagged `ambiguous` and excluded from stats.
  retrace    after the break, does price come back to a target level? Checked from the bar
             AFTER the break bar (inside the break bar the order of high/low is unknown).
             Targets are named relative to the broken side
             (q75 = quarter-line nearest the broken edge, q25 = quarter-line nearest the far edge):
             For an UP break:  edge = high, q75 = low + .75w, mid, q25 = low + .25w, far = low
             For a DOWN break: edge = low,  q75 = low + .25w, mid, q25 = low + .75w, far = high
             plus `open` (the range's own open) and `close` (the range's close).
  mae        extension beyond the broken edge BEFORE the target is hit, measured over the break
             bar through the hit bar inclusive (conservative: the hit bar's extreme counts as
             adverse). If never hit, the extension over the whole horizon.
  maeo       same, but excluding the hit bar (optimistic). The truth lies between the two.
  timing     minutes from range close to break, and from break to each target.

All price distances are stored in points; summaries convert to range-widths and ATR units.
ATR is Wilder(14) on the sequence of same-duration ranges, using only PRIOR ranges (shift 1).
"""
import numpy as np
import pandas as pd

TARGETS = ("q75", "mid", "open", "close", "q25", "far")


def build_ranges(bars: pd.DataFrame, dur: int, offset: int = 0, min_cov: float = 0.6) -> pd.DataFrame:
    """One row per range. bars must be sorted, with columns ts, sess, open..close, volume, delta."""
    naive = bars.ts.dt.tz_localize(None)
    mod = naive.dt.hour.to_numpy() * 60 + naive.dt.minute.to_numpy()
    since = (mod - 18 * 60 - offset) % 1440            # minutes since (18:00 + offset)
    slot_idx = since // dur
    # never let a window straddle the 17:00 break: windows that would reach past 17:00 are dropped
    win_end_since = (slot_idx + 1) * dur
    ok = win_end_since <= (23 * 60 - offset)            # 18:00+offset .. 17:00
    b = bars.loc[ok].copy()
    b["slot_idx"] = slot_idx[ok]
    key = b.sess.astype("int64") // 86_400_000_000 * 1000 + b.slot_idx   # unique per (sess, slot)
    b["rid"] = key.to_numpy()
    g = b.groupby("rid", sort=True)
    r = g.agg(sess=("sess", "first"), slot_idx=("slot_idx", "first"),
              start=("ts", "first"), end=("ts", "last"),
              open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
              volume=("volume", "sum"), delta=("delta", "sum"), n=("close", "size"),
              i0=("pos", "first"), i1=("pos", "last"),
              partial_sess=("partial_sess", "first"), roll_adj=("roll_adj", "first"))
    # which extreme formed first (bar index of max high vs min low)
    r["hi_pos"] = g["high"].idxmax().map(b["pos"])
    r["lo_pos"] = g["low"].idxmin().map(b["pos"])
    r = r.reset_index(drop=True)
    r["dur"], r["offset"] = dur, offset
    nominal = (18 * 60 + offset + r.slot_idx * dur) % 1440
    r["slot"] = (nominal // 60).map("{:02d}".format) + ":" + (nominal % 60).map("{:02d}".format)
    r["slot_min"] = nominal
    r["cov"] = r.n / dur
    r = r[r["cov"] >= min_cov].reset_index(drop=True)
    r["width"] = r.high - r.low
    r["mid"] = (r.high + r.low) / 2
    r["close_pos"] = np.where(r.width > 0, (r.close - r.low) / r.width.where(r.width > 0), np.nan)
    r["open_pos"] = np.where(r.width > 0, (r.open - r.low) / r.width.where(r.width > 0), np.nan)
    r["formed_first"] = np.where(r.hi_pos < r.lo_pos, "high", np.where(r.lo_pos < r.hi_pos, "low", "same"))
    r["delta_frac"] = r.delta / r.volume.replace(0, np.nan)
    # Wilder ATR over the range series, back-adjusted closes so rolls don't create fake TR
    adj_c = (r.close + r.roll_adj).shift(1)
    tr = pd.concat([r.width, (r.high + r.roll_adj - adj_c).abs(), (r.low + r.roll_adj - adj_c).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    r["atr_prev"] = atr.shift(1)
    r["w_atr"] = r.width / r.atr_prev
    # width vs the same slot's trailing 20-session median (time-of-day aware), prior sessions only
    r["w_slot"] = r.width / r.groupby("slot").width.transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).median())
    # relation to the previous adjacent range
    prev_adj = (r.sess == r.sess.shift(1)) & (r.slot_idx == r.slot_idx.shift(1) + 1)
    pw = (r.high.shift(1) - r.low.shift(1)).where(prev_adj)
    r["op_prev"] = (r.open - r.low.shift(1)) / pw.replace(0, np.nan)
    return r


def forward_outcomes(bars: pd.DataFrame, r: pd.DataFrame, horizon_mult: float = 4.0) -> pd.DataFrame:
    H = bars.high.to_numpy(); L = bars.low.to_numpy(); C = bars.close.to_numpy()
    T = bars.ts.dt.tz_localize(None).to_numpy("datetime64[m]").astype("int64")   # minutes
    S = bars.sess.to_numpy("datetime64[D]").astype("int64")
    nb = len(H)
    n = len(r)
    dur = int(r.dur.iloc[0])
    hz = int(round(dur * horizon_mult))
    out = {k: np.full(n, np.nan) for k in
           ["brk_min", "ext_max", "fwd_minutes"] + [f"hit_{t}" for t in TARGETS] +
           [f"min_{t}" for t in TARGETS] + [f"mae_{t}" for t in TARGETS] + [f"maeo_{t}" for t in TARGETS]}
    side = np.full(n, "", dtype=object)
    amb = np.zeros(n, bool)
    close_back = np.full(n, np.nan)
    hi = r.high.to_numpy(); lo = r.low.to_numpy(); w = r.width.to_numpy()
    op = r.open.to_numpy(); cl = r.close.to_numpy(); i1 = r.i1.to_numpy()
    end_t = T[i1]; sess = S[i1]
    for k in range(n):
        s = i1[k] + 1
        if s >= nb or w[k] <= 0:
            continue
        # forward slice: same session, ts <= range end + horizon
        e = s
        lim = end_t[k] + hz
        while e < nb and T[e] <= lim and S[e] == sess[k]:
            e += 1
        if e == s:
            continue
        last_t = T[e - 1]
        out["fwd_minutes"][k] = min(hz, last_t - end_t[k])
        h, l = H[s:e], L[s:e]
        up = h > hi[k]; dn = l < lo[k]
        iu = np.argmax(up) if up.any() else -1
        idn = np.argmax(dn) if dn.any() else -1
        if iu < 0 and idn < 0:
            side[k] = "none"
            continue
        if iu >= 0 and idn >= 0 and iu == idn:
            side[k] = "both"; amb[k] = True
            continue
        if idn < 0 or (iu >= 0 and iu < idn):
            sd, bi, edge = "up", iu, hi[k]
            lv = {"q75": lo[k] + .75 * w[k], "mid": lo[k] + .5 * w[k], "open": op[k], "close": cl[k],
                  "q25": lo[k] + .25 * w[k], "far": lo[k]}
        else:
            sd, bi, edge = "down", idn, lo[k]
            lv = {"q75": lo[k] + .25 * w[k], "mid": lo[k] + .5 * w[k], "open": op[k], "close": cl[k],
                  "q25": lo[k] + .75 * w[k], "far": hi[k]}
        side[k] = sd
        out["brk_min"][k] = T[s + bi] - end_t[k]
        hh, ll, cc = h[bi:], l[bi:], C[s + bi:e]
        ext = (hh - edge) if sd == "up" else (edge - ll)          # extension beyond edge per bar
        run_ext = np.maximum.accumulate(ext)
        out["ext_max"][k] = run_ext[-1]
        # close back inside the range (from the break bar on)
        cb = (cc <= edge) if sd == "up" else (cc >= edge)
        close_back[k] = 1.0 if cb.any() else 0.0
        post_h, post_l = hh[1:], ll[1:]
        for t, lvl in lv.items():
            # target must be on the retrace side of the edge; open/close can sit anywhere inside
            hit = (post_l <= lvl) if sd == "up" else (post_h >= lvl)
            if hit.any():
                j = int(np.argmax(hit)) + 1                    # index into hh/ll (0 = break bar)
                out[f"hit_{t}"][k] = 1.0
                out[f"min_{t}"][k] = T[s + bi + j] - T[s + bi]
                out[f"mae_{t}"][k] = run_ext[j]                 # includes hit bar's extreme (conservative)
                out[f"maeo_{t}"][k] = run_ext[j - 1]            # excludes it (optimistic)
            else:
                out[f"hit_{t}"][k] = 0.0
                out[f"mae_{t}"][k] = run_ext[-1]
                out[f"maeo_{t}"][k] = run_ext[-1]
    res = r.copy()
    for c, v in out.items():
        res[c] = v
    res["side"], res["ambiguous"], res["close_back"] = side, amb, close_back
    res["horizon"] = hz
    res["full_fwd"] = res.fwd_minutes >= 0.9 * hz
    res["broke"] = res.side.isin(["up", "down"])
    return res


def run(bars: pd.DataFrame, durs=(15, 30, 60), offsets=(0,), horizon_mult=4.0) -> pd.DataFrame:
    bars = bars.sort_values("ts").reset_index(drop=True)
    bars["pos"] = np.arange(len(bars))
    frames = []
    for d in durs:
        for o in offsets:
            r = build_ranges(bars, d, o)
            frames.append(forward_outcomes(bars, r, horizon_mult))
    return pd.concat(frames, ignore_index=True)
