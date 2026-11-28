"""Phase 1 datasets: one row per decision, features known at decision time, labels from the future.

B1  level race            decisions 09:30 (window -> 10:59) and 11:00 (window -> 15:59)
                          nearest level above vs below (PLAN §3 levels, > 0.03 ATR away), fixed at decision time.
                          y = 1 above hit first, 0 below first; ties (same bar) and neither-hit kept with y = NaN.
B2  day type at 10:30     ext = max(RTH high - IB high, IB low - RTH low) / IB width
                          y = 1 trend (ext > 1.0), 0 rotation (ext <= 0.5), NaN in between.
B3  hourly, 10:00-15:00   y_far  = hour takes the FAR extreme of the previous hour
                          y_near = hour takes the NEAR extreme (near = closer to the hour's open)

Sign conventions: B1 features are raw (positive = price high / up). B3 and B2 features are folded so that they
do not depend on direction (B3 oriented toward the near side, B2 absolute values).
"""
import numpy as np
import pandas as pd

from engine.levels import LEVELS

TICK = 0.25
MIN_LVL_ATR = 0.03
DELTA_FEATS = ["on_delta", "d_open_delta", "ib_delta", "h_delta"]


def value_area(lo, hi, vol, pct=0.70):
    """Volume spread evenly over every tick of each bar; 70% grown from the POC (as research/stats/value_area.py)."""
    lo_t = np.round(lo / TICK).astype(np.int64)
    hi_t = np.round(hi / TICK).astype(np.int64)
    base = lo_t.min()
    n = hi_t.max() - base + 1
    per = vol / (hi_t - lo_t + 1)
    diff = np.zeros(n + 1)
    np.add.at(diff, lo_t - base, per)
    np.add.at(diff, hi_t - base + 1, -per)
    prof = np.cumsum(diff)[:n]
    poc = int(np.argmax(prof))
    tot = prof.sum()
    a = b = poc
    acc = prof[poc]
    while acc < pct * tot:
        up = prof[b + 1] if b + 1 < n else -1
        dn = prof[a - 1] if a - 1 >= 0 else -1
        if up >= dn:
            b += 1; acc += up
        else:
            a -= 1; acc += dn
    return (base + a) * TICK, (base + b) * TICK, (base + poc) * TICK


def _race(h, l, i0, i1, up, dn):
    """First touch of `up` (high >= up) vs `dn` (low <= dn) in bars i0..i1-1. 1 / 0 / NaN (tie or neither)."""
    hu = h[i0:i1] >= up
    ld = l[i0:i1] <= dn
    a = np.argmax(hu) if hu.any() else 10 ** 9
    b = np.argmax(ld) if ld.any() else 10 ** 9
    if a == b:
        return np.nan, "tie" if a < 10 ** 9 else "neither"
    return (1.0, "up") if a < b else (0.0, "dn")


def _nearest(lv_row, p, atr):
    vals = {k: lv_row[k] for k in LEVELS if np.isfinite(lv_row[k])}
    above = {k: v for k, v in vals.items() if v > p + MIN_LVL_ATR * atr}
    below = {k: v for k, v in vals.items() if v < p - MIN_LVL_ATR * atr}
    if not above or not below:
        return None
    ku = min(above, key=lambda k: above[k] - p)
    kd = max(below, key=lambda k: below[k] - p)
    return ku, above[ku], kd, below[kd]


def build(d, L):
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = d.volume.to_numpy(float)
    dl = d.delta.to_numpy(float)
    hm = L.hm.to_numpy()
    Lv = L[LEVELS].to_numpy(float)
    lvcol = {k: i for i, k in enumerate(LEVELS)}
    atr_a = L.atr.to_numpy(float)
    sess = d.sess.to_numpy()
    split = d.split.to_numpy()
    partial = d.partial.to_numpy()
    pd_ok = d.pd_ok.to_numpy()
    starts = np.r_[0, np.flatnonzero(sess[1:] != sess[:-1]) + 1]
    ends = np.r_[starts[1:], len(d)]

    days, races, hours = [], [], []
    prev = None                      # prior session RTH summary
    rng_hist = []
    week_key, week_open, wk_hi, wk_lo = None, np.nan, -np.inf, np.inf

    def row_at(i):
        return {k: Lv[i, j] for k, j in lvcol.items()}

    for a, b in zip(starts, ends):
        s = pd.Timestamp(sess[a])
        hs = hm[a:b]
        wk = s - pd.Timedelta(days=s.weekday())
        if wk != week_key:
            week_key, week_open, wk_hi, wk_lo = wk, o[a], -np.inf, np.inf

        def idx(minute):
            k = np.flatnonzero(hs == minute)
            return a + k[0] if len(k) else None

        rth = np.flatnonzero((hs >= 570) & (hs < 960)) + a
        i930, i1030, i1100, i800 = idx(570), idx(630), idx(660), idx(480)
        ok = (not partial[a]) and len(rth) > 300 and None not in (i930, i1030, i1100, i800)
        atr = atr_a[a]
        if ok and prev is not None and np.isfinite(atr):
            p930 = o[i930]
            on = slice(a, i930)
            on_h, on_l = h[on].max(), l[on].min()
            pre8 = slice(a, i800)
            r8h, r8l = h[pre8].max(), l[pre8].min()
            wk_hi_now, wk_lo_now = max(wk_hi, on_h), min(wk_lo, on_l)
            lv930 = row_at(i930)
            ib = slice(i930, i1030)
            ib_h, ib_l = h[ib].max(), l[ib].min()
            ibw = ib_h - ib_l
            orb = slice(i930, i930 + 15)
            rth_h, rth_l = h[rth].max(), l[rth].min()
            rest = slice(i1030, rth[-1] + 1)
            ext = max(h[rest].max() - ib_h, ib_l - l[rest].min(), 0) / ibw
            pd_rng = prev["h"] - prev["l"]
            va_w = max(prev["vah"] - prev["val"], TICK)
            hist = np.array(rng_hist[-5:])
            f = dict(
                sess=s, split=split[a], atr=atr,
                open_pos_pd=(p930 - prev["l"]) / pd_rng,
                gap_atr=(p930 - prev["c"]) / atr,
                pos8=(o[i800] - r8l) / max(r8h - r8l, TICK),
                on_range_atr=(on_h - on_l) / atr,
                va_pos=(p930 - prev["val"]) / va_w,
                vwap_day_dist=(p930 - lv930["vwap_day"]) / atr,
                vwap_wk_dist=(p930 - lv930["vwap_wk"]) / atr,
                day_up_fill=(on_h - o[a]) / atr, day_dn_fill=(o[a] - on_l) / atr,
                wk_up_fill=(wk_hi_now - week_open) / (atr * np.sqrt(5)),
                wk_dn_fill=(week_open - wk_lo_now) / (atr * np.sqrt(5)),
                prior_body=(prev["c"] - prev["o"]) / max(pd_rng, TICK),
                prior_range_atr=pd_rng / atr,
                atr_ratio=hist.mean() / atr if len(hist) == 5 else np.nan,
                dow=s.weekday(),
                on_delta=np.nansum(dl[on]) / max(v[on].sum(), 1),
                ib_w_atr=ibw / atr, orb15_w_atr=(h[orb].max() - l[orb].min()) / atr,
                ib_close_pos=(c[i1030 - 1] - ib_l) / max(ibw, TICK),
                vwap_rth_dist_1030=(o[i1030] - Lv[i1030, lvcol["vwap_rth"]]) / atr,
                ib_delta=np.nansum(dl[ib]) / max(v[ib].sum(), 1),
                rth_ret_atr=(c[rth[-1]] - p930) / atr, rth_range_atr=(rth_h - rth_l) / atr,
                ib_ext=ext,
                y_b2=1.0 if ext > 1.0 else (0.0 if ext <= 0.5 else np.nan),
            )
            if not pd_ok[a]:                     # lockbox roll weeks: prior-day prices are another contract
                f.update(open_pos_pd=np.nan, gap_atr=np.nan, va_pos=np.nan)
            days.append(f)

            # ---- B1 races ------------------------------------------------------------------------------------
            for tag, i0, i1 in (("0930", i930, i1100), ("1100", i1100, rth[-1] + 1)):
                p = o[i0]
                nr = _nearest(row_at(i0), p, atr)
                if nr is None:
                    continue
                ku, up, kd, dn = nr
                y, how = _race(h, l, i0, i1, up, dn)
                so_far = slice(i930, i0) if i0 > i930 else slice(i930, i930 + 1)
                r = dict(sess=s, split=split[a], t=tag, y=y, outcome=how, lvl_up=ku, lvl_dn=kd,
                         d_up=(up - p) / atr, d_dn=(p - dn) / atr,
                         geom=np.log((p - dn) / (up - p)),
                         rw_p=(p - dn) / ((up - p) + (p - dn)))
                if tag == "1100":
                    r.update(ib_pos=(p - ib_l) / max(ibw, TICK),
                             vwap_rth_dist=(p - Lv[i0, lvcol["vwap_rth"]]) / atr,
                             vwap_day_dist_now=(p - Lv[i0, lvcol["vwap_day"]]) / atr,
                             move_open=(p - p930) / atr,
                             range_so_far=(h[so_far].max() - l[so_far].min()) / atr,
                             d_open_delta=np.nansum(dl[so_far]) / max(v[so_far].sum(), 1))
                races.append(r)

            # ---- B3 hours ------------------------------------------------------------------------------------
            for H in range(10, 16):
                i0 = idx(H * 60)
                if i0 is None:
                    continue
                i1 = min(i0 + 60, rth[-1] + 1)
                phh, phl = Lv[i0, lvcol["ph_h"]], Lv[i0, lvcol["ph_l"]]
                w = phh - phl
                if not np.isfinite(w) or w <= 0:
                    continue
                p = o[i0]
                pos = (p - phl) / w
                near_hi = pos >= 0.5
                sgn = 1.0 if near_hi else -1.0
                near, far = (phh, phl) if near_hi else (phl, phh)
                hh, ll = h[i0:i1].max(), l[i0:i1].min()
                y_near = float(hh >= near) if near_hi else float(ll <= near)
                y_far = float(ll <= far) if near_hi else float(hh >= far)
                prevh = slice(i0 - 60, i0)
                sofar = slice(i930, i0)
                hours.append(dict(
                    sess=s, split=split[a], hour=H, y_far=y_far, y_near=y_near,
                    pos_fold=pos if near_hi else 1 - pos,
                    ph_range_atr=w / atr,
                    prev_close_fold=((c[i0 - 1] - phl) / w) if near_hi else ((phh - c[i0 - 1]) / w),
                    range_so_far=(h[sofar].max() - l[sofar].min()) / atr,
                    vwap_rth_or=sgn * (p - Lv[i0, lvcol["vwap_rth"]]) / atr,
                    vwap_day_or=sgn * (p - Lv[i0, lvcol["vwap_day"]]) / atr,
                    move_open_or=sgn * (p - p930) / atr,
                    ib_w_atr=(ibw / atr) if H >= 11 else np.nan,
                    h_delta=sgn * np.nansum(dl[prevh]) / max(v[prevh].sum(), 1),
                    dow=s.weekday(),
                ))

        # ---- carry state ---------------------------------------------------------------------------------------
        if len(rth):
            val, vah, poc = value_area(l[rth], h[rth], v[rth])
            prev = dict(o=o[rth[0]], h=h[rth].max(), l=l[rth].min(), c=c[rth[-1]], val=val, vah=vah, poc=poc)
            if not partial[a]:
                rng_hist.append(prev["h"] - prev["l"])
        wk_hi, wk_lo = max(wk_hi, h[a:b].max()), min(wk_lo, l[a:b].min())

    return pd.DataFrame(days), pd.DataFrame(races), pd.DataFrame(hours)
