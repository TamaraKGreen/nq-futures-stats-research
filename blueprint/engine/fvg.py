"""Fair value gaps and their inversions (iFVG) on 1m / 5m / 15m bars.

FVG (3-bar gap, all three bars in the same session):
    bullish  low[i]  > high[i-2]   zone = (high[i-2], low[i])
    bearish  high[i] < low[i-2]    zone = (high[i],   low[i-2])
    The FVG exists from the CLOSE of bar i.

Inversion (iFVG): a later bar on the same timeframe goes through the far side of a live FVG
    mode='close'  the bar CLOSES beyond the far side       (main definition)
    mode='wick'   the bar TRADES beyond the far side         (variant)
    bearish FVG inverted upward  -> bullish iFVG (dir +1, a long signal)
    bullish FVG inverted downward -> bearish iFVG (dir -1, a short signal)
    The iFVG is known at the CLOSE of the inversion bar = open of the next 1m bar (`t_known`).
FVGs die when inverted, after `expiry` bars of their timeframe, or at the session end.
"""
import numpy as np
import pandas as pd

EXPIRY = {1: 60, 5: 24, 15: 16}


def resample(df, minutes):
    """Clock-aligned bars inside each 18:00 session. t_close = bar open + minutes (when the bar is complete)."""
    if minutes == 1:
        b = df[["ts", "sess", "open", "high", "low", "close"]].copy()
        b["t_close"] = b.ts + pd.Timedelta(minutes=1)
        return b.reset_index(drop=True)
    key = df.ts.dt.floor(f"{minutes}min")
    g = df.groupby([df.sess, key], sort=True)
    b = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    b = b.reset_index().rename(columns={"level_1": "ts"})
    b.columns = ["sess", "ts", "open", "high", "low", "close"]
    b["t_close"] = b.ts + pd.Timedelta(minutes=minutes)
    return b


def detect(bars, minutes, mode="close", min_size=0.0, expiry=None):
    """Return (fvgs, ifvgs) DataFrames.

    fvgs : one row per FVG  - dir, lo, hi, t_created (known time), t_end, end ('inv'|'exp'|'sess')
    ifvgs: one row per inversion - dir (+1 long / -1 short), lo, hi (the gap zone), t_known, t_fvg, size
    `min_size` may be a scalar or an array aligned with `bars` (e.g. 0.02 * ATR).
    """
    expiry = expiry or EXPIRY[minutes]
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    s = bars.sess.to_numpy()
    tc = bars.t_close.to_numpy()
    ms = np.broadcast_to(np.asarray(min_size, float), (len(bars),))
    n = len(bars)
    live = []            # [dir, lo, hi, i_created]
    fv, iv = [], []
    for j in range(n):
        # 1) inversions / expiry for FVGs created before bar j
        if live:
            keep = []
            for f in live:
                d, lo, hi, i0 = f
                if s[i0] != s[j]:
                    fv.append((d, lo, hi, tc[i0], tc[j - 1], "sess"))
                    continue
                if j - i0 > expiry:
                    fv.append((d, lo, hi, tc[i0], tc[i0 + expiry], "exp"))
                    continue
                if d == 1:   # bullish gap -> inverted when price goes below its bottom
                    hit = (c[j] < lo) if mode == "close" else (l[j] < lo)
                else:        # bearish gap -> inverted when price goes above its top
                    hit = (c[j] > hi) if mode == "close" else (h[j] > hi)
                if hit:
                    iv.append((-d, lo, hi, tc[j], tc[i0], hi - lo))
                    fv.append((d, lo, hi, tc[i0], tc[j], "inv"))
                else:
                    keep.append(f)
            live = keep
        # 2) new FVG completed by bar j
        if j >= 2 and s[j] == s[j - 2]:
            if l[j] - h[j - 2] > max(ms[j], 0.0) and l[j] > h[j - 2]:
                live.append([1, h[j - 2], l[j], j])
            elif l[j - 2] - h[j] > max(ms[j], 0.0) and h[j] < l[j - 2]:
                live.append([-1, h[j], l[j - 2], j])
    for d, lo, hi, i0 in live:
        fv.append((d, lo, hi, tc[i0], tc[n - 1], "sess"))
    fvgs = pd.DataFrame(fv, columns=["dir", "lo", "hi", "t_created", "t_end", "end"])
    ifvgs = pd.DataFrame(iv, columns=["dir", "lo", "hi", "t_known", "t_fvg", "size"])
    fvgs["tf"] = minutes
    ifvgs["tf"] = minutes
    return fvgs.sort_values("t_created").reset_index(drop=True), ifvgs.sort_values("t_known").reset_index(drop=True)


def live_at(fvgs, t):
    """FVGs alive at time t (created at or before t, not yet ended)."""
    return fvgs[(fvgs.t_created <= t) & (fvgs.t_end > t)]
