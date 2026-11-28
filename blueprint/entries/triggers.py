"""Phase 2b triggers (PLAN §13). Each returns (long_event, short_event) boolean arrays over the 1m bars: True on the
1m bar whose CLOSE confirms the trigger (entry is the next bar's open)."""
import numpy as np
import pandas as pd

from engine import fvg


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def wilder(x, n):
    return pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def stoch_k(h, l, c, n=14, smooth=3):
    hh = pd.Series(h).rolling(n).max().to_numpy()
    ll = pd.Series(l).rolling(n).min().to_numpy()
    raw = 100 * (c - ll) / np.where(hh - ll > 0, hh - ll, np.nan)
    return pd.Series(raw).rolling(smooth).mean().to_numpy()


def rsi(c, n=2):
    d = np.diff(c, prepend=c[0])
    up, dn = wilder(np.maximum(d, 0), n), wilder(np.maximum(-d, 0), n)
    return 100 - 100 / (1 + up / np.where(dn > 0, dn, np.nan))


def supertrend_dir(h, l, c, n=10, mult=3.0):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])))
    atr = wilder(tr, n)
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    fu, fl = ub.copy(), lb.copy()
    d = np.ones(len(c), int)
    for i in range(1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if d[i - 1] == -1 and c[i] > fu[i - 1]:
            d[i] = 1
        elif d[i - 1] == 1 and c[i] < fl[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
    return d


def cross_up(x, lvl):
    p = np.r_[np.nan, x[:-1]]
    return (p < lvl) & (x >= lvl)


def cross_dn(x, lvl):
    p = np.r_[np.nan, x[:-1]]
    return (p > lvl) & (x <= lvl)


def _map_tf(ctx, b, ev_l, ev_s):
    """Events on resampled bars -> the 1m bar whose close equals the resampled bar's close."""
    idx = ctx.B.index_at(b.t_close.to_numpy()) - 1
    ok = (idx >= 0) & (idx < ctx.n)
    L, S = np.zeros(ctx.n, bool), np.zeros(ctx.n, bool)
    L[idx[ok & ev_l]] = True
    S[idx[ok & ev_s]] = True
    return L, S


def build_all(ctx):
    o, h, l, c = ctx.o, ctx.h, ctx.l, ctx.c
    T = {}
    T["T0 none"] = None
    k = stoch_k(h, l, c)
    T["T1 stoch 1m"] = (cross_up(k, 20), cross_dn(k, 80))
    b5 = fvg.resample(ctx.d, 5)
    k5 = stoch_k(b5.high.to_numpy(), b5.low.to_numpy(), b5.close.to_numpy())
    T["T2 stoch 5m"] = _map_tf(ctx, b5, cross_up(k5, 20), cross_dn(k5, 80))
    e9, e21 = ema(c, 9), ema(c, 21)
    pc, pe9 = np.r_[np.nan, c[:-1]], np.r_[np.nan, e9[:-1]]
    T["T3 EMA9 reclaim"] = ((pc <= pe9) & (c > e9), (pc >= pe9) & (c < e9))
    diff = e9 - e21
    T["T4 EMA9/21 cross"] = (cross_up(diff, 0), cross_dn(diff, 0))
    st = supertrend_dir(h, l, c)
    pst = np.r_[st[0], st[:-1]]
    T["T5 supertrend 1m"] = ((pst == -1) & (st == 1), (pst == 1) & (st == -1))
    b3 = fvg.resample(ctx.d, 3)
    st3 = supertrend_dir(b3.high.to_numpy(), b3.low.to_numpy(), b3.close.to_numpy())
    pst3 = np.r_[st3[0], st3[:-1]]
    T["T6 supertrend 3m"] = _map_tf(ctx, b3, (pst3 == -1) & (st3 == 1), (pst3 == 1) & (st3 == -1))
    r2 = rsi(c, 2)
    T["T7 RSI2"] = (cross_up(r2, 10), cross_dn(r2, 90))
    iv = ctx.ifvgs[1]
    ci = ctx.B.index_at(iv.t_known.to_numpy()) - 1
    L, S = np.zeros(ctx.n, bool), np.zeros(ctx.n, bool)
    L[ci[iv.dir.to_numpy() == 1]] = True
    S[ci[iv.dir.to_numpy() == -1]] = True
    T["T8 iFVG 1m"] = (L, S)
    from entries.signals import e1
    conf = e1(ctx, 1, "5", "market", 1e6)          # 1m iFVG inside a live aligned 5m FVG (level filter off)
    L, S = np.zeros(ctx.n, bool), np.zeros(ctx.n, bool)
    L[conf.e[conf.side == 1].to_numpy() - 1] = True
    S[conf.e[conf.side == -1].to_numpy() - 1] = True
    T["T9 iFVG 1m in 5m"] = (L, S)
    po, pcl = np.r_[np.nan, o[:-1]], np.r_[np.nan, c[:-1]]
    rng = h - l
    body = np.abs(c - o)
    eng_l = (c > o) & (pcl < po) & (o <= pcl) & (c >= po)
    eng_s = (c < o) & (pcl > po) & (o >= pcl) & (c <= po)
    pin_l = (rng > 0) & (np.minimum(o, c) - l >= 2 * body) & (np.minimum(o, c) - l >= 0.6 * rng) & (c >= h - rng / 3)
    pin_s = (rng > 0) & (h - np.maximum(o, c) >= 2 * body) & (h - np.maximum(o, c) >= 0.6 * rng) & (c <= l + rng / 3)
    T["T10 engulf/pin"] = (eng_l | pin_l, eng_s | pin_s)
    return T
