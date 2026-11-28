"""Phase 2 candidate generators (PLAN §12). Each returns a DataFrame of candidates for engine.sim.run:
    e (entry bar index), side, stop, r_mult or target, limit/window (limit entries), plus descriptive columns."""
import numpy as np
import pandas as pd

from engine import fvg
from engine.data import TICK
from engine.levels import LEVELS
from engine.sim import Bars

WIN0, WIN1 = 9 * 60 + 31, 15 * 60 + 44
FLOOR_ATR = 0.02
R_MULT = 1.5


class Ctx:
    def __init__(self, d, L):
        self.d, self.L = d, L
        self.hm = L.hm.to_numpy()
        self.B = Bars(d, self.hm)
        self.o, self.h, self.l, self.c = self.B.o, self.B.h, self.B.l, self.B.c
        self.v = d.volume.to_numpy(float)
        self.delta = d.delta.to_numpy(float)
        self.atr = L.atr.to_numpy(float)
        self.Lv = L[LEVELS].to_numpy(float)
        self.sess = d.sess.to_numpy()
        self.ts = self.B.ts
        self.n = len(d)
        self.bars, self.fvgs, self.ifvgs = {}, {}, {}
        for tf in (1, 5, 15):
            b = fvg.resample(d, tf)
            f, iv = fvg.detect(b, tf)
            self.bars[tf], self.fvgs[tf], self.ifvgs[tf] = b, f, iv

    def entry_index(self, t_known):
        e = self.B.index_at(t_known)
        ok = e < self.n
        e = np.where(ok, e, 0)
        ok &= self.ts[e] == np.asarray(t_known, dtype=self.ts.dtype)
        ok &= (self.hm[e] >= WIN0) & (self.hm[e] <= WIN1)
        return e, ok

    def floor_stop(self, side, ref, stop, e):
        f = FLOOR_ATR * self.atr[e]
        return np.where(side == 1, np.minimum(stop, ref - f), np.maximum(stop, ref + f))

    def near_level(self, e, lo, hi, ref, dnear):
        lv = self.Lv[e]                                           # (n, levels)
        a = (dnear * self.atr[e])[:, None]
        inside = (lv >= lo[:, None]) & (lv <= hi[:, None])
        close = np.abs(lv - ref[:, None]) <= a
        hit = np.nan_to_num(inside | close, nan=False).astype(bool) & np.isfinite(lv)
        any_ = hit.any(1)
        which = np.where(any_, np.array(LEVELS, dtype=object)[hit.argmax(1)], None)
        return any_, which


# --------------------------------------------------------------------------------------------- E1 iFVG at a level
def e1(ctx, tf, conf, entry, dnear):
    iv = ctx.ifvgs[tf]
    e, ok = ctx.entry_index(iv.t_known.to_numpy())
    iv = iv[ok].reset_index(drop=True)
    e = e[ok]
    side = iv.dir.to_numpy()
    lo, hi = iv.lo.to_numpy(), iv.hi.to_numpy()
    if entry == "market":
        ref = ctx.o[e]
        limit = np.full(len(iv), np.nan)
    else:
        ref = np.where(side == 1, hi, lo)
        limit = ref
    near, which = ctx.near_level(e, lo, hi, ref, dnear)
    stop_lo, stop_hi = lo.copy(), hi.copy()
    keep = near.copy()
    if conf != "none":
        F = ctx.fvgs[int(conf)]
        fc = F.t_created.to_numpy()
        order = np.argsort(fc)
        F = F.iloc[order]
        fc, fe = F.t_created.to_numpy(), F.t_end.to_numpy()
        fd, flo, fhi = F.dir.to_numpy(), F.lo.to_numpy(), F.hi.to_numpy()
        tk = iv.t_known.to_numpy()
        span = np.timedelta64(int(conf) * fvg.EXPIRY[int(conf)] + int(conf), "m")
        for i in np.flatnonzero(keep):
            a, b = np.searchsorted(fc, tk[i] - span), np.searchsorted(fc, tk[i], side="right")
            m = (fe[a:b] > tk[i]) & (fd[a:b] == side[i]) & (flo[a:b] <= hi[i]) & (fhi[a:b] >= lo[i])
            if not m.any():
                keep[i] = False
                continue
            j = a + np.flatnonzero(m)[-1]                         # most recent aligned FVG
            stop_lo[i], stop_hi[i] = min(lo[i], flo[j]), max(hi[i], fhi[j])
    stop = np.where(side == 1, stop_lo - TICK, stop_hi + TICK)
    stop = ctx.floor_stop(side, ref, stop, e)
    out = pd.DataFrame(dict(e=e, side=side, stop=stop, r_mult=R_MULT, limit=limit,
                            window=10 if entry != "market" else 0, level=which, zone=hi - lo))
    return out[keep].reset_index(drop=True)


# --------------------------------------------------------------------------------------------- E2 sweep & reclaim
def e2(ctx, tf, entry, reclaim=5):
    b = ctx.bars[tf]
    bi = ctx.B.index_at(b.ts.to_numpy())                         # 1m index of each bar's first minute
    bi = np.minimum(bi, ctx.n - 1)
    o, h, l, c = (b[k].to_numpy() for k in ("open", "high", "low", "close"))
    s = b.sess.to_numpy()
    hmb = ctx.hm[bi]
    tclose = b.t_close.to_numpy()
    n = len(b)
    rows = []
    for k, name in enumerate(LEVELS):
        lev = ctx.Lv[bi, k]
        for side in (1, -1):
            if side == 1:
                sw = (o > lev) & (l <= lev - TICK)
            else:
                sw = (o < lev) & (h >= lev + TICK)
            sw &= np.isfinite(lev) & (hmb >= 570) & (hmb < 945)
            for i in np.flatnonzero(sw):
                for j in range(i, min(i + reclaim, n)):
                    if s[j] != s[i]:
                        break
                    if (side == 1 and c[j] > lev[i]) or (side == -1 and c[j] < lev[i]):
                        ext = l[i:j + 1].min() if side == 1 else h[i:j + 1].max()
                        rows.append((s[i], name, side, tclose[j], lev[i], ext))
                        break
    df = pd.DataFrame(rows, columns=["sess", "level", "side", "t_known", "lev", "ext"])
    df = df.sort_values("t_known")
    e, ok = ctx.entry_index(df.t_known.to_numpy())
    df = df.assign(e=e)[ok]
    df = df.drop_duplicates(["sess", "level", "side"], keep="first").reset_index(drop=True)
    side = df.side.to_numpy()
    e = df.e.to_numpy()
    if entry == "market":
        ref, limit = ctx.o[e], np.full(len(df), np.nan)
    else:
        ref = limit = df.lev.to_numpy()
    stop = np.where(side == 1, df.ext - TICK, df.ext + TICK)
    stop = ctx.floor_stop(side, ref, stop, e)
    return pd.DataFrame(dict(e=e, side=side, stop=stop, r_mult=R_MULT, limit=limit,
                             window=10 if entry != "market" else 0, level=df.level))


# --------------------------------------------------------------------------------------------- E3 failed IB break
def e3(ctx, N, gate, target, narrow_cut):
    rows = []
    L = ctx.L
    ibh, ibl = L.ib_h.to_numpy(), L.ib_l.to_numpy()
    starts = np.r_[0, np.flatnonzero(ctx.sess[1:] != ctx.sess[:-1]) + 1]
    ends = np.r_[starts[1:], ctx.n]
    for a, b in zip(starts, ends):
        idx = a + np.flatnonzero((ctx.hm[a:b] >= 630) & (ctx.hm[a:b] <= WIN1))
        if len(idx) < 30 or not np.isfinite(ibh[idx[0]]) or not np.isfinite(ctx.atr[idx[0]]):
            continue
        H, Lo = ibh[idx[0]], ibl[idx[0]]
        if gate == "narrow" and (H - Lo) / ctx.atr[idx[0]] >= narrow_cut:
            continue
        for side in (-1, 1):                                       # -1: failed break of IB high -> short
            out = (ctx.c[idx] > H) if side == -1 else (ctx.c[idx] < Lo)
            back = (ctx.c[idx] < H) if side == -1 else (ctx.c[idx] > Lo)
            k = 0
            while k < len(idx):
                if not out[k]:
                    k += 1
                    continue
                fail = None
                for m in range(k + 1, min(k + 1 + N, len(idx))):
                    if back[m]:
                        fail = m
                        break
                if fail is None:
                    k += 1
                    continue
                j = idx[fail]
                ext = ctx.h[idx[k]:j + 1].max() if side == -1 else ctx.l[idx[k]:j + 1].min()
                rows.append((j + 1, side, ext, (H + Lo) / 2))
                break
    df = pd.DataFrame(rows, columns=["e", "side", "ext", "mid"])
    df = df[(df.e < ctx.n)]
    e = df.e.to_numpy()
    ok = (ctx.hm[e] >= WIN0) & (ctx.hm[e] <= WIN1) & (ctx.sess[e] == ctx.sess[e - 1])
    df, e = df[ok], e[ok]
    side = df.side.to_numpy()
    ref = ctx.o[e]
    stop = ctx.floor_stop(side, ref, np.where(side == 1, df.ext - TICK, df.ext + TICK), e)
    out = pd.DataFrame(dict(e=e, side=side, stop=stop))
    if target == "1.5R":
        out["r_mult"] = R_MULT
    else:
        out["target"] = df.mid.to_numpy()
        out = out[np.sign(out.target - ref) == out.side]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------------- E4 VWAP pullback
def e4(ctx, anchor, trigger, trend_n=30, look=10):
    vw = ctx.L[anchor].to_numpy()
    above = ctx.c > vw
    below = ctx.c < vw
    s = pd.Series(above.astype(int))
    up_run = s.groupby(ctx.sess).transform(lambda x: x.rolling(trend_n).sum()).shift(1).to_numpy() == trend_n
    s = pd.Series(below.astype(int))
    dn_run = s.groupby(ctx.sess).transform(lambda x: x.rolling(trend_n).sum()).shift(1).to_numpy() == trend_n
    tol = 0.02 * ctx.atr
    rth = (ctx.hm >= 570) & (ctx.hm <= WIN1)
    touch_up = up_run & (ctx.l <= vw + tol) & rth
    touch_dn = dn_run & (ctx.h >= vw - tol) & rth
    if trigger == "ifvg":
        iv = ctx.ifvgs[1]
        ci = ctx.B.index_at(iv.t_known.to_numpy()) - 1                # bar whose close confirmed it
        bull = np.zeros(ctx.n, bool); bear = np.zeros(ctx.n, bool)
        bull[ci[iv.dir.to_numpy() == 1]] = True
        bear[ci[iv.dir.to_numpy() == -1]] = True
    else:
        po, pc = np.r_[np.nan, ctx.o[:-1]], np.r_[np.nan, ctx.c[:-1]]
        bull = (ctx.c > ctx.o) & (pc < po) & (ctx.o <= pc) & (ctx.c >= po)
        bear = (ctx.c < ctx.o) & (pc > po) & (ctx.o >= pc) & (ctx.c <= po)
    rows = []
    for side, touch, trig in ((1, touch_up, bull), (-1, touch_dn, bear)):
        last = -1
        for i in np.flatnonzero(touch):
            if i <= last:
                continue
            for j in range(i, min(i + look, ctx.n - 1)):
                if ctx.sess[j] != ctx.sess[i]:
                    break
                still = (ctx.c[j] > vw[j] - 0.05 * ctx.atr[j]) if side == 1 else (ctx.c[j] < vw[j] + 0.05 * ctx.atr[j])
                if not still:
                    break
                if trig[j]:
                    ext = ctx.l[i:j + 1].min() if side == 1 else ctx.h[i:j + 1].max()
                    rows.append((j + 1, side, ext))
                    last = j + look
                    break
    df = pd.DataFrame(rows, columns=["e", "side", "ext"])
    e = df.e.to_numpy()
    ok = (ctx.hm[e] >= WIN0) & (ctx.hm[e] <= WIN1) & (ctx.sess[e] == ctx.sess[e - 1])
    df, e = df[ok], e[ok]
    side = df.side.to_numpy()
    stop = ctx.floor_stop(side, ctx.o[e], np.where(side == 1, df.ext - TICK, df.ext + TICK), e)
    return pd.DataFrame(dict(e=e, side=side, stop=stop, r_mult=R_MULT))
