"""Phase 2b setups (PLAN §13). One row per setup instance:
    setup, sess, side, start (1m index whose close activates it), end (exclusive: triggers allowed on bars
    start..end-1), target (setup target price)."""
import numpy as np
import pandas as pd


def _first(mask):
    k = np.flatnonzero(mask)
    return k[0] if len(k) else None


def build(ctx, narrow_orb, narrow_ib):
    o, h, l, c, hm = ctx.o, ctx.h, ctx.l, ctx.c, ctx.hm
    L = ctx.L
    pdh, pdl, onh, onl = (L[k].to_numpy() for k in ("pdh", "pdl", "on_h", "on_l"))
    phh, phl = L.ph_h.to_numpy(), L.ph_l.to_numpy()
    starts = np.r_[0, np.flatnonzero(ctx.sess[1:] != ctx.sess[:-1]) + 1]
    ends = np.r_[starts[1:], ctx.n]
    rows = []
    for a, b in zip(starts, ends):
        if ctx.d.partial.iat[a]:
            continue
        hs = hm[a:b]
        pos = {m: a + k for k, m in enumerate(hs)}
        need = (480, 570, 585, 630, 660, 945)
        if any(m not in pos for m in need):
            continue
        i800, i930, i945, i1030, i1100, i1545 = (pos[m] for m in need)
        atr = ctx.atr[i930]
        if not np.isfinite(atr):
            continue
        s = ctx.sess[a]

        def add(name, side, st, en, tgt):
            if st is not None and en is not None and en > st:
                rows.append((name, s, side, st, en, tgt))

        def end_after(st, cond, hard):
            k = _first(cond[st + 1 - a:hard - a]) if hard > st + 1 else None
            return min(st + 1 + k if k is not None else hard, hard)

        # ---- ORB (S1, S2) ---------------------------------------------------------------------------------------
        oh, ol = h[i930:i945].max(), l[i930:i945].min()
        ow = oh - ol
        seg = slice(i945, i1100)
        cs = c[a:b]
        for side in (1, -1):
            brk = (c[seg] > oh) if side == 1 else (c[seg] < ol)
            k = _first(brk)
            if k is None:
                continue
            bi = i945 + k
            back_in = (cs < oh) if side == 1 else (cs > ol)
            beyond = (cs > oh) if side == 1 else (cs < ol)
            add("S1 ORB cont", side, bi, end_after(bi, back_in, i1100), oh + ow if side == 1 else ol - ow)
            if ow / atr < narrow_orb:
                m = _first(back_in[bi + 1 - a:min(bi + 16, i1100) - a])
                if m is not None:
                    fi = bi + 1 + m
                    add("S2 ORB fail", -side, fi, end_after(fi, beyond, i1100), ol if side == 1 else oh)
        # ---- IB (S3, S4) ---------------------------------------------------------------------------------------
        ih, il = h[i930:i1030].max(), l[i930:i1030].min()
        iw = ih - il
        if iw / atr < narrow_ib:
            seg = slice(i1030, i1545)
            for side in (1, -1):
                brk = (c[seg] > ih) if side == 1 else (c[seg] < il)
                k = _first(brk)
                if k is None:
                    continue
                bi = i1030 + k
                back_in = (cs < ih) if side == 1 else (cs > il)
                beyond = (cs > ih) if side == 1 else (cs < il)
                add("S3 IB cont", side, bi, end_after(bi, back_in, i1545), ih + 0.5 * iw if side == 1 else il - 0.5 * iw)
                m = _first(back_in[bi + 1 - a:min(bi + 16, i1545) - a])
                if m is not None:
                    fi = bi + 1 + m
                    add("S4 IB fade", -side, fi, end_after(fi, beyond, i1545), (ih + il) / 2)
        # ---- S5 prior-day magnet -------------------------------------------------------------------------------------
        p, H, Lo = o[i930], pdh[i930], pdl[i930]
        if np.isfinite(H) and Lo < p < H:
            side = 1 if (p - Lo) / (H - Lo) >= 0.5 else -1
            tgt = H if side == 1 else Lo
            touched = (h[a:b] >= tgt) if side == 1 else (l[a:b] <= tgt)
            add("S5 PD magnet", side, i930, end_after(i930 - 1, touched, i1545), tgt)
        # ---- S6 8am third ----------------------------------------------------------------------------------------------
        r8h, r8l = h[a:i800].max(), l[a:i800].min()
        if r8h > r8l:
            q = (o[i800] - r8l) / (r8h - r8l)
            side = 1 if q >= 2 / 3 else (-1 if q <= 1 / 3 else 0)
            if side:
                tgt = onh[i930] if side == 1 else onl[i930]
                if (tgt - p) * side > 0:
                    touched = (h[a:b] >= tgt) if side == 1 else (l[a:b] <= tgt)
                    add("S6 8am third", side, i930, end_after(i930 - 1, touched, i1100), tgt)
        # ---- S7 hour magnet (9:00 hour from 9:30; 10:00 hour) -------------------------------------------------------------
        for H0, st in ((540, i930), (600, pos.get(600))):
            if st is None or H0 not in pos:
                continue
            hstart = pos[H0]
            ph, pl = phh[hstart], phl[hstart]
            if not (np.isfinite(ph) and ph > pl):
                continue
            side = 1 if (o[hstart] - pl) / (ph - pl) >= 0.5 else -1
            tgt = ph if side == 1 else pl
            if st > hstart:   # already taken between the hour start and 9:30?
                done = (h[hstart:st].max() >= tgt) if side == 1 else (l[hstart:st].min() <= tgt)
                if done:
                    continue
            hend = pos.get(H0 + 60, st + (60 - (hm[st] - H0)))
            touched = (h[a:b] >= tgt) if side == 1 else (l[a:b] <= tgt)
            add("S7 hour magnet", side, st, end_after(st - 1, touched, hend), tgt)
    return pd.DataFrame(rows, columns=["setup", "sess", "side", "start", "end", "target"])
