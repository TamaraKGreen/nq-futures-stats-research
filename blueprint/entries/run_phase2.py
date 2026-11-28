"""Phase 2 - entry families vs random control on Dev (PLAN §12).   python3 blueprint/entries/run_phase2.py
Writes blueprint/results/phase2_entries.txt, phase2_variants.csv and cache/phase2_trades.parquet."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from engine.data import load_split, SLIP, PT_VALUE          # noqa: E402
from engine.levels import build_levels                       # noqa: E402
from engine import sim                                        # noqa: E402
from entries import signals as S                              # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 500)
CACHE, RES = HERE / "cache", HERE / "results"
HALF = pd.Timestamp("2022-10-01")
N_CTRL = 20
rng = np.random.default_rng(7)

t0 = time.time()
d, L = build_levels(load_split("dev"))
ctx = S.Ctx(d, L)
days = pd.read_parquet(CACHE / "days_dev.parquet")
NARROW = float(np.nanquantile(days.ib_w_atr, 1 / 3))
print(f"context built {time.time() - t0:.0f}s; narrow IB cut {NARROW:.3f} ATR")

# ------------------------------------------------------------------------------------------ variants (§12)
VARIANTS = []
for tf in (1, 5, 15):
    for conf in (("none", "5", "15") if tf == 1 else ("none",)):
        for entry in ("market", "retest"):
            for dn in (0.03, 0.06):
                VARIANTS.append(("E1", f"tf{tf} conf{conf} {entry} d{dn}", lambda tf=tf, conf=conf, entry=entry, dn=dn: S.e1(ctx, tf, conf, entry, dn)))
for tf in (1, 5):
    for entry in ("market", "retest"):
        VARIANTS.append(("E2", f"tf{tf} {entry}", lambda tf=tf, entry=entry: S.e2(ctx, tf, entry)))
for N in (5, 15):
    for gate in ("narrow", "all"):
        for tgt in ("1.5R", "mid"):
            VARIANTS.append(("E3", f"N{N} {gate} tgt{tgt}", lambda N=N, gate=gate, tgt=tgt: S.e3(ctx, N, gate, tgt, NARROW)))
for anchor in ("vwap_rth", "vwap_day"):
    for trig in ("ifvg", "engulf"):
        VARIANTS.append(("E4", f"{anchor} {trig}", lambda anchor=anchor, trig=trig: S.e4(ctx, anchor, trig)))


# ------------------------------------------------------------------------------------------ control
def controls(tr):
    """Mean net of N_CTRL random same-session entries within +/-60 min, same side / stop distance / target rule."""
    out = np.full(len(tr), np.nan)
    B = ctx.B
    for k, r in enumerate(tr.itertuples(index=False)):
        e = r.i_fill
        lo, hi = max(e - 60, 0), min(e + 60, ctx.n - 1)
        cand = np.arange(lo, hi + 1)
        cand = cand[(ctx.sess[cand] == ctx.sess[e]) & (ctx.hm[cand] >= S.WIN0) & (ctx.hm[cand] <= S.WIN1)]
        if len(cand) == 0:
            continue
        picks = rng.choice(cand, N_CTRL)
        nets = []
        for p in picks:
            f = ctx.o[p] + r.side * SLIP
            stop = f - r.side * r.risk
            if np.isfinite(getattr(r, "r_mult", np.nan)):
                res = sim.trade(B, p, r.side, stop, r_mult=r.r_mult)
            else:
                res = sim.trade(B, p, r.side, stop, target=f + r.side * abs(r.target - r.fill))
            if res is not None:
                nets.append(res["net"])
        if nets:
            out[k] = np.mean(nets)
    return out


# ------------------------------------------------------------------------------------------ filters
lvx = ctx.Lv
q20, q80 = None, None
rth_idx = np.flatnonzero((ctx.hm >= S.WIN0) & (ctx.hm <= S.WIN1))
dsum = pd.Series(ctx.delta).rolling(5).sum().shift(1).to_numpy()
vsum = pd.Series(ctx.v).rolling(5).sum().shift(1).to_numpy()
dv = dsum / np.maximum(vsum, 1)
q20, q80 = np.nanquantile(dv[rth_idx], [0.2, 0.8])
b3 = pd.read_parquet(CACHE / "bias_oof_B3_near.parquet").set_index(["sess", "hour"]).p_logit
phh, phl = L.ph_h.to_numpy(), L.ph_l.to_numpy()


def add_filters(tr):
    e = tr.i_fill.to_numpy()
    side = tr.side.to_numpy()
    p = ctx.o[e]
    a = ctx.atr[e]
    lv = lvx[e]
    up = np.where(lv > (p + 0.03 * a)[:, None], lv - p[:, None], np.inf).min(1)
    dn = np.where(lv < (p - 0.03 * a)[:, None], p[:, None] - lv, np.inf).min(1)
    p_long = np.where(np.isfinite(up) & np.isfinite(dn), dn / (up + dn), np.nan)
    p_dir = np.where(side == 1, p_long, 1 - p_long)
    tr["A1"] = p_dir >= 0.6
    # A2 hourly near-extreme agreement
    hour = ctx.hm[e] // 60
    hs = e - (ctx.hm[e] - hour * 60)
    sess = pd.to_datetime(ctx.sess[e])
    pn = b3.reindex(pd.MultiIndex.from_arrays([sess, hour])).to_numpy()
    near_hi = (ctx.o[hs] - phl[hs]) / (phh[hs] - phl[hs]) >= 0.5
    untaken = np.array([(ctx.h[s0:x].max() < phh[s0]) if nh else (ctx.l[s0:x].min() > phl[s0]) if x > s0 else True
                        for s0, x, nh in zip(hs, e, near_hi)])
    tr["A2"] = (pn >= 0.75) & untaken & (side == np.where(near_hi, 1, -1))
    tr["A2_avail"] = np.isfinite(pn)
    # E5 order-flow
    x = dv[e]
    tr["E5_keep"] = ~(((side == 1) & (x <= q20)) | ((side == -1) & (x >= q80)))
    return tr


# ------------------------------------------------------------------------------------------ metrics
def stats(tr):
    if len(tr) < 5:
        return dict(n=len(tr))
    net = tr.net.to_numpy()
    edge = net - tr.ctrl.to_numpy()
    edge = edge[np.isfinite(edge)]
    months = max((tr.sess.max() - tr.sess.min()).days / 30.44, 1)
    gw, gl = net[net > 0].sum(), -net[net < 0].sum()
    yr = tr.groupby(tr.sess.dt.year).net.sum()
    h1, h2 = tr[tr.sess < HALF].net, tr[tr.sess >= HALF].net
    return dict(n=len(tr), per_mo=len(tr) / months, win=100 * (net > 0).mean(), net=net.mean(),
                t_net=net.mean() / (net.std(ddof=1) / np.sqrt(len(net))), pf=gw / gl if gl else np.inf,
                avg_R=tr.R.mean(), ctrl=tr.ctrl.mean(), edge=edge.mean(),
                t_edge=edge.mean() / (edge.std(ddof=1) / np.sqrt(len(edge))),
                yrs_pos=f"{int((yr > 0).sum())}/{len(yr)}", net_h1=h1.mean(), net_h2=h2.mean(),
                usd_mo=net.sum() * PT_VALUE / months, risk=tr.risk.median())


rows, all_tr = [], []
for fam, name, fn in VARIANTS:
    t1 = time.time()
    cands = fn()
    tr = sim.run(ctx.B, cands)
    if len(tr) == 0:
        rows.append(dict(family=fam, variant=name, n=0))
        continue
    tr["sess"] = pd.to_datetime(ctx.sess[tr.i_fill.to_numpy()])
    tr["ctrl"] = controls(tr)
    tr = add_filters(tr)
    tr["family"], tr["variant"] = fam, name
    all_tr.append(tr)
    base = stats(tr)
    promote = (base.get("t_edge", 0) >= 2.0) and (base.get("net_h1", -1) > 0) and (base.get("net_h2", -1) > 0)
    rows.append(dict(family=fam, variant=name, filt="-", promote=promote, **base))
    for f, m in (("A1", tr.A1), ("E5", tr.E5_keep), ("A1+E5", tr.A1 & tr.E5_keep)):
        rows.append(dict(family=fam, variant=name, filt=f, **stats(tr[m.to_numpy()])))
    av = tr.A2_avail.to_numpy()
    rows.append(dict(family=fam, variant=name, filt="2022+ all", **stats(tr[av])))
    rows.append(dict(family=fam, variant=name, filt="2022+ A2", **stats(tr[av & tr.A2.to_numpy()])))
    print(f"{fam} {name:32s} cands {len(cands):6d} trades {len(tr):5d} net {base.get('net', np.nan):6.2f} "
          f"edge {base.get('edge', np.nan):6.2f} t {base.get('t_edge', np.nan):5.2f}  {time.time() - t1:.0f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(RES / "phase2_variants.csv", index=False)
pd.concat(all_tr, ignore_index=True).to_parquet(CACHE / "phase2_trades.parquet", index=False)

cols = ["family", "variant", "filt", "n", "per_mo", "win", "net", "t_net", "pf", "avg_R", "ctrl", "edge", "t_edge",
        "yrs_pos", "net_h1", "net_h2", "usd_mo", "risk"]
txt = ["PHASE 2 - ENTRY FAMILIES ON DEV (2020-01 -> 2025-07), neutral exit: structural stop (>= 0.02 ATR), 1.5R, 30 min",
       "net = avg net points per trade after $6.50 RT (0.65 pt) and 1-tick slippage; 5 MNQ = $10/pt.",
       "ctrl = same trade rules from random entries +/-60 min same session; edge = net - ctrl; t_edge paired.",
       "promote = t_edge >= 2 and net > 0 in both halves (h1 < 2022-10, h2 >= 2022-10).", "",
       "UNFILTERED", res[res.filt == "-"][cols + ["promote"]].round(2).to_string(index=False), "",
       "FILTERS (A1 nearer-level agreement, E5 order-flow skip, A2 hourly near-extreme agreement - 2022+ only)",
       res[res.filt != "-"][cols].round(2).to_string(index=False)]
(RES / "phase2_entries.txt").write_text("\n".join(txt))
print("\n".join(txt[:7]))
print(f"total {time.time() - t0:.0f}s")
