"""Phase 2b - setup x trigger grid on Dev (PLAN §13).   python3 blueprint/entries/run_phase2b.py
Writes results/phase2b_grid.txt, phase2b_cells.csv, cache/phase2b_trades.parquet."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from engine.data import load_split, SLIP, TICK, PT_VALUE     # noqa: E402
from engine.levels import build_levels                         # noqa: E402
from engine import sim                                          # noqa: E402
from entries import signals as S                                # noqa: E402
from entries import setups as SU                                # noqa: E402
from entries.triggers import build_all                          # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
CACHE, RES = HERE / "cache", HERE / "results"
HALF = pd.Timestamp("2022-10-01")
TEST_YEARS = [2022, 2023, 2024, 2025]

t0 = time.time()
d, L = build_levels(load_split("dev"))
ctx = S.Ctx(d, L)
days = pd.read_parquet(CACHE / "days_dev.parquet")
NARROW_ORB = float(np.nanquantile(days.orb15_w_atr, 1 / 3))
NARROW_IB = float(np.nanquantile(days.ib_w_atr, 1 / 3))
inst = SU.build(ctx, NARROW_ORB, NARROW_IB)
T = build_all(ctx)
print(f"built {time.time() - t0:.0f}s; narrow ORB {NARROW_ORB:.3f} IB {NARROW_IB:.3f} ATR; instances:\n"
      f"{inst.groupby('setup').size().to_string()}", flush=True)

lowmin = pd.Series(ctx.l).rolling(10, min_periods=1).min().to_numpy()
highmax = pd.Series(ctx.h).rolling(10, min_periods=1).max().to_numpy()


def candidates(setup, trig):
    rows = []
    ev = T[trig]
    for r in inst[inst.setup == setup].itertuples(index=False):
        if ev is None:
            j = r.start
        else:
            arr = ev[0] if r.side == 1 else ev[1]
            k = np.flatnonzero(arr[r.start:r.end])
            if not len(k):
                continue
            j = r.start + k[0]
        e = j + 1
        if e >= ctx.n or ctx.sess[e] != ctx.sess[j] or not (S.WIN0 <= ctx.hm[e] <= S.WIN1):
            continue
        ref = ctx.o[e]
        stop = lowmin[j] - TICK if r.side == 1 else highmax[j] + TICK
        f = S.FLOOR_ATR * ctx.atr[e]
        stop = min(stop, ref - f) if r.side == 1 else max(stop, ref + f)
        rows.append((e, r.side, stop, r.target, r.sess))
    return pd.DataFrame(rows, columns=["e", "side", "stop", "tgt_level", "sess0"])


def control(tr):
    out = np.full(len(tr), np.nan)
    for k, r in enumerate(tr.itertuples(index=False)):
        nets = []
        for side in (1, -1):
            f = ctx.o[r.i_fill] + side * SLIP
            stop = f - side * r.risk
            if np.isfinite(r.r_mult):
                x = sim.trade(ctx.B, r.i_fill, side, stop, r_mult=r.r_mult)
            else:
                x = sim.trade(ctx.B, r.i_fill, side, stop, target=f + side * abs(r.target - r.fill))
            if x is not None:
                nets.append(x["net"])
        if len(nets) == 2:
            out[k] = np.mean(nets)
    return out


def stats(tr):
    if len(tr) < 5:
        return dict(n=len(tr))
    net = tr.net.to_numpy()
    edge = (tr.net - tr.ctrl).dropna().to_numpy()
    months = (tr.sess.max() - tr.sess.min()).days / 30.44 or 1
    gw, gl = net[net > 0].sum(), -net[net < 0].sum()
    yr = tr.groupby(tr.sess.dt.year).net.sum()
    return dict(n=len(tr), per_mo=len(tr) / months, win=100 * (net > 0).mean(), net=net.mean(),
                t_net=net.mean() / (net.std(ddof=1) / np.sqrt(len(net))), pf=gw / gl if gl else np.inf,
                edge=edge.mean(), t_edge=edge.mean() / (edge.std(ddof=1) / np.sqrt(len(edge))) if len(edge) > 2 else np.nan,
                yrs_pos=f"{int((yr > 0).sum())}/{len(yr)}", net_h1=tr[tr.sess < HALF].net.mean(),
                net_h2=tr[tr.sess >= HALF].net.mean(), usd_mo=net.sum() * PT_VALUE / months,
                risk=tr.risk.median(), mins=tr.mins.mean())


rows, trades = [], []
for setup in sorted(inst.setup.unique()):
    for trig in T:
        c = candidates(setup, trig)
        if not len(c):
            continue
        for tgt in ("setup", "1.5R"):
            cc = c.copy()
            if tgt == "setup":
                cc["target"], cc["r_mult"] = cc.tgt_level, np.nan
            else:
                cc["target"], cc["r_mult"] = np.nan, 1.5
            tr = sim.run(ctx.B, cc)
            if not len(tr):
                continue
            tr["sess"] = pd.to_datetime(ctx.sess[tr.i_fill.to_numpy()])
            tr["ctrl"] = control(tr)
            tr["setup"], tr["trigger"], tr["tgt"] = setup, trig, tgt
            trades.append(tr)
            st = stats(tr)
            rows.append(dict(setup=setup, trigger=trig, tgt=tgt, **st))
        print(f"{setup:16s} {trig:18s} cands {len(c):5d}  " + "  ".join(
            f"{r['tgt']}: n {r.get('n', 0):4d} net {r.get('net', np.nan):6.2f} t_edge {r.get('t_edge', np.nan):5.2f}"
            for r in rows[-2:]), flush=True)

cells = pd.DataFrame(rows)
cells["pass"] = (cells.t_edge >= 3) & (cells.net_h1 > 0) & (cells.net_h2 > 0)
allt = pd.concat(trades, ignore_index=True)
allt.to_parquet(CACHE / "phase2b_trades.parquet", index=False)

# ---- trigger vs no trigger, within setup and target --------------------------------------------------------------
base = cells[cells.trigger == "T0 none"].set_index(["setup", "tgt"])[["net", "n"]]
cells = cells.join(base.rename(columns={"net": "net_T0", "n": "n_T0"}), on=["setup", "tgt"])
cells["vs_T0"] = cells.net - cells.net_T0
cells.to_csv(RES / "phase2b_cells.csv", index=False)

# ---- walk-forward choice per setup -------------------------------------------------------------------------------
wf_rows = []
for setup, g in allt.groupby("setup"):
    picks, test = [], []
    for yr in TEST_YEARS:
        tr = g[g.sess.dt.year < yr]
        perf = tr.groupby(["trigger", "tgt"]).net.agg(["mean", "size"])
        perf = perf[perf["size"] >= 30]
        if perf.empty:
            continue
        best = perf["mean"].idxmax()
        picks.append(f"{yr}:{best[0].split()[0]}/{best[1]}")
        te = g[(g.sess.dt.year == yr) & (g.trigger == best[0]) & (g.tgt == best[1])]
        test.append(te)
    te = pd.concat(test) if test else pd.DataFrame()
    st = stats(te) if len(te) else {}
    wf_rows.append(dict(setup=setup, picks=" ".join(picks), **st))
wf = pd.DataFrame(wf_rows)

# ---- comparison holds (not used for selection) -------------------------------------------------------------------
cmp_rows = []
for setup in ("S1 ORB cont", "S3 IB cont"):
    for trig in T:
        c = candidates(setup, trig)
        for label, until in (("to 11:00", 11 * 60), ("to 15:59", 16 * 60)):
            res = []
            for r in c.itertuples(index=False):
                mh = until - ctx.hm[r.e]
                if mh <= 0:
                    continue
                x = sim.trade(ctx.B, r.e, r.side, r.stop, target=None, max_hold=mh, capped=False)
                if x is not None:
                    x["sess"] = r.sess0
                    res.append(x)
            if res:
                tr = pd.DataFrame(res)
                tr["sess"] = pd.to_datetime(tr.sess)
                tr["ctrl"] = np.nan
                st = stats(tr)
                cmp_rows.append(dict(setup=setup, trigger=trig, hold=label,
                                     **{k: st.get(k) for k in ("n", "win", "net", "t_net", "pf", "yrs_pos", "net_h1", "net_h2", "usd_mo", "mins")}))
cmp = pd.DataFrame(cmp_rows)

cols = ["setup", "trigger", "tgt", "n", "per_mo", "win", "net", "t_net", "pf", "edge", "t_edge", "yrs_pos", "net_h1",
        "net_h2", "usd_mo", "risk", "mins", "vs_T0", "pass"]
txt = ["PHASE 2b - SETUP x TRIGGER GRID ON DEV (2020-01 -> 2025-07). 30-min / 50-pt caps, $6.50 RT + 1-tick slippage.",
       "edge = net - direction-neutral control (same bar, same bracket); vs_T0 = net minus the same setup/target with no trigger.",
       f"pass = t_edge >= 3 and net > 0 in both halves. Cells: {len(cells)}.", "",
       "ALL CELLS (sorted by t_edge)", cells.sort_values("t_edge", ascending=False)[cols].round(2).to_string(index=False), "",
       "WALK-FORWARD CHOICE PER SETUP (best trigger x target on prior years, scored on the next year; 2022-2025)",
       wf.round(2).to_string(index=False), "",
       "COMPARISON ONLY - continuation setups held to 11:00 / 15:59, stop only, no target (never used for selection)",
       cmp.round(2).to_string(index=False)]
(RES / "phase2b_grid.txt").write_text("\n".join(txt))
print("\n".join(txt[:4]))
print(cells[cells["pass"]][cols].round(2).to_string(index=False))
print(wf.round(2).to_string(index=False))
print(f"total {time.time() - t0:.0f}s")
