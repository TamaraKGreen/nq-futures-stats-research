"""
Out-of-sample scorecard for the NQ statistics.

    py -m oos.scorecard build      # build event tables (cached in oos/cache)
    py -m oos.scorecard discover   # discovery data only -> frozen shortlist
    py -m oos.scorecard holdout    # open the lockbox for the frozen shortlist
    py -m oos.scorecard report     # write oos/results/scorecard.md

Discovery (<= 2025-03-14) runs rangelab's four defences on every test:
inner 2019-06-01 train/test, per-year sign agreement, day/month-block bootstrap,
and a permutation null + BH-FDR over all cells in the test.

The shortlist is frozen and hashed BEFORE the holdout (2025-03-17 -> 2026-04-29)
is read. The holdout is then scored only for those frozen cells. Every opening
is appended to results/lockbox_ledger.csv. If you change a definition after
seeing holdout results, the holdout is no longer clean for that test - the
ledger is there so you can tell.
"""
import hashlib
import json
import sys
import time
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from . import engine
from . import families as F
from .data import load_nq, period_of, INNER_SPLIT, DISCOVERY_END, HOLDOUT_START

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CACHE = HERE / "cache"

# ---------------------------------------------------------------- the tests --
# (table, outcome, conditioners). Each line is one "statistic family" from your
# notebooks; every cell it produces counts toward that test's multiple-testing bar.
S, H = "session", "hourly"
TESTS = [
    # RTH direction (HODLOD cells 0/5, comprehensive price action)
    (S, "rth_bull", ["open_vs_pd"]), (S, "rth_bull", ["gap_b"]), (S, "rth_bull", ["ldn_vs_asia"]),
    (S, "rth_bull", ["on_dir"]), (S, "rth_bull", ["open_vs_69"]), (S, "rth_bull", ["dow"]),
    (S, "rth_bull", ["open_vs_pd", "gap_b"]), (S, "rth_bull", ["ldn_vs_asia", "on_dir"]),
    (S, "post_ib_bull", ["ib_dir"]), (S, "post_ib_bull", ["ib_dir", "ib_close_pos"]),
    (S, "post_ib_bull", ["orb_first_in_ib", "ib_dir"]), (S, "post_ib_bull", ["ib_vs_pd"]),
    # gap fill (HODLOD cell 12)
    (S, "gap_filled", ["gap_b"]), (S, "gap_filled", ["gap_b", "open_vs_pd"]),
    (S, "gap_filled", ["gap_b", "dow"]), (S, "gap_filled", ["gap_b", "on_dir"]),
    # prior-day levels
    (S, "touch_pdh", ["open_vs_pd"]), (S, "touch_pdh", ["gap_b"]), (S, "touch_pdh", ["ldn_vs_asia"]),
    (S, "touch_pdl", ["open_vs_pd"]), (S, "touch_pdl", ["gap_b"]), (S, "touch_pdl", ["ldn_vs_asia"]),
    # 6-9 range (HODLOD cell 8)
    (S, "rth_hits_69_mid", ["open_vs_69"]), (S, "rth_hits_69_mid", ["open_vs_69", "gap_b"]),
    # pre-8am HOD/LOD (HODLOD cell 2)
    (S, "pre8_high_holds", ["pos_at_8"]), (S, "pre8_high_holds", ["ldn_vs_asia"]),
    (S, "pre8_high_holds", ["pos_at_8", "on_dir_8"]),
    (S, "pre8_low_holds", ["pos_at_8"]), (S, "pre8_low_holds", ["ldn_vs_asia"]),
    (S, "pre8_low_holds", ["pos_at_8", "on_dir_8"]),
    # Initial balance (Initial balance nb, HODLOD cell 7, NQ IB Orb)
    (S, "ib_broke", ["ib_w_q"]), (S, "ib_both_sides", ["ib_w_q"]), (S, "ib_both_sides", ["ib_w_q", "ib_dir"]),
    (S, "ib_up_first", ["ib_dir"]), (S, "ib_up_first", ["ib_close_pos"]), (S, "ib_up_first", ["orb_first_in_ib"]),
    (S, "ib_up_first", ["open_vs_pd"]), (S, "ib_up_first", ["ib_vs_pd"]), (S, "ib_up_first", ["ib_w_q", "ib_close_pos"]),
    (S, "ib_back_to_mid", ["ib_w_q"]), (S, "ib_back_to_mid", ["ib_close_pos"]),
    (S, "ib_ext_half", ["ib_w_q"]), (S, "ib_ext_half", ["ib_dir"]),
    (S, "ib_close_beyond_break", ["ib_w_q"]), (S, "ib_close_beyond_break", ["ib_dir"]),
    # ORB 15m (HODLOD cell 20, NQ IB Orb)
    (S, "orb_up_first", ["orb_dir"]), (S, "orb_up_first", ["orb_close_pos"]), (S, "orb_up_first", ["open_vs_pd"]),
    (S, "orb_up_first", ["gap_b"]), (S, "orb_both_sides", ["orb_w_q"]), (S, "orb_back_to_mid", ["orb_w_q"]),
    (S, "orb_ext_half", ["orb_w_q"]), (S, "orb_close_beyond_break", ["orb_w_q"]),
    # hourly sweeps / mids (HODLOD cells 9-11, hourly range mid & open retracements)
    (H, "sweep_prev_high", ["hour", "open_pos"]), (H, "sweep_prev_low", ["hour", "open_pos"]),
    (H, "sweep_closes_inside", ["hour", "swept_side"]), (H, "hits_prev_mid", ["hour", "open_pos"]),
    (H, "hour_bull", ["hour", "prev_dir"]), (H, "hour_bull", ["hour", "open_pos"]),
]
for tf in ("1h", "4h", "1D"):        # candle continuations & reversals notebook
    T = f"candle_{tf}"
    TESTS += [(T, "c3_high_first", ["c2_rel", "c2_close"]), (T, "c3_high_first", ["c2_rel", "colours"]),
              (T, "c3_takes_c2_high", ["c2_rel", "c2_close"]), (T, "c3_takes_c2_low", ["c2_rel", "c2_close"]),
              (T, "c3_bull", ["c2_rel", "colours"]), (T, "c3_bull", ["c2_rel", "c2_close"])]
    if tf != "1D":
        TESTS += [(T, "c3_high_first", ["c3_hour", "c2_rel"])]
R = "ranges"                          # rangelab
TESTS += [(R, f, by) for f in ("break_up", "rev_mid", "rev_far_edge", "cont_1p0R")
          for by in (["duration", "w_atr_q"], ["duration", "slot", "op_bucket"])]

OUTCOME_TEXT = {
    "rth_bull": "RTH closes above its 09:30 open",
    "post_ib_bull": "RTH close above the 10:30 price",
    "gap_filled": "RTH gap (>=0.15 ATR) fills to prior RTH close",
    "touch_pdh": "RTH trades up to prior-day high (opened below it)",
    "touch_pdl": "RTH trades down to prior-day low (opened above it)",
    "rth_hits_69_mid": "RTH returns to 6-9am range mid (opened outside 6-9)",
    "pre8_high_holds": "High set 18:00-08:00 survives to 16:00",
    "pre8_low_holds": "Low set 18:00-08:00 survives to 16:00",
    "ib_broke": "IB broken (either side) 10:30-16:00",
    "ib_both_sides": "Both IB extremes taken 10:30-16:00",
    "ib_up_first": "First IB break is UP (given a break)",
    "ib_back_to_mid": "After first IB break, price returns to IB mid",
    "ib_ext_half": "First IB break extends >=0.5x IB width",
    "ib_close_beyond_break": "RTH closes beyond the first-broken IB edge",
    "orb_up_first": "First 15m-ORB break is UP (given a break)",
    "orb_both_sides": "Both ORB extremes taken 09:45-16:00",
    "orb_back_to_mid": "After first ORB break, price returns to ORB mid",
    "orb_ext_half": "First ORB break extends >=0.5x ORB width",
    "orb_close_beyond_break": "RTH closes beyond the first-broken ORB edge",
    "sweep_prev_high": "Hour takes previous hour's high",
    "sweep_prev_low": "Hour takes previous hour's low",
    "sweep_closes_inside": "Hour sweeps one side of prev hour, closes back inside",
    "hits_prev_mid": "Hour trades through previous hour's mid",
    "hour_bull": "Hour closes above its open",
    "c3_high_first": "C3 takes C2 high before C2 low",
    "c3_takes_c2_high": "C3 takes C2 high",
    "c3_takes_c2_low": "C3 takes C2 low",
    "c3_bull": "C3 closes bullish",
    "break_up": "Range breaks HIGH first (rangelab)",
    "rev_mid": "Broken range returns to its mid (rangelab)",
    "rev_far_edge": "Broken range returns to far edge (rangelab)",
    "cont_1p0R": "Broken range travels >=1 width past edge (rangelab)",
}

SETTINGS = {  # per table: min cell size, per-year min n, permutations
    "session": dict(min_n=80, min_year_n=8, n_perm=1000, n_boot=300),
    "hourly": dict(min_n=150, min_year_n=10, n_perm=500, n_boot=200),
    "candle_1h": dict(min_n=300, min_year_n=20, n_perm=300, n_boot=150),
    "candle_4h": dict(min_n=150, min_year_n=10, n_perm=500, n_boot=200),
    "candle_1D": dict(min_n=80, min_year_n=6, n_perm=1000, n_boot=300),
    "ranges": dict(min_n=50, min_year_n=20, n_perm=200, n_boot=100),
}


# ------------------------------------------------------------------- tables --

def build_tables(names=None, refresh=False):
    bars = None
    builders = {
        "session": lambda b: F.session_table(b),
        "hourly": lambda b: F.hourly_table(b),
        "candle_1h": lambda b: F.candle_table(b, "1h"),
        "candle_4h": lambda b: F.candle_table(b, "4h"),
        "candle_1D": lambda b: F.candle_table(b, "1D"),
        "ranges": lambda b: F.range_table(b),
    }
    out = {}
    for n in names or builders:
        f = CACHE / f"table_{n}.parquet"
        if f.exists() and not refresh:
            out[n] = pd.read_parquet(f)
            continue
        if bars is None:
            t = time.time(); bars = load_nq(); print(f"bars {len(bars):,} ({time.time()-t:.0f}s)")
        t = time.time()
        tb = builders[n](bars)
        tb["period"] = period_of(tb["sess"])
        tb.to_parquet(f)
        print(f"  {n:10s} {len(tb):>8,} rows  ({time.time()-t:.0f}s)  "
              f"discovery {int((tb['period']=='discovery').sum()):,}  holdout {int((tb['period']=='holdout').sum()):,}")
        out[n] = tb
    return out


def population(t, outcome, by):
    """Exactly the rows score_cells would use for this outcome."""
    d = t
    if "usable" in d.columns:
        d = d[d["usable"].astype(bool)]
    if outcome == "break_up":
        d = d[d["broke"].astype(bool)].copy()
        d["break_up"] = (d["break_side"] == "High").astype("float64")
    elif outcome.startswith(("rev_", "cont_")):
        d = d[d["broke"].astype(bool)]
    d = d[np.isfinite(d[outcome].to_numpy(dtype="float64"))]
    for c in by:
        d = d[d[c].notna()]
    return d


def rw_col(t, outcome):
    """Random-walk benchmark column for 'which side first' outcomes, if any."""
    if f"{outcome}_rw" in t.columns:
        return f"{outcome}_rw"
    if outcome == "break_up" and "close_pos" in t.columns:
        return "close_pos"
    return None


_CELL_CACHE = {}


def _cell_table(t, key, outcome, by):
    """All cells of one test in one pass: n, rate, rw per cell, plus the base."""
    ck = (key, outcome, tuple(by))
    if ck not in _CELL_CACHE:
        p = population(t, outcome, by)
        k = p[by[0]].astype(str)
        for c in by[1:]:
            k = k + " | " + p[c].astype(str)
        y = p[outcome].astype("float64")
        g = pd.DataFrame({"k": k.to_numpy(), "y": y.to_numpy()})
        rc = rw_col(p, outcome)
        g["rw"] = p[rc].to_numpy("float64") if rc else np.nan
        agg = g.groupby("k").agg(n=("y", "size"), rate=("y", "mean"), rw=("rw", "mean"))
        _CELL_CACHE[ck] = (agg, float(y.mean()) if len(y) else np.nan)
    return _CELL_CACHE[ck]


def cell_stats(t, outcome, by, vals, key):
    """n, rate, base and random-walk expectation for one cell. `key` names the
    table slice (e.g. 'ranges/holdout') so each test is filtered only once."""
    agg, base = _cell_table(t, key, outcome, by)
    cell = " | ".join(str(vals[c]) for c in by)
    if cell in agg.index:
        a = agg.loc[cell]
        return {"n": int(a["n"]), "rate": float(a["rate"]), "base": base, "rw": float(a["rw"])}
    return {"n": 0, "rate": np.nan, "base": base, "rw": np.nan}


# ---------------------------------------------------------------- discovery --

def discover():
    tabs = build_tables()
    cells, diags, shorts = [], [], []
    for i, (tn, outcome, by) in enumerate(TESTS):
        t = tabs[tn]
        disc = t[t["period"] == "discovery"]
        st = SETTINGS[tn]
        t0 = time.time()
        c, dg = engine.score_cells(disc, outcome, by=by, min_n=st["min_n"], n_boot=st["n_boot"],
                                   n_perm=st["n_perm"], split_date=INNER_SPLIT, block_col="block",
                                   seed=i, min_year_n=st["min_year_n"])
        tid = f"T{i:03d}"
        bys = " x ".join(by)
        c.insert(0, "test", tid); c.insert(1, "table", tn); c.insert(2, "outcome", outcome); c.insert(3, "by", bys)
        c["cell"] = c[by].astype(str).agg(" | ".join, axis=1)
        s = engine.shortlist(c, dg, top=10_000)
        bvn = dg["observed_best_edge"] / dg["null_best_edge_p95"] if dg["null_best_edge_p95"] else np.nan
        diags.append({"test": tid, "table": tn, "outcome": outcome, "by": bys,
                      "rows": dg["rows"], "cells_tested": dg["cells_tested"],
                      "base_%": 100 * dg["base_rate"], "best_edge_pp": 100 * dg["observed_best_edge"],
                      "null_p95_pp": 100 * dg["null_best_edge_p95"], "best_vs_null": bvn,
                      "fdr_pass": int(c["fdr_pass"].sum()), "shortlisted": len(s)})
        print(f"{tid} {tn:10s} {outcome:22s} by {bys:28s} cells {dg['cells_tested']:>4} "
              f"best/null {bvn:5.2f}  shortlisted {len(s):>3}  ({time.time()-t0:.0f}s)", flush=True)
        keep = ["test", "table", "outcome", "by", "cell", "n", "rate", "base", "edge", "train_n", "train_rate",
                "test_n", "test_rate", "train_shrunk", "holds_oos", "years_seen", "year_agree_frac",
                "boot_lo", "boot_hi", "p_perm", "q_value", "fdr_pass"]
        cells.append(c[[k for k in keep if k in c.columns] + by])
        if len(s):
            shorts.append(s[[k for k in keep if k in s.columns] + by])

    allc = pd.concat(cells, ignore_index=True)
    # one more bar: BH across EVERY cell of EVERY test, not just within a test
    rej, q = engine.bh_fdr(allc["p_perm"].where(allc["n"] >= 0).to_numpy(), alpha=0.10)
    allc["global_q"] = q
    allc.to_parquet(OUT / "discovery_cells.parquet")
    pd.DataFrame(diags).to_csv(OUT / "discovery_tests.csv", index=False)

    sl = pd.concat(shorts, ignore_index=True) if shorts else pd.DataFrame()
    if len(sl):
        sl = sl.merge(allc[["test", "cell", "global_q"]], on=["test", "cell"], how="left")
    by_cols = sorted({c for _, _, by in TESTS for c in by})
    sl = sl[[c for c in sl.columns if c not in by_cols] + [c for c in by_cols if c in sl.columns]]
    sl.to_csv(OUT / "frozen_shortlist.csv", index=False)
    h = hashlib.sha256((OUT / "frozen_shortlist.csv").read_bytes()).hexdigest()
    json.dump({"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "sha256": h,
               "discovery_end": str(DISCOVERY_END.date()), "holdout_start": str(HOLDOUT_START.date()),
               "tests": len(TESTS), "cells_tested": int(sum(d["cells_tested"] for d in diags)),
               "shortlisted": len(sl)}, open(OUT / "lockbox.json", "w"), indent=2)
    print(f"\n{len(TESTS)} tests, {sum(d['cells_tested'] for d in diags):,} cells, "
          f"{len(sl)} shortlisted. Frozen sha256 {h[:12]}")


# ------------------------------------------------------------------ holdout --

def _p_one_sided(edge, base, n, sign):
    if n <= 0 or not (0 < base < 1):
        return np.nan
    z = sign * edge / sqrt(base * (1 - base) / n)
    return 0.5 * (1 - erf(z / sqrt(2)))


def annotate(sl: pd.DataFrame) -> pd.DataFrame:
    """Discovery-only: add the random-walk benchmark and flag cells whose edge
    over the base rate is just distance-to-level. Touches no holdout rows."""
    tabs = build_tables(sorted(sl["table"].unique()))
    disc = {n: t[t["period"] == "discovery"] for n, t in tabs.items()}
    rw, bench, edge, kind = [], [], [], []
    for _, r in sl.iterrows():
        by = r["by"].split(" x ")
        ds = cell_stats(disc[r["table"]], r["outcome"], by, r, f"{r['table']}/discovery")
        rate = float(r["rate"])
        if np.isfinite(ds["rw"]):
            e = rate - ds["rw"]
            se = sqrt(max(ds["rw"] * (1 - ds["rw"]), 1e-9) / max(ds["n"], 1))
            k = "beats random walk" if abs(e) > 2 * se and abs(e) >= 0.02 else "MECHANICAL (= random walk)"
            rw.append(ds["rw"]); bench.append(ds["rw"]); edge.append(e); kind.append(k)
        else:
            rw.append(np.nan); bench.append(float(r["base"])); edge.append(rate - float(r["base"]))
            kind.append("vs base rate")
    sl = sl.copy()
    sl["disc_rw"], sl["disc_bench"], sl["disc_edge"], sl["edge_type"] = rw, bench, edge, kind
    return sl


def preview():
    """Discovery-only summary of the frozen shortlist, for review BEFORE the
    lockbox is opened."""
    sl = annotate(pd.read_csv(OUT / "frozen_shortlist.csv", dtype=str))
    sl.to_csv(OUT / "shortlist_preview.csv", index=False)
    print(sl.groupby(["table", "outcome", "edge_type"]).size().to_string())


def holdout():
    lock = json.load(open(OUT / "lockbox.json"))
    f = OUT / "frozen_shortlist.csv"
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    if h != lock["sha256"]:
        sys.exit("frozen_shortlist.csv changed since it was frozen - re-run discover first.")
    sl = annotate(pd.read_csv(f, dtype=str))
    tabs = build_tables(sorted(sl["table"].unique()))
    hold = {n: t[t["period"] == "holdout"] for n, t in tabs.items()}
    rows = []
    for _, r in sl.iterrows():
        tn, outcome = r["table"], r["outcome"]
        by = r["by"].split(" x ")
        hs = cell_stats(hold[tn], outcome, by, r, f"{tn}/holdout")
        n_h, rate_h = hs["n"], hs["rate"]
        # benchmark: random walk for 'which side first' stats, else the base rate
        bench_h = hs["rw"] if np.isfinite(hs["rw"]) else hs["base"]
        e_d = float(r["disc_edge"])
        e_h = rate_h - bench_h if n_h else np.nan
        sgn = np.sign(e_d)
        p = _p_one_sided(e_h, bench_h, n_h, sgn) if n_h else np.nan
        ret = e_h / e_d if e_d else np.nan
        if n_h < 20:
            v = "TOO FEW"
        elif np.sign(e_h) != sgn:
            v = "FLIPPED"
        elif p < 0.10 and ret >= 0.5:
            v = "HOLDS"
        elif ret >= 0.5:
            v = "CONSISTENT (not sig)"
        else:
            v = "FADED"
        rows.append({**r.to_dict(), "hold_n": n_h, "hold_rate": rate_h, "hold_base": hs["base"],
                     "hold_rw": hs["rw"], "hold_edge": e_h, "retained": ret, "hold_p": p, "verdict": v})
    res = pd.DataFrame(rows)
    if len(res):
        res["hold_q"] = engine.bh_fdr(res["hold_p"].to_numpy(), alpha=0.10)[1]
    res.to_csv(OUT / "holdout_results.csv", index=False)
    led = OUT / "lockbox_ledger.csv"
    pd.DataFrame([{"opened_at": time.strftime("%Y-%m-%d %H:%M:%S"), "shortlist_sha256": h,
                   "cells": len(res)}]).to_csv(led, mode="a", header=not led.exists(), index=False)
    if len(res):
        print(res["verdict"].value_counts().to_string())


# ------------------------------------------------------------------- report --

def _md(df: pd.DataFrame, index=True) -> str:
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in d.to_numpy()]
    return "\n".join([head, sep, *body])


def _pct(x):
    return "" if pd.isna(x) else f"{100*float(x):.1f}%"


def report():
    lock = json.load(open(OUT / "lockbox.json"))
    tests = pd.read_csv(OUT / "discovery_tests.csv")
    res = pd.read_csv(OUT / "holdout_results.csv") if (OUT / "holdout_results.csv").exists() else None
    L = [f"# NQ statistics: out-of-sample scorecard\n",
         f"- Discovery: start of data to {lock['discovery_end']} (everything your notebooks used)",
         f"- Holdout: {lock['holdout_start']} to 2026-04-29, unseen; shortlist frozen {lock['frozen_at']} "
         f"(sha256 `{lock['sha256'][:12]}`)",
         f"- {lock['tests']} tests, {lock['cells_tested']:,} cells scored, {lock['shortlisted']} survived discovery\n",
         "## Verdicts\n"]
    if res is not None and len(res):
        order = ["HOLDS", "CONSISTENT (not sig)", "FADED", "FLIPPED", "TOO FEW"]
        L.append(_md(res["verdict"].value_counts().reindex(order).fillna(0).astype(int).to_frame("cells").rename_axis("verdict")))
        for v in order:
            sub = res[res["verdict"] == v].copy()
            if not len(sub):
                continue
            sub["abs"] = sub["disc_edge"].abs()
            sub = sub.sort_values(["outcome", "abs"], ascending=[True, False])
            L.append(f"\n### {v} ({len(sub)})\n")
            L.append("| statistic | condition | benchmark | disc n | disc rate | disc bench | holdout n | holdout rate | holdout bench | retained |")
            L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
            for _, r in sub.iterrows():
                hb = r["hold_rw"] if pd.notna(r.get("hold_rw")) else r["hold_base"]
                L.append(f"| {OUTCOME_TEXT.get(r['outcome'], r['outcome'])} ({r['table']}) | {r['cell']} | "
                         f"{r['edge_type']} | {int(float(r['n']))} | {_pct(r['rate'])} | {_pct(r['disc_bench'])} | "
                         f"{int(r['hold_n'])} | {_pct(r['hold_rate'])} | {_pct(hb)} | "
                         f"{'' if pd.isna(r['retained']) else f'{r['retained']:.2f}'} |")
    L.append("\n## Per test: was the best cell better than luck?\n")
    t = tests.copy()
    for c in ("base_%", "best_edge_pp", "null_p95_pp", "best_vs_null"):
        t[c] = t[c].round(2)
    t["outcome"] = t["outcome"].map(lambda o: OUTCOME_TEXT.get(o, o))
    L.append(_md(t.drop(columns=["rows"]), index=False))
    (OUT / "scorecard.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT / 'scorecard.md'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("build", "all"):
        build_tables(refresh="--refresh" in sys.argv)
    if cmd in ("discover", "all"):
        discover()
    if cmd == "preview":
        preview()
    if cmd in ("holdout", "all"):
        holdout()
    if cmd in ("report", "all"):
        report()
