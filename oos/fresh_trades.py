"""
Re-run the pre-registered trade tests (tradability, entries, scalps) with their own
setup code, unchanged, and report three periods:
  discovery (2020 - 14 Mar 2025)  |  holdout (17 Mar 2025 - 29 Apr 2026)  |  fresh (30 Apr - 24 Nov 2026)
Variant selection follows each script's own rule on DISCOVERY only. NOTE: discovery here
starts in 2020 (no 2010-19 bars), so a chosen variant can differ from the original choice.

    python -m oos.fresh_trades
"""
import numpy as np
import pandas as pd

from .data import load_nq, fresh_of
from . import families as F
from . import tradability as TR
from . import entries as EN
from . import scalps as SC

PER = ("discovery", "holdout", "fresh")
OUT = TR.OUT


def tradability():
    b, days, atr, ib, pre8, h = TR.build_levels(load_nq())
    D = TR.Days(b)
    specs = []
    for k in (0.25, 0.5, 0.75, 1.0):
        specs.append(("IB fade to mid - narrow IB (3.1)", f"stop {k}x IB beyond", lambda k=k: TR.setup_ib_fade(D, b, days, atr, ib, True, k)))
        specs.append(("IB fade to mid - all days (control)", f"stop {k}x IB beyond", lambda k=k: TR.setup_ib_fade(D, b, days, atr, ib, False, k)))
    for m in (1, 2, 3, 5):
        specs.append(("8am third -> overnight extreme (3.2)", f"stop {m}x target dist", lambda m=m: TR.setup_on_extreme(D, days, pre8, m)))
        specs.append(("9:00 hour -> 8:00 mid (3.5)", f"stop {m}x target dist", lambda m=m: TR.setup_nine_mid(D, h, days, m)))
    for k in (0.25, 0.5, 1.0):
        specs.append(("9:00 sweep continuation (3.5)", f"stop {k}x prev-hr range", lambda k=k: TR._sweep_trades(D, b, h, days, 9, k, "cont")))
        specs.append(("11-13 sweep of low, fade (3.5)", f"stop {k}x prev-hr range",
                      lambda k=k: pd.concat([TR._sweep_trades(D, b, h, days, H_, k, "fade", "L") for H_ in (11, 12, 13)], ignore_index=True)))
    rows = []
    for name, var, fn in specs:
        t = fn()
        if not len(t):
            continue
        t["period"] = fresh_of(t["day"])
        for per in PER:
            rows.append({"setup": name, "variant": var, "period": per, **TR.summarise(t[t["period"] == per])})
    res = pd.DataFrame(rows)
    disc = res[res.period == "discovery"]
    pick = disc.loc[disc.groupby("setup")["avg_net_pts"].idxmax(), ["setup", "variant"]]
    return res, res.merge(pick, on=["setup", "variant"])


def entries():
    b, days, atr, ib, pre8, h = TR.build_levels(load_nq())
    D = TR.Days(b)
    orb = F.window(b, "09:30", "09:45", 0.8).reindex(days)
    ibq = F.disc_quintiles((ib["h"] - ib["l"]) / atr, days)
    orbq = F.disc_quintiles((orb["h"] - orb["l"]) / atr, days)
    specs = []
    for ctx in ("none", "narrow IB", "wide IB"):
        for x in (0, 0.5):
            for sk in ("mid", "far"):
                specs.append(("A 1h candle continuation", ctx, f"target +{x}x C2, stop C2 {sk}", lambda x=x, sk=sk, ctx=ctx: EN.method_a(D, h, ibq, x, sk, ctx)))
    for ctx in ("all", "narrow", "wide"):
        for sk in ("mid", "opp"):
            for tk in ("1x", "close"):
                specs.append(("B IB breakout", ctx, f"stop {sk}, target {tk}", lambda sk=sk, tk=tk, ctx=ctx: EN.method_breakout(D, b, ib, ibq, "10:30", sk, tk, ctx)))
                specs.append(("C ORB breakout", ctx, f"stop {sk}, target {tk}", lambda sk=sk, tk=tk, ctx=ctx: EN.method_breakout(D, b, orb, orbq, "09:45", sk, tk, ctx)))
        specs.append(("D IB second side", ctx, "limit at mid, stop at extreme", lambda ctx=ctx: EN.method_second_side(D, b, ib, ibq, ctx)))
    rows = []
    for meth, ctx, var, fn in specs:
        t = fn()
        if not len(t):
            continue
        t["period"] = fresh_of(t["day"])
        for per in PER:
            rows.append({"method": meth, "context": ctx, "variant": var, "period": per, **EN._stats(t[t["period"] == per])})
    res = pd.DataFrame(rows)
    disc = res[res.period == "discovery"]
    pick = disc.loc[disc.groupby(["method", "context"])["net_R"].idxmax(), ["method", "context", "variant"]]
    return res, res.merge(pick, on=["method", "context", "variant"])


def scalps():
    bars = load_nq()
    b = F.prep(bars[bars["Datetime"] >= pd.Timestamp("2019-11-01", tz="America/New_York")])
    L, ref = SC.levels_table(b)
    t = SC.run_trades(TR.Days(b), L)
    t["period"] = fresh_of(t["day"])
    keys = ["level", "mode", "T", "S"]
    disc = SC.vs_placebo(t[t["period"] == "discovery"])
    keep = disc[(disc["est_net_R"] > 0) & (disc["t_cons"] > 2) & (disc["t_opt"] > 2)]
    out = [disc.assign(period="discovery")]
    for per in ("holdout", "fresh"):
        out.append(SC.vs_placebo(t[t["period"] == per]).assign(period=per))
    return pd.concat(out, ignore_index=True), keep[keys]


if __name__ == "__main__":
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 30)
    res, chosen = tradability()
    res.to_csv(f"{OUT}/fresh_tradability_all.csv", index=False)
    print("=== TRADABILITY: variant chosen on discovery, all three periods ===")
    print(chosen[["setup", "variant", "period", "trades", "win_%", "avg_net_pts", "avg_R", "profit_factor", "years_positive"]].to_string(index=False))
    res, chosen = entries()
    res.to_csv(f"{OUT}/fresh_entries_all.csv", index=False)
    print("\n=== ENTRIES: variant chosen on discovery, all three periods ===")
    print(chosen[["method", "context", "variant", "period", "trades", "win_%", "net_R", "net_R_se", "net_pts"]].to_string(index=False))
    allp, keep = scalps()
    allp.to_csv(f"{OUT}/fresh_scalps_all.csv", index=False)
    print(f"\n=== SCALPS vs placebo: {len(keep)} cells pass the discovery bar ===")
    if len(keep):
        print(allp.merge(keep, on=["level", "mode", "T", "S"])[["level", "mode", "T", "S", "period", "n", "edge_R_cons", "edge_R_opt", "t_cons", "t_opt", "est_net_R", "est_net_pts"]].to_string(index=False))
    print("\nbest 15 discovery cells by est_net_R, with later periods:")
    top = allp[allp.period == "discovery"].nlargest(15, "est_net_R")[["level", "mode", "T", "S"]]
    print(allp.merge(top, on=["level", "mode", "T", "S"])[["level", "mode", "T", "S", "period", "n", "t_cons", "t_opt", "est_net_R", "est_net_pts"]].to_string(index=False))
