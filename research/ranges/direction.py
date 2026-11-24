"""
Exploratory: which side of a range breaks first, within a limited wait (DEV split only).

Wait = WAIT_MULT x range duration (default 1: the next window).
Random-walk baseline: for a driftless walk starting at the range close, P(high breaks first)
equals close_pos = (close - low) / width. Every feature is reported as
    observed P(up first)   vs   mean close_pos in that bucket   ->   excess (pp)
so that "the close was already near the high" is not mistaken for a directional signal.
Excess is also shown per year (how many dev years share the sign) as a stability check.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from research.scripts.load import load
from research.ranges.engine import run

WAIT_MULT = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
OUT = Path(__file__).resolve().parents[1] / "output"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)

bars = load("dev")
ev = run(bars, durs=(15, 30, 60), offsets=(0,), horizon_mult=WAIT_MULT)
ev = ev[ev.full_fwd & ~ev.partial_sess.astype(bool) & (ev.width > 0)].copy()
ev["year"] = ev.sess.dt.year

# ---------------------------------------------------------------- features (all known at range close)
g = ev.groupby("dur", group_keys=False)
ev["bull_bar"] = np.where(ev.close > ev.open, "close>open", np.where(ev.close < ev.open, "close<open", "flat"))
ev["delta_q"] = g.delta_frac.transform(lambda s: pd.qcut(s, 5, labels=["D1 sellers", "D2", "D3", "D4", "D5 buyers"]))
ev["width_q"] = g.w_slot.transform(lambda s: pd.qcut(s, 5, labels=["W1 tight", "W2", "W3", "W4", "W5 wide"]))
ev["close_dec"] = pd.cut(ev.close_pos, np.linspace(0, 1, 11), include_lowest=True)
h = ev.slot_min // 60
ev["block"] = np.select([(h >= 18) | (h < 2), h < 8, (h < 9) | ((h == 9) & (ev.slot_min % 60 < 30)), h < 12],
                        ["Asia 18-02", "London 02-08", "Pre-RTH 08-0930", "NY AM 0930-12"], "NY PM 12-17")
# where this range closed relative to the PREVIOUS adjacent range
prev_adj = (ev.sess == ev.sess.shift(1)) & (ev.dur == ev.dur.shift(1)) & (ev.slot_idx == ev.slot_idx.shift(1) + 1)
ev["vs_prev"] = np.where(~prev_adj, "n/a",
                np.where(ev.close > ev.high.shift(1), "closed above prev high",
                np.where(ev.close < ev.low.shift(1), "closed below prev low", "closed inside prev")))
# short trend: close vs close 4 ranges earlier (same session, same dur), in ATR units
c4 = ev.groupby(["dur", "sess"]).close.shift(4)
tr = (ev.close - c4) / ev.atr_prev
ev["trend4"] = pd.cut(tr, [-np.inf, -2, -0.5, 0.5, 2, np.inf], labels=["strong down", "down", "flat", "up", "strong up"])

brk = ev[ev.broke].copy()
brk["up"] = (brk.side == "up").astype(float)


def edge_table(d, by):
    def f(x):
        yrs = x.groupby("year").apply(lambda y: (y.up.mean() - y.close_pos.mean()), include_groups=False)
        ex = x.up.mean() - x.close_pos.mean()
        return pd.Series({"n": len(x), "up_first%": 100 * x.up.mean(), "rw_expect%": 100 * x.close_pos.mean(),
                          "excess_pp": 100 * ex, "years_same_sign": f"{int((np.sign(yrs) == np.sign(ex)).sum())}/{len(yrs)}",
                          "med_run_past_edge_x_w": (x.ext_max / x.width).median()})
    return d.groupby(by, observed=True).apply(f, include_groups=False).round(1)


print(f"WAIT = {WAIT_MULT:g} x duration   (dev split, full sessions only)\n")
print("=" * 110, "\n0. HOW OFTEN DOES A BREAK HAPPEN WITHIN THE WAIT\n")
print(ev.groupby("dur").apply(lambda d: pd.Series({
    "ranges": len(d), "no break%": 100 * (d.side == "none").mean(), "both same bar%": 100 * d.ambiguous.mean(),
    "up first%": 100 * (d.side == "up").sum() / d.broke.sum(), "med min to break": d.brk_min.median()}),
    include_groups=False).round(1).to_string())

for title, by in [("1. CLOSE POSITION (decile) - the mechanical driver", "close_dec"),
                  ("2. CANDLE DIRECTION", "bull_bar"),
                  ("3. WHICH EXTREME FORMED FIRST", "formed_first"),
                  ("4. ORDER FLOW: delta / volume quintile", "delta_q"),
                  ("5. WIDTH vs SAME-SLOT NORM", "width_q"),
                  ("6. CLOSE vs PREVIOUS RANGE", "vs_prev"),
                  ("7. SHORT TREND (close vs 4 ranges ago, ATR units)", "trend4"),
                  ("8. SESSION BLOCK", "block")]:
    print("\n" + "=" * 110, f"\n{title}\n")
    print(edge_table(brk, ["dur", by]).to_string())

for c in ("close_dec", "delta_q", "width_q", "trend4"):
    brk[c] = brk[c].astype(str)
brk.to_parquet(OUT / f"direction_dev_wait{WAIT_MULT:g}.parquet", index=False)
