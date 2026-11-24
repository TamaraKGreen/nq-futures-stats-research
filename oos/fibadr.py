"""
Fibonacci and ADR levels through the same level-touch scalp test as scalps.py
(first touch 09:30-10:40, fade / break, T 10/20, S 10/15/20, vs placebo levels).

    py -m oos.fibadr

Levels (all known at 09:30):
  PD fib  0.382 / 0.5 / 0.618 / 0.786 of the prior RTH range, measured from PDL
  ON fib  same fractions of the overnight (18:00-09:29) range, from ONL
  ADR     RTH open +/- 0.5 x ADR14 and +/- 1.0 x ADR14 (ADR = mean RTH range)
"""
import numpy as np
import pandas as pd

from . import scalps as SC
from .orderflow import load_flow
from .tradability import Days, OUT

FR = (0.382, 0.5, 0.618, 0.786)


def run():
    b = load_flow()
    L, ref = SC.levels_table(b)
    rth = __import__("oos.families", fromlist=["window"]).window(b, "09:30", "16:00", 0.9)
    o930 = rth["o"].reindex(L.index)
    new = {}
    for f in FR:
        new[f"PDF{int(f*1000)}"] = L["PDL"] + f * (L["PDH"] - L["PDL"])
        new[f"ONF{int(f*1000)}"] = L["ONL"] + f * (L["ONH"] - L["ONL"])
    for m in (0.5, 1.0):
        new[f"ADRU{int(m*10)}"] = o930 + m * L["atr"]
        new[f"ADRD{int(m*10)}"] = o930 - m * L["atr"]
    for k, v in new.items():
        L[k] = v
    keep = {k: v for k, v in SC.WINDOWS.items() if k.startswith("PA")}
    SC.WINDOWS = {**{k: ("09:30", SC.LAST_ENTRY) for k in new}, **keep}
    SC.PLACEBO_OF = {k: "PA" for k in new}
    t = SC.run_trades(Days(b), L)
    res = SC.vs_placebo(t)
    res.to_csv(f"{OUT}/fibadr_levels.csv", index=False)
    return res


if __name__ == "__main__":
    r = run()
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    cols = ["level", "mode", "T", "S", "n", "per_month", "win_cons_%", "win_opt_%", "placebo_win_cons_%",
            "edge_R_cons", "edge_R_opt", "t_cons", "t_opt", "est_net_R", "est_net_pts"]
    print(r.sort_values("est_net_R", ascending=False)[cols].head(25).to_string(index=False))
    print(f"\n{((r.est_net_R > 0) & (r.t_cons > 2) & (r.t_opt > 2)).sum()} of {len(r)} cells beat placebo (t>2 both) after costs")
