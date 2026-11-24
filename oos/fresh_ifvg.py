"""
Re-run the pre-registered iFVG candidate (ifvg_htf + ifvg_levels) unchanged and report
three periods: 2020-Mar25 (discovery), Mar25-Apr26 (original 'unseen'), May-Nov26 (fresh).

    python -m oos.fresh_ifvg
"""
import numpy as np
import pandas as pd
from . import ifvg_htf as IH
from . import ifvg_levels as IL
from .data import fresh_of
from .tradability import OUT

if __name__ == "__main__":
    IH.run()                                   # writes ifvg_htf_trades.parquet
    t = IL.run()
    t["period"] = pd.Series(fresh_of(t["day"])).map(
        {"discovery": "1 2020-Mar25", "holdout": "2 Mar25-Apr26", "fresh": "3 FRESH May-Nov26"}).to_numpy()
    t.to_parquet(f"{OUT}/ifvg_levels_trades_fresh.parquet")
    s = t.groupby("near_level").apply(IH.summarise, include_groups=False).reset_index().drop(columns="level_1")
    s.to_csv(f"{OUT}/ifvg_levels_summary_fresh.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    focus = s[(s.ctx == "inside, aligned") & (s.entry == "B retest")].sort_values(["stop", "near_level", "period"])
    print("CANDIDATE: inside aligned 15m FVG, B retest\n"); print(focus.to_string(index=False))
    print("\nALL CONTEXTS, B retest, near key level\n")
    print(s[(s.entry == "B retest") & (s.near_level == "near key level")].sort_values(["ctx", "stop", "period"]).to_string(index=False))
