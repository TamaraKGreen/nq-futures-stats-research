"""
Day-trading exploration of the Kuant Gapline ORB. PRE-DECLARED families (one change at a time from AS CODED):
  A flat time     A1 flat 16:00 (RTH close)
  B target        B1 1R | B2 2R | B3 no target (hold to flat)
  C last entry    C1 13:30 | C2 12:00
  D gap filter    D1 none (every day) | D2 gap >= 0.15 ATR
  E direction     E1 long + short mirror on gap-down days
  F trades/day    F1 1 | F2 2
Protocol (as for the ATR breakout): select on 2020-2024; keep the best of each family only if it beats
AS CODED by >= 0.10 Sharpe there; combine the kept changes; 2025-2026 is reported, never used to choose.
The combination is then compared with random entries (same exits) as a control.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from research.strategies.gap_orb_day import prep, run, stats, by_year

V = {"AS CODED (day only)": {},
     "A1 flat 16:00": dict(FLAT=960),
     "B1 target 1R": dict(RR=1.0), "B2 target 2R": dict(RR=2.0), "B3 no target": dict(RR=None),
     "C1 last entry 13:30": dict(LAST_SIG=810), "C2 last entry 12:00": dict(LAST_SIG=720),
     "D1 no gap filter": dict(GAP=None), "D2 gap >= 0.15 ATR": dict(GAP=("atr", 0.15)),
     "E1 + short mirror": dict(SIDES="both"),
     "F1 1 trade/day": dict(MAX_TRADES=1), "F2 2 trades/day": dict(MAX_TRADES=2)}

if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    d, D = prep()
    days = D.index[D.n >= 380]
    sel, chk = days[days.year <= 2024], days[days.year >= 2025]
    rows, T = {}, {}
    for name, kw in V.items():
        t = run(d, D, **kw); T[name] = t
        a = stats(t[t.exit_day.dt.year <= 2024], sel); b = stats(t[t.exit_day.dt.year >= 2025], chk)
        y = by_year(t, days)["sharpe"]
        rows[name] = {"trades/yr": len(t) / 6.9, "avg R": t.R.mean(), "SEL sharpe 20-24": a.sharpe, "SEL max DD $": a["max DD $"],
                      "CHECK sharpe 25-26": b.sharpe, "CHECK net $": b["net $"], **{str(k): v for k, v in y.items()}}
    R = pd.DataFrame(rows).T.astype(float)
    print("SINGLE CHANGES (select on 2020-24; 2025-26 only reported)\n")
    print(R.round(2).to_string())
    base = R.loc["AS CODED (day only)", "SEL sharpe 20-24"]
    keep = {}
    for fam in "ABCDEF":
        f = R[[n_.startswith(fam) for n_ in R.index]]
        best = f["SEL sharpe 20-24"].idxmax()
        if f.loc[best, "SEL sharpe 20-24"] >= base + 0.10:
            keep[best] = V[best]
    print(f"\nkept (beat AS CODED 2020-24 Sharpe {base:.2f} by >= 0.10): {list(keep) or 'none'}")
    combo = {k: v for kw in keep.values() for k, v in kw.items()}
    final = run(d, D, **combo)
    label = "COMBINED" if combo else "AS CODED (day only)"
    print(f"\n{label} by year:\n")
    print(by_year(final, days).astype(float).round(2).to_string())
    print(f"\n{label}: by direction / exit / entry hour")
    print(final.groupby("dir").R.agg(["size", "mean"]).round(3).to_string())
    print(final.how.value_counts().to_string())
    final["hr"] = final.entry_min // 60
    print(final.groupby("hr").R.agg(["size", "mean"]).round(3).to_string())
    print("\nRANDOM-ENTRY CONTROL (same exits/filters as the final version):")
    for sd in range(5):
        t = run(d, D, RANDOM_P=0.06, SEED=sd, **combo)
        y = by_year(t, days)["sharpe"]
        print(f"  seed {sd}: trades {len(t)}  avg R {t.R.mean():+.3f}  Sharpe ALL {y['ALL']:.2f}  "
              f"2020-24 {stats(t[t.exit_day.dt.year <= 2024], sel).sharpe:.2f}  2025-26 {stats(t[t.exit_day.dt.year >= 2025], chk).sharpe:.2f}")
