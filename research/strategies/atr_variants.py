"""
Variants 1-5 of the RTH ATR breakout, PRE-DECLARED before running:

  family 1 exits      1a trail 0.5 ATR | 1b trail 1.0 ATR | 1c bank half at +1R, rest to breakeven
  family 2 cut-off    2a last entry 12:00 ET | 2b last entry 13:30 ET
  family 3 trend day  3a entry beyond RTH VWAP | 3b VWAP slope (30 min) with the trade
  family 4 re-entry   4a 1 trade/day | 4b 2 trades/day
  family 5 loss limit 5  stop for the day after -$1,000 realised

Protocol: SELECT on 2020-2024 only (before the decay). From each family keep the variant with the
best 2020-24 Sharpe if it beats the base by >= 0.10; combine the kept ones. 2025-2026 is only
reported, never used to choose. (2025-26 has been looked at for the base strategy, so it is a
check, not a clean out-of-sample; live data from now on is the real test.)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from research.strategies.atr_breakout import prep, run, stats, by_year

VARIANTS = {
    "base": {},
    "1a trail 0.5 ATR": dict(TRAIL_K=0.5), "1b trail 1.0 ATR": dict(TRAIL_K=1.0),
    "1c half at +1R, BE": dict(PARTIAL=True),
    "2a last entry 12:00": dict(LAST_ENTRY=720), "2b last entry 13:30": dict(LAST_ENTRY=810),
    "3a beyond VWAP": dict(VWAP="side"), "3b VWAP slope": dict(VWAP="slope"),
    "4a 1 trade/day": dict(MAX_TRADES=1), "4b 2 trades/day": dict(MAX_TRADES=2),
    "5 day loss $1000": dict(DAY_LOSS=1000),
}

if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    d, D = prep()
    days = D.index[D.index >= D.index[20]]
    sel_days, chk_days = days[days.year <= 2024], days[days.year >= 2025]
    res, trades = {}, {}
    for name, kw in VARIANTS.items():
        t = run(d, D, **kw); trades[name] = t
        a = stats(t[t.sess.dt.year <= 2024], sel_days); b = stats(t[t.sess.dt.year >= 2025], chk_days)
        res[name] = pd.concat([a.add_prefix("SEL 20-24 "), b.add_prefix("CHECK 25-26 ")])
    R = pd.DataFrame(res).T
    cols = ["SEL 20-24 trades", "SEL 20-24 avg R", "SEL 20-24 $/trade", "SEL 20-24 sharpe", "SEL 20-24 max DD $",
            "CHECK 25-26 trades", "CHECK 25-26 avg R", "CHECK 25-26 $/trade", "CHECK 25-26 sharpe", "CHECK 25-26 max DD $"]
    print("SINGLE VARIANTS (selection on 2020-24; 2025-26 shown, not used)\n")
    print(R[cols].astype(float).round(2).to_string())
    base_sh = R.loc["base", "SEL 20-24 sharpe"]
    keep = {}
    for fam in "12345":
        f = R[[n.startswith(fam) for n in R.index]]
        best = f["SEL 20-24 sharpe"].astype(float).idxmax()
        if f.loc[best, "SEL 20-24 sharpe"] >= base_sh + 0.10:
            keep[best] = VARIANTS[best]
    print(f"\nkept (beat base 2020-24 Sharpe {base_sh:.2f} by >= 0.10): {list(keep) or 'none'}")
    combo = {k: v for kw in keep.values() for k, v in kw.items()}
    if combo:
        t = run(d, D, **combo); trades["COMBINED"] = t
        print("\nCOMBINED by year:\n")
        print(by_year(t, D.loc[days]).astype(float).round(2).to_string())
    print("\nBASE by year (for reference):\n")
    print(by_year(trades["base"], D.loc[days]).astype(float).round(2).to_string())
    print("\nSharpe by year, every variant:\n")
    sh = {n: by_year(t, D.loc[days])["sharpe"] for n, t in trades.items()}
    print(pd.DataFrame(sh).T.astype(float).round(2).to_string())
