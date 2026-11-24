"""
1m iFVG (inside / not inside a 15m FVG) NEAR A KEY LEVEL.

    py -m oos.ifvg_levels

PRE-REGISTERED 20 Nov 2026, before running. Levels = the key liquidity levels
of FVG_IFVG_KeyLevels (session day 18:00 -> 17:00 ET):
  PDH / PDL   previous session-day high / low
  PWH / PWL   previous session-week high / low
  ASH / ASL   Asia 18:00-02:00 high / low
  LNH / LNL   London 02:00-08:00 high / low
  DO / WO     daily open (18:00) / weekly open (Sunday 18:00)
"near" = the level is within 10 today-equivalent pts of the 1m iFVG zone
(0 if inside it) AND on the supporting side: at or below the zone top for a
long, at or above the zone bottom for a short.
Trades are those of oos/ifvg_htf.py (same entries, stops, target, baseline).
Reported for each 15m context x near / not near, both periods.
"""
import numpy as np
import pandas as pd

from . import families as F
from .orderflow import load_flow
from .tradability import OUT
from .ifvg_htf import summarise

NEAR = 10.0


def key_levels():
    b = F.prep(load_flow())
    full = F.window(b, "18:00", "17:00", 0.3)
    asia = F.window(b, "18:00", "02:00", 0.3)
    ldn = F.window(b, "02:00", "08:00", 0.3)
    full = full.dropna(subset=["o"])
    lv = pd.DataFrame(index=full.index)
    lv["PDH"], lv["PDL"] = full["h"].shift(1), full["l"].shift(1)
    wk = full.index - pd.to_timedelta(full.index.dayofweek, unit="D")
    W = full.groupby(wk).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"))
    W[["ph", "pl"]] = W[["h", "l"]].shift(1)
    lv["PWH"], lv["PWL"], lv["WO"] = W["ph"].reindex(wk).to_numpy(), W["pl"].reindex(wk).to_numpy(), W["o"].reindex(wk).to_numpy()
    lv["ASH"], lv["ASL"] = asia["h"], asia["l"]
    lv["LNH"], lv["LNL"] = ldn["h"], ldn["l"]
    lv["DO"] = full["o"]
    return lv


def run():
    t = pd.read_parquet(f"{OUT}/ifvg_htf_trades.parquet")
    lv = key_levels()
    names = list(lv.columns)
    L = lv.reindex(t["day"]).to_numpy()
    top, bot, d, sc = (t[c].to_numpy()[:, None] for c in ("ztop", "zbot", "dirn", "scale"))
    dist = np.where(L > top, L - top, np.where(L < bot, bot - L, 0.0)) / sc
    side_ok = np.where(d > 0, L <= top, L >= bot)
    near = (dist <= NEAR) & side_ok & np.isfinite(L)
    t["near_level"] = np.where(near.any(axis=1), "near key level", "not near")
    t["which"] = [",".join(np.array(names)[r]) for r in near]
    return t


if __name__ == "__main__":
    t = run()
    t.to_parquet(f"{OUT}/ifvg_levels_trades.parquet")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    s = t.groupby("near_level").apply(summarise, include_groups=False).reset_index().drop(columns="level_1")
    s = s[s["entry"] == "B retest"].sort_values(["ctx", "stop", "near_level", "period"])
    s.to_csv(f"{OUT}/ifvg_levels_summary.csv", index=False)
    print(s.to_string(index=False))
    a = t.groupby("near_level").apply(summarise, include_groups=False).reset_index().drop(columns="level_1")
    print("\nentry A (inversion close):")
    print(a[a["entry"] == "A inversion close"].sort_values(["ctx", "stop", "near_level", "period"]).to_string(index=False))
    print("\nshare near a level:", t.drop_duplicates(["t_inv", "dirn"])["near_level"].value_counts(normalize=True).round(2).to_dict())
