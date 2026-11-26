"""
Directional bias in the first 90 minutes of RTH (09:30-11:00 ET).

Every conditioner is scored ONLY on the move AFTER it is known (no overlap with its own move):
  known at 09:30 -> outcome 09:30 open .. 11:00 (close of the 10:59 bar)
  ORB (09:30-09:45) -> 09:45 .. 11:00
  10:00 state (VWAP side, first-30-min direction) -> 10:00 .. 11:00
  first 5m FVG (09:30-10:00) -> signal bar close .. 11:00
Outcomes (all symmetric, so 50% / 0 is "no information"):
  up %      11:00 price above the decision price
  race %    +0.25 ATR reached before -0.25 ATR (from the decision price, by 11:00); no-touch days dropped
  ret ATR   mean (11:00 - decision price) / ATR, with t-stat
ATR = mean of the prior 14 full RTH ranges. Baseline row = all days (NQ's own drift).
'years' = share of years whose mean return has the same sign as the cell's (vs the all-days mean).
dev (to 2025-07-09) and later (2025-07-10 on) shown separately.

Conditioners: from oos.families.session_table (your definitions) + prior value area / POC
(research/stats/value_area.py, RTH and ETH profiles) + prior-day candle and close location.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from oos.families import session_table
from oos.data import load_nq

OUT = ROOT / "research" / "output"
SPLIT = pd.Timestamp("2025-07-10")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)


def outcomes():
    b = load_nq()
    b["t"] = b.Datetime.dt.tz_localize(None)
    b["day"] = b.t.dt.normalize()
    b["mod"] = b.t.dt.hour * 60 + b.t.dt.minute
    r = b[(b["mod"] >= 570) & (b["mod"] < 960)].reset_index(drop=True)
    tp = (r.High + r.Low + r.Close) / 3
    r["vwap"] = (tp * r.Volume).groupby(r.day).cumsum() / r.Volume.groupby(r.day).cumsum()
    return r


def day_rows(r):
    rows = {}
    for day, g in r.groupby("day"):
        if len(g) < 380:
            continue
        m = g["mod"].to_numpy(); H = g.High.to_numpy(); L = g.Low.to_numpy(); C = g.Close.to_numpy()
        pos = {mm: k for k, mm in enumerate(m)}
        if not all(x in pos for x in (570, 584, 599, 659)):
            continue
        rows[day] = dict(o930=g.Open.iloc[0], c945=C[pos[584]], c1000=C[pos[599]], c1100=C[pos[659]],
                         vwap1000=g.vwap.iloc[pos[599]], rth_h=H.max(), rth_l=L.min(), rth_c=C[-1],
                         H=H, L=L, C=C, m=m)
    return rows


def race(H, L, m, t_from, px, k):
    sel = (m >= t_from) & (m <= 659)
    up = np.flatnonzero(H[sel] >= px + k); dn = np.flatnonzero(L[sel] <= px - k)
    iu = up[0] if len(up) else 10**9; idn = dn[0] if len(dn) else 10**9
    return np.nan if iu == idn else float(iu < idn)


def build():
    r = outcomes()
    rows = day_rows(r)
    s = session_table(load_nq()).set_index("sess")
    D = pd.DataFrame({k: {kk: v for kk, v in x.items() if kk not in ("H", "L", "C", "m")} for k, x in rows.items()}).T
    D = D.astype(float)
    rng = (D.rth_h - D.rth_l)
    D["atr"] = rng.rolling(14, min_periods=10).mean().shift(1)
    D = D.join(s[["open_vs_pd", "gap_b", "on_dir", "ldn_vs_asia", "pos_at_8", "orb_dir", "orb_close_pos"]], how="left")
    # prior day: direction, close location, candle relation (C2 vs C1)
    pc, ph, pl, po = D.rth_c.shift(1), D.rth_h.shift(1), D.rth_l.shift(1), D.o930.shift(1)
    D["prev_day"] = np.where(pc > po, "prior RTH bull", "prior RTH bear")
    D["prev_close_loc"] = pd.cut((pc - pl) / (ph - pl), [-.01, 1/3, 2/3, 1.01], labels=["closed low third", "closed mid third", "closed high third"]).astype(object)
    p2h, p2l = D.rth_h.shift(2), D.rth_l.shift(2)
    D["prev_candle"] = np.select([(ph > p2h) & (pc > p2h), (pl < p2l) & (pc < p2l), (ph <= p2h) & (pl >= p2l)],
                                 ["took prior high, closed above", "took prior low, closed below", "inside day"], "other")
    for prof in ("rth", "eth"):
        v = pd.read_parquet(OUT / f"value_area_opens_{prof}.parquet")
        v = v[v.open_time == "09:30 open"].set_index("sess")
        D[f"va_{prof}"] = v["loc"].reindex(D.index).str.replace("inside Q", "inside VA Q")
        D[f"poc_{prof}"] = np.where(v.open.reindex(D.index) > v.poc.reindex(D.index), "above POC", "below POC")
        D.loc[v.reindex(D.index).open.isna().to_numpy(), [f"va_{prof}", f"poc_{prof}"]] = None
    # intra-window states
    D["vwap_1000"] = np.where(D.c1000 > D.vwap1000, "10:00 above VWAP", "10:00 below VWAP")
    D["first30"] = np.where(D.c1000 > D.o930, "09:30-10:00 up", "09:30-10:00 down")
    # first 5m FVG 09:30-10:00
    f = pd.read_parquet(OUT / "fvg_rth_signals.parquet")
    f = f[(f.tf == "5m") & (f.signal == "first FVG") & (f.sig_min < 600)].set_index("day")
    D["fvg5"] = f.dir.reindex(D.index).map({"bull": "first 5m FVG bull", "bear": "first 5m FVG bear"})
    D["fvg5_min"] = f.sig_min.reindex(D.index)
    # outcomes per decision time
    k = 0.25 * D.atr
    out = {"0930": [], "0945": [], "1000": [], "fvg": []}
    for day in D.index:
        x = rows[day]; kk = k.get(day)
        out["0930"].append(race(x["H"], x["L"], x["m"], 570, x["o930"], kk))
        out["0945"].append(race(x["H"], x["L"], x["m"], 585, x["c945"], kk))
        out["1000"].append(race(x["H"], x["L"], x["m"], 600, x["c1000"], kk))
        fm = D.at[day, "fvg5_min"]
        if np.isfinite(fm):
            j = np.flatnonzero(x["m"] == fm - 1)
            px = x["C"][j[0]] if len(j) else np.nan
            out["fvg"].append((race(x["H"], x["L"], x["m"], fm, px, kk), px))
        else:
            out["fvg"].append((np.nan, np.nan))
    D["race_0930"], D["race_0945"], D["race_1000"] = out["0930"], out["0945"], out["1000"]
    D["race_fvg"] = [a for a, _ in out["fvg"]]; D["px_fvg"] = [b for _, b in out["fvg"]]
    D["ret_0930"] = (D.c1100 - D.o930) / D.atr
    D["ret_0945"] = (D.c1100 - D.c945) / D.atr
    D["ret_1000"] = (D.c1100 - D.c1000) / D.atr
    D["ret_fvg"] = (D.c1100 - D.px_fvg) / D.atr
    D = D[D.atr.notna()]
    D["period"] = np.where(D.index < SPLIT, "dev", "later")
    return D


def score(D, col, h):
    ret, rc = D[f"ret_{h}"], D[f"race_{h}"]
    allm = ret.mean()
    rows = []
    for val, g in D[D[col].notna()].groupby(col):
        x = g[f"ret_{h}"].dropna()
        if len(x) < 40:
            continue
        yr = x.groupby(x.index.year).mean() - allm
        d = g[g.period == "dev"]; l = g[g.period == "later"]
        rows.append({"conditioner": col, "value": val, "n": len(x),
                     "up %": 100 * (x > 0).mean(), "race up %": 100 * g[f"race_{h}"].mean(),
                     "ret ATR": x.mean(), "t": x.mean() / (x.std() / np.sqrt(len(x))),
                     "vs all t": (x.mean() - allm) / (x.std() / np.sqrt(len(x))),
                     "years": f"{int((np.sign(yr) == np.sign(x.mean() - allm)).sum())}/{len(yr)}",
                     "dev ret": d[f"ret_{h}"].mean(), "later ret": l[f"ret_{h}"].mean(), "later n": len(l)})
    return rows


if __name__ == "__main__":
    D = build()
    D.drop(columns=[c for c in D.columns if c.startswith("px_")]).to_parquet(OUT / "bias_90.parquet")
    print(f"days {len(D)}  ({D.index.min().date()} -> {D.index.max().date()})\n")
    base = {h: (100 * (D[f'ret_{h}'] > 0).mean(), 100 * D[f'race_{h}'].mean(), D[f'ret_{h}'].mean()) for h in ("0930", "0945", "1000")}
    print("ALL DAYS (baseline):", {h: f"up {a:.1f}% race {b:.1f}% ret {c:+.3f} ATR" for h, (a, b, c) in base.items()}, "\n")
    groups = [("KNOWN AT 09:30 -> 09:30-11:00", "0930",
               ["open_vs_pd", "gap_b", "va_rth", "va_eth", "poc_rth", "poc_eth", "pos_at_8", "on_dir", "ldn_vs_asia",
                "prev_day", "prev_close_loc", "prev_candle"]),
              ("ORB (09:30-09:45) -> 09:45-11:00", "0945", ["orb_dir", "orb_close_pos"]),
              ("STATE AT 10:00 -> 10:00-11:00", "1000", ["vwap_1000", "first30"]),
              ("FIRST 5m FVG (09:30-10:00) -> signal-11:00", "fvg", ["fvg5"])]
    allrows = []
    for title, h, cols in groups:
        rows = [r_ for c in cols for r_ in score(D, c, h)]
        allrows += [dict(r_, window=h) for r_ in rows]
        print("=" * 130, f"\n{title}\n")
        print(pd.DataFrame(rows).set_index(["conditioner", "value"]).round(3).to_string(), "\n")
    A = pd.DataFrame(allrows)
    A.to_csv(OUT / "bias_90_cells.csv", index=False)
    print("=" * 130, f"\nCELLS TESTED: {len(A)}.  |vs all t| > 2: {(A['vs all t'].abs() > 2).sum()} "
          f"(about {0.046 * len(A):.1f} expected by chance)\n")
    print(A.reindex(A["vs all t"].abs().sort_values(ascending=False).index).head(12)[
        ["window", "conditioner", "value", "n", "up %", "race up %", "ret ATR", "vs all t", "years", "dev ret", "later ret", "later n"]].round(3).to_string(index=False))
