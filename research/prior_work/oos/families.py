"""
Event tables: one row per occurrence, columns = conditions known at decision
time + outcomes measured afterwards. Every table has `sess` (a date) and
`block` (what the permutation null shuffles within).

Outcomes are 0/1 floats. NaN means "not applicable" - e.g. touch_pdh is NaN
when RTH opened above PDH, because it would be trivially 1. Leaving trivial
rows in is how 95% hit rates get manufactured.

Definitions follow your notebooks where they are unambiguous:
  session anchor 18:00 ET, RTH 09:30-16:00, IB 09:30-10:30, ORB 09:30-09:45,
  6-9 range 06:00-09:00, pre-8 = 18:00-08:00, Asia 20:00-00:00, London 02:00-05:00.
"""
import numpy as np
import pandas as pd

from .data import add_clock, DISCOVERY_END


# ------------------------------------------------------------------ helpers --

def _sm(hhmm: str) -> int:
    """Minutes since 18:00 ET for a clock time."""
    h, m = map(int, hhmm.split(":"))
    return (h * 60 + m - 1080) % 1440


def prep(bars):
    b = add_clock(bars)
    b["sm"] = ((b["mod"].astype("int32") - 1080) % 1440).astype("int16")
    return b


def window(b, start, end, min_frac=0.6):
    """OHLC per rth_date for bars in [start, end) clock time, plus the minute
    (since 18:00) the high and low were set. Days with < min_frac coverage -> NaN."""
    s, e = _sm(start), _sm(end) or 1440
    w = b[(b["sm"] >= s) & (b["sm"] < e)]
    g = w.groupby("rth_date", sort=True)
    out = g.agg(o=("Open", "first"), h=("High", "max"), l=("Low", "min"),
                c=("Close", "last"), n=("Close", "size"))
    out["th"] = w["sm"].to_numpy()[w.index.get_indexer(g["High"].idxmax())]
    out["tl"] = w["sm"].to_numpy()[w.index.get_indexer(g["Low"].idxmin())]
    out.loc[out["n"] < min_frac * (e - s)] = np.nan
    return out


def first_touch(b, start, end, level: pd.Series, side: str, after: pd.Series = None):
    """First minute (since 18:00) in [start, end) where price trades through
    `level` (side 'up': High > level, 'dn': Low < level), optionally only after
    minute `after`. NaN if never."""
    s, e = _sm(start), _sm(end) or 1440
    w = b[(b["sm"] >= s) & (b["sm"] < e)]
    lv = w["rth_date"].map(level).to_numpy("float64")
    hit = (w["High"].to_numpy() > lv) if side == "up" else (w["Low"].to_numpy() < lv)
    if after is not None:
        hit &= w["sm"].to_numpy() > w["rth_date"].map(after).to_numpy("float64")
    return w.loc[hit].groupby("rth_date")["sm"].min().reindex(level.index)


def touches(b, start, end, level: pd.Series, after: pd.Series):
    """1.0 if price trades AT level strictly after minute `after`, 0.0 if not,
    NaN where `after` is NaN."""
    s, e = _sm(start), _sm(end) or 1440
    w = b[(b["sm"] >= s) & (b["sm"] < e)]
    lv = w["rth_date"].map(level).to_numpy("float64")
    t0 = w["rth_date"].map(after).to_numpy("float64")
    hit = (w["Low"].to_numpy() <= lv) & (w["High"].to_numpy() >= lv) & (w["sm"].to_numpy() > t0)
    got = w.loc[hit].groupby("rth_date").size().reindex(level.index).fillna(0) > 0
    return got.astype("float64").where(after.notna())


def pos_bins(x, labels):
    return pd.cut(x, [-np.inf, 0, .5, 1, np.inf], labels=labels, right=False).astype(object)


def disc_quintiles(x: pd.Series, dates) -> pd.Series:
    """Quintiles with edges fitted on the DISCOVERY period only, so the holdout
    is bucketed with thresholds it had no say in."""
    d = pd.to_datetime(pd.Series(np.asarray(dates), index=x.index))
    edges = np.nanquantile(x[d <= DISCOVERY_END], [.2, .4, .6, .8])
    return pd.cut(x, [-np.inf, *edges, np.inf],
                  labels=["Q1 narrow", "Q2", "Q3", "Q4", "Q5 wide"]).astype(object)


def _f(x):
    return pd.Series(x).astype("float64")


def _month_block(idx):
    return pd.to_datetime(pd.Series(idx)).dt.strftime("%Y-%m").to_numpy()


# --------------------------------------------------------- session / IB / ORB --

def session_table(bars) -> pd.DataFrame:
    b = prep(bars)
    full = window(b, "18:00", "17:00", 0.5)
    on = window(b, "18:00", "09:30", 0.5)
    pre8 = window(b, "18:00", "08:00", 0.5)
    asia = window(b, "20:00", "00:00", 0.5)
    ldn = window(b, "02:00", "05:00", 0.5)
    r69 = window(b, "06:00", "09:00", 0.6)
    orb = window(b, "09:30", "09:45", 0.8)
    ib = window(b, "09:30", "10:30", 0.8)
    rth = window(b, "09:30", "16:00", 0.9)        # drops early-close days
    ny8 = window(b, "08:00", "16:00", 0.9)

    d = pd.DataFrame(index=rth.dropna(subset=["o"]).index)
    R = rth.reindex(d.index)
    prev = rth.dropna(subset=["o"]).shift(1)
    gap_days = d.index.to_series().diff().dt.days
    prev[gap_days.to_numpy() > 5] = np.nan
    pdh, pdl, pdc = prev["h"], prev["l"], prev["c"]
    atr = (rth["h"] - rth["l"]).dropna().rolling(14, min_periods=10).mean().shift(1).reindex(d.index)

    # roll days: unadjusted front-month series jump at the vendor's roll.
    f = full.reindex(d.index)
    jump = (f["o"] - prev["c"]).abs() / prev["c"]
    mth, dom = d.index.month, d.index.day
    roll = (jump > 0.0025) & np.isin(mth, [3, 6, 9, 12]) & (dom >= 5) & (dom <= 21)

    d["sess"] = d.index
    d["block"] = _month_block(d.index)
    d["dow"] = d.index.day_name().str[:3]
    d["roll_suspect"] = roll.to_numpy()

    # -- conditions known at 09:30
    d["open_vs_pd"] = pos_bins((R["o"] - pdl) / (pdh - pdl),
                               ["below PDL", "lower half PD", "upper half PD", "above PDH"])
    gap = (R["o"] - pdc) / atr
    d["gap_b"] = pd.cut(gap, [-np.inf, -.5, -.15, .15, .5, np.inf],
                        labels=["gap dn >0.5atr", "gap dn", "flat", "gap up", "gap up >0.5atr"]).astype(object)
    d.loc[roll.to_numpy(), "gap_b"] = None
    O = on.reindex(d.index)
    d["on_dir"] = np.where(O["c"] > O["o"], "ON bull", np.where(O["c"] < O["o"], "ON bear", None))
    A, L = asia.reindex(d.index), ldn.reindex(d.index)
    th, tl = L["h"] > A["h"], L["l"] < A["l"]
    d["ldn_vs_asia"] = np.select([th & tl, th, tl], ["took both", "took Asia H", "took Asia L"], "inside Asia")
    d.loc[A["o"].isna() | L["o"].isna(), "ldn_vs_asia"] = None
    S = r69.reindex(d.index)
    d["open_vs_69"] = pos_bins((R["o"] - S["l"]) / (S["h"] - S["l"]),
                               ["below 6-9", "lower 6-9", "upper 6-9", "above 6-9"])

    # -- RTH outcomes from the open
    d["rth_bull"] = _f(R["c"] > R["o"])
    up, dn = gap >= .15, gap <= -.15
    d["gap_filled"] = np.where(up, R["l"] <= pdc, np.where(dn, R["h"] >= pdc, np.nan)).astype("float64")
    d.loc[roll.to_numpy(), "gap_filled"] = np.nan
    d["touch_pdh"] = _f(R["h"] >= pdh).where(R["o"] < pdh)
    d["touch_pdl"] = _f(R["l"] <= pdl).where(R["o"] > pdl)
    mid69 = (S["h"] + S["l"]) / 2
    outside69 = (R["o"] > S["h"]) | (R["o"] < S["l"])
    d["rth_hits_69_mid"] = _f((R["l"] <= mid69) & (R["h"] >= mid69)).where(outside69)

    # -- 08:00: is the overnight extreme the day's extreme?
    P, N = pre8.reindex(d.index), ny8.reindex(d.index)
    px8 = P["c"]
    d["pos_at_8"] = pd.cut((px8 - P["l"]) / (P["h"] - P["l"]), [-np.inf, 1/3, 2/3, np.inf],
                           labels=["8am low third", "8am mid third", "8am high third"]).astype(object)
    d["on_dir_8"] = np.where(px8 > P["o"], "pre8 bull", "pre8 bear")
    d["pre8_high_holds"] = _f(N["h"] <= P["h"]).where(N["o"].notna() & P["o"].notna())
    d["pre8_low_holds"] = _f(N["l"] >= P["l"]).where(N["o"].notna() & P["o"].notna())

    # -- ORB (09:30-09:45), decisions at 09:45
    B = orb.reindex(d.index)
    bw = B["h"] - B["l"]
    d["orb_w_q"] = disc_quintiles(bw / atr, d.index)
    d["orb_dir"] = np.where(B["c"] > B["o"], "ORB bull", "ORB bear")
    d["orb_close_pos"] = pd.cut((B["c"] - B["l"]) / bw, [-np.inf, 1/3, 2/3, np.inf],
                                labels=["closed low third", "closed mid third", "closed high third"]).astype(object)
    _breaks(d, b, "orb", B, "09:45", "16:00", R)

    # -- IB (09:30-10:30), decisions at 10:30
    I = ib.reindex(d.index)
    iw = I["h"] - I["l"]
    d["ib_w_q"] = disc_quintiles(iw / atr, d.index)
    d["ib_dir"] = np.where(I["c"] > I["o"], "IB bull", "IB bear")
    d["ib_close_pos"] = pd.cut((I["c"] - I["l"]) / iw, [-np.inf, 1/3, 2/3, np.inf],
                               labels=["closed low third", "closed mid third", "closed high third"]).astype(object)
    ou = first_touch(b, "09:45", "10:30", B["h"], "up")
    od = first_touch(b, "09:45", "10:30", B["l"], "dn")
    d["orb_first_in_ib"] = np.select([ou.notna() & (od.isna() | (ou < od)), od.notna()],
                                     ["ORB broke up", "ORB broke down"], "ORB held")
    d["ib_vs_pd"] = np.select([(I["h"] > pdh) & (I["l"] < pdl), I["h"] > pdh, I["l"] < pdl],
                              ["IB took PDH+PDL", "IB took PDH", "IB took PDL"], "IB inside PD")
    _breaks(d, b, "ib", I, "10:30", "16:00", R)
    # direction from 10:30 on - conditioning rth_bull on IB direction would count
    # the IB's own move as part of the outcome
    d["post_ib_bull"] = _f(R["c"] > I["c"]).where(I["c"].notna())
    return d


def _breaks(d, b, p, W, start, end, R):
    """Break outcomes of an opening window W over [start, end)."""
    w = W["h"] - W["l"]
    mid = (W["h"] + W["l"]) / 2
    tu = first_touch(b, start, end, W["h"], "up")
    td = first_touch(b, start, end, W["l"], "dn")
    broke = tu.notna() | td.notna()
    up_first = tu.notna() & (td.isna() | (tu < td))
    tb = pd.concat([tu, td], axis=1).min(axis=1)
    ok = W["o"].notna() & R["o"].notna()
    d[f"{p}_broke"] = _f(broke).where(ok)
    d[f"{p}_up_first"] = _f(up_first).where(ok & broke)
    # random-walk expectation of up-first from where the window closed:
    # P(hit H before L) = (c - L) / (H - L) for a driftless walk
    d[f"{p}_up_first_rw"] = ((W["c"] - W["l"]) / w).where(ok & broke)
    d[f"{p}_both_sides"] = _f(tu.notna() & td.notna()).where(ok)
    d[f"{p}_back_to_mid"] = touches(b, start, end, mid, tb).where(ok & broke)
    # extension 0.5x the window's width in the direction of the first break
    hi_after = R["h"]; lo_after = R["l"]
    ext = np.where(up_first, hi_after >= W["h"] + .5 * w, lo_after <= W["l"] - .5 * w)
    d[f"{p}_ext_half"] = pd.Series(ext, index=d.index, dtype="float64").where(ok & broke)
    hold = np.where(up_first, R["c"] > W["h"], R["c"] < W["l"])
    d[f"{p}_close_beyond_break"] = pd.Series(hold, index=d.index, dtype="float64").where(ok & broke)


# ------------------------------------------------------------- hourly sweeps --

def hourly_table(bars, hours=range(9, 16)) -> pd.DataFrame:
    b = prep(bars)
    b["hr"] = (b["mod"] // 60).astype("int8")
    g = b.groupby(["rth_date", "hr"], sort=True)
    h = g.agg(o=("Open", "first"), hi=("High", "max"), lo=("Low", "min"),
              c=("Close", "last"), n=("Close", "size")).reset_index()
    h.loc[h["n"] < 45, ["o", "hi", "lo", "c"]] = np.nan
    h = h.sort_values(["rth_date", "hr"])
    for c in ("hi", "lo", "o", "c"):
        h[f"p_{c}"] = h.groupby("rth_date")[c].shift(1)
    h["p_hr"] = h.groupby("rth_date")["hr"].shift(1)
    h = h[h["hr"].isin(list(hours)) & (h["p_hr"] == h["hr"] - 1)].dropna(subset=["o", "p_hi"])

    out = pd.DataFrame({"sess": h["rth_date"].to_numpy(), "block": h["rth_date"].dt.strftime("%Y-%m-%d").to_numpy()})
    out["hour"] = h["hr"].map(lambda x: f"{x:02d}:00").to_numpy()
    pr = (h["o"] - h["p_lo"]) / (h["p_hi"] - h["p_lo"])
    out["open_pos"] = pos_bins(pr, ["open below prev L", "open lower half", "open upper half",
                                    "open above prev H"]).to_numpy()
    out["prev_dir"] = np.where(h["p_c"] > h["p_o"], "prev hr bull", "prev hr bear")
    sh = (h["hi"] > h["p_hi"]).where(h["o"] <= h["p_hi"])
    sl = (h["lo"] < h["p_lo"]).where(h["o"] >= h["p_lo"])
    out["sweep_prev_high"] = _f(sh).to_numpy()
    out["sweep_prev_low"] = _f(sl).to_numpy()
    one = (sh.fillna(False).astype(bool) ^ sl.fillna(False).astype(bool))
    inside = (h["c"] <= h["p_hi"]) & (h["c"] >= h["p_lo"])
    out["swept_side"] = np.where(one & sh.fillna(False).astype(bool), "swept H only",
                                 np.where(one, "swept L only", None))
    out["sweep_closes_inside"] = _f(inside).where(one).to_numpy()
    pm = (h["p_hi"] + h["p_lo"]) / 2
    out["hits_prev_mid"] = _f((h["lo"] <= pm) & (h["hi"] >= pm)).to_numpy()
    out["hour_bull"] = _f(h["c"] > h["o"]).to_numpy()
    return out.reset_index(drop=True)


# ------------------------------------------------------------ 3-candle setups --

def candle_table(bars, tf: str) -> pd.DataFrame:
    """C1, C2 known; outcome is what C3 does to C2's high and low."""
    b = prep(bars)
    div = {"1h": 60, "4h": 240, "1D": 1440}[tf]
    day = b["rth_date"].to_numpy().astype("datetime64[D]").astype("int64")
    b["k"] = day * 100 + (b["sm"].to_numpy() // div)
    g = b.groupby("k", sort=True)
    c = g.agg(o=("Open", "first"), h=("High", "max"), l=("Low", "min"), c=("Close", "last"),
              n=("Close", "size"), rth_date=("rth_date", "first"), sm0=("sm", "first"))
    c = c[c["n"] >= (0.5 * div if tf != "1D" else 600)]
    c1, c2 = c.shift(2), c.shift(1)

    rel = np.select(
        [(c2["h"] <= c1["h"]) & (c2["l"] >= c1["l"]),
         (c2["h"] > c1["h"]) & (c2["l"] < c1["l"]),
         c2["h"] > c1["h"], c2["l"] < c1["l"]],
        ["inside", "outside", "higher (took C1 H)", "lower (took C1 L)"], None)
    cpos = np.select([c2["c"] > c1["h"], c2["c"] < c1["l"]],
                     ["C2 closed above C1 H", "C2 closed below C1 L"], "C2 closed inside C1")
    col = (np.where(c1["c"] >= c1["o"], "C1 bull", "C1 bear").astype(object) + " / " +
           np.where(c2["c"] >= c2["o"], "C2 bull", "C2 bear").astype(object))

    lv_h = pd.Series(c2["h"].to_numpy(), index=c.index)
    lv_l = pd.Series(c2["l"].to_numpy(), index=c.index)
    bk = b[b["k"].isin(c.index)]
    mh = bk["k"].map(lv_h).to_numpy()
    ml = bk["k"].map(lv_l).to_numpy()
    tu = bk.loc[bk["High"].to_numpy() > mh].groupby("k")["sm"].min().reindex(c.index)
    td = bk.loc[bk["Low"].to_numpy() < ml].groupby("k")["sm"].min().reindex(c.index)

    gap_out = (c["o"] > c2["h"]) | (c["o"] < c2["l"])      # C3 opened outside C2: trivial
    takes_h = _f(tu.notna()).where(c["o"] <= c2["h"])
    takes_l = _f(td.notna()).where(c["o"] >= c2["l"])
    any_ = tu.notna() | td.notna()
    h_first = _f(tu.notna() & (td.isna() | (tu < td))).where(any_ & ~gap_out)

    out = pd.DataFrame({
        "sess": c["rth_date"].to_numpy(),
        "block": (c["rth_date"].dt.strftime("%Y-%m-%d") if tf != "1D"
                  else c["rth_date"].dt.strftime("%Y-%m")).to_numpy(),
        "tf": tf,
        "c3_hour": (((c["sm0"] // div * div + 1080) % 1440) // 60).map(lambda x: f"{int(x):02d}:00").to_numpy(),
        "c2_rel": rel, "c2_close": cpos, "colours": col.to_numpy() if hasattr(col, "to_numpy") else col,
        "c3_takes_c2_high": takes_h.to_numpy(),
        "c3_takes_c2_low": takes_l.to_numpy(),
        "c3_high_first": h_first.to_numpy(),
        "c3_high_first_rw": ((c["o"] - c2["l"]) / (c2["h"] - c2["l"])).where(h_first.notna()).to_numpy(),
        "c3_bull": _f(c["c"] > c["o"]).to_numpy(),
    })
    valid = c1["o"].notna().to_numpy() & c2["o"].notna().to_numpy()
    out = out[valid & pd.notna(out["c2_rel"]).to_numpy()]
    return out.reset_index(drop=True)


# ----------------------------------------------------------- rangelab ranges --

def range_table(bars, durations=(15, 30, 60)) -> pd.DataFrame:
    from . import engine
    r = engine.build(None, symbol="NQ", durations=durations, bars=bars, verbose=True)
    r["w_atr"] = r["width"] / r["atr"]
    sess = pd.to_datetime(r["sess"].astype(str))
    r["w_atr_q"] = None
    for du in durations:
        m = (r["duration"] == du).to_numpy()
        r.loc[m, "w_atr_q"] = disc_quintiles(r.loc[m, "w_atr"], sess[m]).to_numpy()
    for c in ("slot", "op_bucket", "duration"):
        r[c] = r[c].astype(str).where(r[c].notna())
    r["block"] = r["sess"].astype(str)
    r["break_up_rw"] = r["close_pos"]          # random-walk P(high first) from the close
    # rangelab's `sess` is the 18:00 start date; shift to the RTH-date convention
    # so the lockbox boundary is the same for every family.
    r["sess"] = sess + pd.Timedelta(days=1)
    return r
