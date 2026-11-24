import os, sys, math, warnings
import numpy as np
import pandas as pd

# ========================================================================
# data.py
# ========================================================================
NY = "America/New_York"
OHLC = ["Open", "High", "Low", "Close"]

# Verified per-file source timezones (see detect_timezone to check a new file).
KNOWN_TZ = {
    "nq_15y_1min.csv": NY,
    "GC_15y_1min.csv": "UTC",
}


def _read_raw(path, usecols=None):
    """Read the Date,Time,O,H,L,C[,Volume] export as-is (no tz handling)."""
    cols = usecols or ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
    head = pd.read_csv(path, nrows=0)
    cols = [c for c in cols if c in head.columns]
    dtypes = {c: "float64" for c in cols if c in OHLC + ["Volume"]}
    return pd.read_csv(path, usecols=cols, dtype={**dtypes, "Date": str, "Time": str})


def detect_timezone(path, verbose=True):
    """
    Infer the file's timezone from the CME daily maintenance break (17:00-18:00 ET).

    Returns "America/New_York" if the emptiest hour sits at 17:00 in both winter
    and summer, "UTC" if it shifts by one hour between them (tracking the ET-UTC
    offset), else None - in which case inspect it by hand before trusting it.
    """
    df = _read_raw(path, usecols=["Date", "Time"])
    hour = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
    month = df["Date"].str.slice(5, 7).astype(int)

    def emptiest(mask):
        sub = hour[mask]
        if len(sub) < 1000:
            return None
        return int(sub.value_counts().reindex(range(24), fill_value=0).idxmin())

    winter = emptiest(month.isin([12, 1, 2]))
    summer = emptiest(month.isin([6, 7, 8]))

    if winter is None or summer is None:
        tz = None
    elif winter == summer == 17:
        tz = NY
    elif (winter - summer) % 24 == 1:
        tz = "UTC"
    else:
        tz = None

    if verbose:
        print(f"{os.path.basename(path)}: emptiest hour winter={winter}:00 "
              f"summer={summer}:00 -> {tz or 'UNRECOGNISED - inspect manually'}")
    return tz


def load_bars(path, tz=None, start=None, end=None, symbol=None):
    """
    Load 1-minute bars as the engine's expected frame.

    tz    : source timezone of the timestamps. None = look up KNOWN_TZ by
            filename, else auto-detect. Pass explicitly for a new file.
    start/end : optional ET date strings, e.g. "2024-01-01". Filtering happens
            AFTER conversion, so the bounds mean what you'd expect.

    Returns columns Datetime (tz-aware ET), Open, High, Low, Close [, Volume],
    sorted, de-duplicated on Datetime.
    """
    base = os.path.basename(path)
    if tz is None:
        tz = KNOWN_TZ.get(base) or detect_timezone(path, verbose=False)
    if tz is None:
        raise ValueError(
            f"could not determine the timezone of {base}. Run "
            f"detect_timezone() and pass tz= explicitly - guessing would "
            f"silently shift every session boundary.")

    df = _read_raw(path)
    naive = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y-%m-%d %H:%M:%S")

    if tz == NY:
        # Wall-clock ET. The spring-forward hour does not exist and the
        # fall-back hour is ambiguous; the 17:00-18:00 CME break means the
        # 02:00 DST switch sits inside live trading, so resolve rather than raise.
        dt = naive.dt.tz_localize(NY, ambiguous=True, nonexistent="shift_forward")
    else:
        dt = naive.dt.tz_localize(tz).dt.tz_convert(NY)

    out = pd.DataFrame({"Datetime": dt})
    for c in OHLC:
        out[c] = df[c].to_numpy()
    if "Volume" in df.columns:
        out["Volume"] = df["Volume"].to_numpy()

    out = (out.dropna(subset=["Datetime"] + OHLC)
              .drop_duplicates("Datetime")
              .sort_values("Datetime", kind="mergesort")
              .reset_index(drop=True))

    if start is not None:
        out = out[out["Datetime"] >= pd.Timestamp(start, tz=NY)]
    if end is not None:
        out = out[out["Datetime"] <= pd.Timestamp(end, tz=NY)]
    out = out.reset_index(drop=True)
    out.attrs["symbol"] = symbol or base.split("_")[0].upper()
    out.attrs["source_tz"] = tz
    return out


def audit_bars(df, label=""):
    """
    Sanity report before you trust a backtest on this frame. Checks the things
    that quietly break range logic: session coverage, duplicate or missing
    minutes, zero/negative prices, and OHLC consistency.
    """
    d = df["Datetime"]
    sess = (d - pd.Timedelta(hours=18)).dt.date      # 18:00 ET session anchor
    bad_ohlc = (
        (df["High"] < df["Low"])
        | (df["High"] < df[["Open", "Close"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close"]].min(axis=1))
    ).sum()
    rows = {
        "bars": len(df),
        "first bar (ET)": str(d.iloc[0]),
        "last bar (ET)": str(d.iloc[-1]),
        "sessions": sess.nunique(),
        "duplicate timestamps": int(df["Datetime"].duplicated().sum()),
        "non-positive prices": int((df[OHLC] <= 0).any(axis=1).sum()),
        "OHLC inconsistencies": int(bad_ohlc),
        "median bars/session": int(sess.value_counts().median()),
        "sessions < 600 bars": int((sess.value_counts() < 600).sum()),
    }
    width = max(len(k) for k in rows)
    print(f"--- bar audit {label or df.attrs.get('symbol', '')} ---")
    for k, v in rows.items():
        print(f"  {k:<{width}} : {v}")
    return rows

# ========================================================================
# ranges.py
# ========================================================================
NY = "America/New_York"
ANCHOR_HOUR = 18

# The MTF notebook's open-position buckets, kept identical so results line up.
OP_EDGES = [-np.inf, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, np.inf]
OP_LABELS = ["below pLow", "0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
             "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0", "pHigh or above"]


def _session_date(dt: pd.Series) -> pd.Series:
    """18:00-anchored trading date. A bar at/after 18:00 belongs to the next day."""
    return (dt - pd.Timedelta(hours=ANCHOR_HOUR)).dt.date


def build_ranges(bars: pd.DataFrame, duration: int, min_coverage: float = 0.60,
                 width_window: int = 250, atr_window: int = 14,
                 atr_rank_window: int = 250) -> pd.DataFrame:
    """
    Aggregate 1-minute bars into `duration`-minute ranges and attach features.

    bars         : Datetime (tz-aware ET), Open/High/Low/Close[/Volume]
    min_coverage : drop ranges holding fewer than this fraction of their
                   expected 1-min bars. Holiday half-days and the ragged ends of
                   a data file otherwise produce ranges built from a handful of
                   bars, which then dominate thin cells.
    width_window : trailing completed ranges used for width percentile rank.

    Returns one row per range, ordered by time.
    """
    if duration < 10:
        raise ValueError(f"duration must be >= 10 minutes, got {duration}")

    d = bars.loc[:, ["Datetime", "Open", "High", "Low", "Close"]].copy()
    d["Volume"] = bars["Volume"].to_numpy() if "Volume" in bars.columns else np.nan

    dt = d["Datetime"]
    mins = dt.dt.hour * 60 + dt.dt.minute
    since_anchor = (mins - ANCHOR_HOUR * 60) % 1440          # 0 at 18:00 ET
    sess = _session_date(dt)

    # A range is identified by its session date plus its slot index within the
    # session, so ranges never straddle the 18:00 boundary.
    slot_idx = (since_anchor // duration).astype(int)
    d["_sess"] = sess
    d["_slot_idx"] = slot_idx
    d["_rid"] = pd.factorize(
        pd.Series(sess).astype(str) + "_" + slot_idx.astype(str), sort=False)[0]

    g = d.groupby("_rid", sort=True)
    r = g.agg(
        start=("Datetime", "first"),
        end=("Datetime", "last"),
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
        n_bars=("Close", "size"),
    )
    r["sess"] = g["_sess"].first().to_numpy()
    r["slot_idx"] = g["_slot_idx"].first().to_numpy()

    # Which extreme formed first: compare the bar index of the high and the low.
    r["high_at"] = g["High"].idxmax().to_numpy()
    r["low_at"] = g["Low"].idxmin().to_numpy()
    r["formed_first"] = np.where(r["high_at"] < r["low_at"], "High", "Low")

    # Volume split either side of the range's own midpoint. Needs the mid, which
    # is only known once the range closes, so map it back onto the bars.
    r["mid"] = (r["High"] + r["Low"]) / 2.0
    mid_per_bar = d["_rid"].map(r["mid"])
    upper = d["Volume"].where(d["Close"] > mid_per_bar, 0.0)
    lower = d["Volume"].where(d["Close"] <= mid_per_bar, 0.0)
    r["vol_upper"] = upper.groupby(d["_rid"]).sum().to_numpy()
    r["vol_lower"] = lower.groupby(d["_rid"]).sum().to_numpy()

    r = r.reset_index(drop=True)

    # --- coverage filter -----------------------------------------------------
    r["coverage"] = r["n_bars"] / float(duration)
    r["ok_coverage"] = r["coverage"] >= min_coverage

    # --- geometry ------------------------------------------------------------
    r["width"] = r["High"] - r["Low"]
    r["width_pct_px"] = 100.0 * r["width"] / r["Open"].replace(0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r["close_pos"] = (r["Close"] - r["Low"]) / r["width"].replace(0, np.nan)
    r["vol_balance"] = (r["vol_upper"] - r["vol_lower"]) / \
                       (r["vol_upper"] + r["vol_lower"]).replace(0, np.nan)

    # --- clock / calendar ----------------------------------------------------
    # `slot` must be the NOMINAL window boundary, not the timestamp of the first
    # bar that happens to be present. A window starting 23:45 whose first bars
    # are missing would otherwise be labelled 23:53 and become its own spurious
    # cell with n=1, polluting the grid and inflating the multiple-comparison
    # count. Derive it from the slot index instead.
    nominal = (ANCHOR_HOUR * 60 + r["slot_idx"] * duration) % 1440
    r["slot"] = (nominal // 60).astype(int).map("{:02d}".format) + ":" + \
                (nominal % 60).astype(int).map("{:02d}".format)
    r["dow"] = r["start"].dt.day_name()
    r["duration"] = duration

    # --- relation to the PREVIOUS range -------------------------------------
    # Only meaningful when the previous range is directly adjacent in time, the
    # same requirement the Range Probability indicator enforces via `sAdj`.
    pH, pL, pO, pC = (r[c].shift(1) for c in ("High", "Low", "Open", "Close"))
    p_end = r["end"].shift(1)
    gap_ok = (r["start"] - p_end).dt.total_seconds().between(0, 60 * duration)
    r["adjacent"] = gap_ok & r["ok_coverage"] & r["ok_coverage"].shift(1).fillna(False)

    pw = (pH - pL).replace(0, np.nan)
    r["op"] = (r["Open"] - pL) / pw                       # open position, 0=pLow 1=pHigh
    r["op_bucket"] = pd.cut(r["op"], bins=OP_EDGES, labels=OP_LABELS, right=False)
    r["prev_high"], r["prev_low"] = pH, pL
    r["prev_open"], r["prev_close"] = pO, pC
    r["prev_width"] = pw

    # Overlap / engulf relation. Cheap, and not in any of the existing notebooks.
    inside = (r["High"] <= pH) & (r["Low"] >= pL)
    engulf = (r["High"] >= pH) & (r["Low"] <= pL)
    above = r["Low"] > pH
    below = r["High"] < pL
    r["overlap_rel"] = np.select(
        [below, above, inside, engulf], ["gap below", "gap above", "inside prev", "engulfs prev"],
        default="overlaps")
    inter = (np.minimum(r["High"], pH) - np.maximum(r["Low"], pL)).clip(lower=0)
    r["overlap_frac"] = inter / pw

    # --- trailing normalisations (no lookahead: shift(1) then rolling) -------
    w = r["width"].where(r["ok_coverage"])
    trail = w.shift(1).rolling(width_window, min_periods=50)
    r["width_pct_rank"] = _trailing_pct_rank(w, width_window, min_periods=50)
    r["width_z"] = (w - trail.mean()) / trail.std(ddof=0)
    r["width_bucket"] = pd.cut(r["width_pct_rank"], [0, .2, .4, .6, .8, 1.0],
                               labels=["1 tightest", "2", "3", "4", "5 widest"],
                               include_lowest=True)

    rv = r["Volume"].where(r["ok_coverage"])
    r["rel_volume"] = rv / rv.shift(1).rolling(width_window, min_periods=50).median()

    # ATR on the range series itself, Wilder-smoothed, then percentile-ranked.
    prev_close = r["Close"].shift(1)
    tr = pd.concat([r["High"] - r["Low"],
                    (r["High"] - prev_close).abs(),
                    (r["Low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_window, adjust=False, min_periods=atr_window).mean()
    r["atr"] = atr
    r["atr_pct_rank"] = _trailing_pct_rank(atr, atr_rank_window, min_periods=50)
    r["atr_zone"] = pd.cut(r["atr_pct_rank"], [0, .25, .5, .75, 1.0],
                           labels=["1 low", "2 mid-low", "3 mid-high", "4 high"],
                           include_lowest=True)

    return r


def _trailing_pct_rank(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """
    Fraction of the TRAILING `window` values strictly below the current value.

    Uses only prior observations, so it is safe to condition on. Equivalent in
    spirit to Pine's ta.percentrank but expressed as a 0-1 fraction.
    """
    v = s.to_numpy(dtype="float64")
    n = len(v)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window)
        past = v[lo:i]
        past = past[~np.isnan(past)]
        if len(past) >= min_periods and not np.isnan(v[i]):
            out[i] = (past < v[i]).mean()
    return pd.Series(out, index=s.index)


def build_all(bars: pd.DataFrame, durations=(15, 30, 60, 120, 240), **kw) -> pd.DataFrame:
    """Build ranges for several durations and stack them."""
    frames = [build_ranges(bars, d, **kw) for d in durations]
    out = pd.concat(frames, ignore_index=True)
    out.attrs["symbol"] = bars.attrs.get("symbol", "UNKNOWN")
    return out

# ========================================================================
# outcomes.py
# ========================================================================
TARGETS = ("mid", "range_open", "far_edge")


def forward_outcomes(bars: pd.DataFrame, ranges: pd.DataFrame,
                     horizon_mult: float = 3.0, session_end=None
                     ) -> pd.DataFrame:
    """
    Attach forward outcomes to `ranges` (output of ranges.build_ranges).

    horizon_mult : forward window length, as a multiple of the range duration.
    session_end  : optional "HH:MM" ET cap, applied on top of the horizon. None
                   lets the horizon run across the 17:00-18:00 break.

    Returns `ranges` with the outcome columns appended.
    """
    # Work in tz-naive UTC datetime64 internally: a tz-aware Series converts to
    # object dtype under .to_numpy(), which breaks timedelta arithmetic.
    b_dt = _naive(bars["Datetime"])
    b_hi = bars["High"].to_numpy(dtype="float64")
    b_lo = bars["Low"].to_numpy(dtype="float64")
    n_bars = len(b_dt)

    r = ranges.reset_index(drop=True)
    duration = int(r["duration"].iloc[0])
    horizon = pd.Timedelta(minutes=duration * horizon_mult)

    r_end = _naive(r["end"])
    # forward window starts at the first bar strictly after the range's last bar
    start_i = np.searchsorted(b_dt, r_end, side="right")
    stop_ts = r_end + horizon.to_timedelta64()
    if session_end is not None:
        cap = _naive(_session_end_ts(r["end"], session_end))
        stop_ts = np.minimum(stop_ts, cap)
    stop_i = np.searchsorted(b_dt, stop_ts, side="right")

    hi = r["High"].to_numpy(dtype="float64")
    lo = r["Low"].to_numpy(dtype="float64")
    op = r["Open"].to_numpy(dtype="float64")
    mid = ((hi + lo) / 2.0)
    width = (hi - lo)

    n = len(r)
    out = {
        "break_side": np.full(n, "", dtype=object),
        "break_ambiguous": np.zeros(n, dtype=bool),
        "mins_to_break": np.full(n, np.nan),
        "cont_R": np.full(n, np.nan),          # continuation beyond the edge, in R
        "mae_to_mid_R": np.full(n, np.nan),    # walk-away adverse excursion, in R
        "terminal_R": np.full(n, np.nan),      # where price ended vs the broken edge
        "fwd_bars": np.zeros(n, dtype=int),
    }
    for t in TARGETS:
        out[f"rev_{t}"] = np.full(n, np.nan)        # 1/0, NaN if no break
        out[f"mins_to_{t}"] = np.full(n, np.nan)

    for k in range(n):
        s, e = int(start_i[k]), int(stop_i[k])
        if s >= n_bars or e <= s or not np.isfinite(width[k]) or width[k] <= 0:
            continue
        out["fwd_bars"][k] = e - s
        H, L = b_hi[s:e], b_lo[s:e]

        up = H > hi[k]
        dn = L < lo[k]
        i_up = int(np.argmax(up)) if up.any() else -1
        i_dn = int(np.argmax(dn)) if dn.any() else -1

        if i_up < 0 and i_dn < 0:
            out["break_side"][k] = "none"
            continue

        if i_up >= 0 and i_dn >= 0 and i_up == i_dn:
            # same bar pierced both edges: take the nearer edge to that bar's
            # own range, flag it, and carry on.
            out["break_ambiguous"][k] = True
            bar_mid = (H[i_up] + L[i_up]) / 2.0
            side = "High" if abs(bar_mid - hi[k]) <= abs(bar_mid - lo[k]) else "Low"
            bi = i_up
        elif i_dn < 0 or (i_up >= 0 and i_up < i_dn):
            side, bi = "High", i_up
        else:
            side, bi = "Low", i_dn

        out["break_side"][k] = side
        out["mins_to_break"][k] = (b_dt[s + bi] - r_end[k]) / np.timedelta64(1, "m")

        post_H, post_L = H[bi:], L[bi:]
        if side == "High":
            edge, far = hi[k], lo[k]
            out["cont_R"][k] = (post_H.max() - edge) / width[k]
            out["terminal_R"][k] = (b_hi[e - 1] + b_lo[e - 1]) / 2.0
            out["terminal_R"][k] = (out["terminal_R"][k] - edge) / width[k]
            tgt_vals = {"mid": mid[k], "range_open": op[k], "far_edge": far}
            # reversion is a move DOWN to the target, strictly after the break bar
            for t, lvl in tgt_vals.items():
                hit = np.flatnonzero(post_L[1:] <= lvl)
                _record(out, k, t, hit, bi, s, b_dt, r_end)
            # walk-away: adverse (further up) only until the mid is reached
            stop_at = _first_idx(post_L[1:] <= mid[k])
            seg = post_H[:stop_at + 2] if stop_at is not None else post_H
            out["mae_to_mid_R"][k] = (seg.max() - edge) / width[k]
        else:
            edge, far = lo[k], hi[k]
            out["cont_R"][k] = (edge - post_L.min()) / width[k]
            tm = (b_hi[e - 1] + b_lo[e - 1]) / 2.0
            out["terminal_R"][k] = (edge - tm) / width[k]
            tgt_vals = {"mid": mid[k], "range_open": op[k], "far_edge": far}
            for t, lvl in tgt_vals.items():
                hit = np.flatnonzero(post_H[1:] >= lvl)
                _record(out, k, t, hit, bi, s, b_dt, r_end)
            stop_at = _first_idx(post_H[1:] >= mid[k])
            seg = post_L[:stop_at + 2] if stop_at is not None else post_L
            out["mae_to_mid_R"][k] = (edge - seg.min()) / width[k]

    res = r.copy()
    for col, arr in out.items():
        res[col] = arr
    res["broke"] = res["break_side"].isin(["High", "Low"])
    res["horizon_mult"] = horizon_mult

    # Forward coverage. Slots late in the session have their horizon cut short by
    # the 17:00 close and the maintenance break, so they get fewer chances to
    # break or revert. Without this, "did not break" at 16:45 looks like a
    # signal when it is really an absence of data.
    expected_fwd = duration * horizon_mult
    res["fwd_coverage"] = res["fwd_bars"] / expected_fwd
    res["ok_forward"] = res["fwd_coverage"] >= 0.90
    res["usable"] = res["ok_coverage"] & res["ok_forward"] & res["adjacent"]

    # Reversion flags are only defined conditional on a break.
    for t in TARGETS:
        res.loc[~res["broke"], f"rev_{t}"] = np.nan

    # Magnitude-conditioned outcomes. A 70% direction call that travels 0.2R is
    # not a trade idea, so these ask whether the break actually went somewhere.
    broke = res["broke"].to_numpy()
    for thr in (0.5, 1.0, 2.0):
        tag = f"cont_{str(thr).replace('.', 'p')}R"
        res[tag] = np.where(broke, (res["cont_R"].to_numpy() >= thr).astype("float64"), np.nan)
    is_up = (res["break_side"] == "High").to_numpy()
    res["up_and_1R"] = np.where(broke, (is_up & (res["cont_R"].to_numpy() >= 1.0)).astype("float64"), np.nan)
    res["dn_and_1R"] = np.where(broke, (~is_up & (res["cont_R"].to_numpy() >= 1.0)).astype("float64"), np.nan)
    # Did it continue rather than revert? The cleanest breakout-vs-reversion split.
    res["cont_not_rev"] = np.where(
        broke, ((res["cont_R"].to_numpy() >= 1.0) & (res["rev_mid"].to_numpy() == 0)).astype("float64"), np.nan)
    return res


def _naive(s: pd.Series) -> np.ndarray:
    """tz-aware datetime Series -> tz-naive UTC datetime64[ns] array."""
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.to_numpy("datetime64[ns]")


def _first_idx(mask):
    nz = np.flatnonzero(mask)
    return int(nz[0]) if len(nz) else None


def _record(out, k, t, hit, bi, s, b_dt, r_end):
    if len(hit):
        out[f"rev_{t}"][k] = 1.0
        j = s + bi + 1 + int(hit[0])
        out[f"mins_to_{t}"][k] = (b_dt[j] - r_end[k]) / np.timedelta64(1, "m")
    else:
        out[f"rev_{t}"][k] = 0.0


def _session_end_ts(end: pd.Series, session_end: str) -> pd.Series:
    """Timestamp of `session_end` on the trading day each range belongs to."""
    hh, mm = (int(x) for x in session_end.split(":"))
    anchored = end - pd.Timedelta(hours=18)
    day = anchored.dt.normalize()
    same = day + pd.Timedelta(hours=18) + pd.Timedelta(hours=hh, minutes=mm)
    nxt = same + pd.Timedelta(days=1)
    return pd.Series(np.where(same > end, same, nxt), index=end.index)

# ========================================================================
# cells.py
# ========================================================================
DEFAULT_BY = ["duration", "slot", "op_bucket"]


# ---------------------------------------------------------------- utilities --

def _codes(df: pd.DataFrame, by) :
    by = list(by)
    key = df[by[0]].astype(str)                 # column-wise join: same keys,
    for c in by[1:]:                            # ~100x faster than agg(axis=1)
        key = key + "|" + df[c].astype(str)
    codes, uniq = pd.factorize(key, sort=True)
    idx = pd.MultiIndex.from_tuples([tuple(u.split("|")) for u in uniq], names=list(by))
    return codes, idx


def _rates(codes: np.ndarray, y: np.ndarray, k: int) :
    """Mean of y per cell code, plus the count. y must be finite."""
    cnt = np.bincount(codes, minlength=k).astype("float64")
    tot = np.bincount(codes, weights=y, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tot / np.where(cnt > 0, cnt, np.nan), cnt


def wilson(p, n, z=1.96):
    """Wilson interval. Kept for comparison against the block bootstrap."""
    p = np.asarray(p, dtype="float64"); n = np.asarray(n, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        d = 1 + z**2 / n
        c = (p + z**2 / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def bh_fdr(p: np.ndarray, alpha: float = 0.10):
    """Benjamini-Hochberg. Returns (reject, qvalue)."""
    p = np.asarray(p, dtype="float64")
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    rej = np.zeros(p.shape, dtype=bool)
    pv = p[ok]
    if pv.size == 0:
        return rej, q
    order = np.argsort(pv)
    ranked = pv[order]
    m = pv.size
    qv = ranked * m / (np.arange(m) + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]          # enforce monotonicity
    qv = np.clip(qv, 0, 1)
    out_q = np.empty(m); out_q[order] = qv
    q[ok] = out_q
    rej[ok] = out_q <= alpha
    return rej, q


# -------------------------------------------------------------- the scoring --

def score_cells(df: pd.DataFrame, outcome: str, by=DEFAULT_BY, *,
                min_n: int = 50, shrink_k: float = 150.0,
                split_date="2019-06-01",
                n_boot: int = 400, n_perm: int = 400,
                block_col: str = "sess", seed: int = 0,
                alpha: float = 0.10, magnitude="cont_R",
                min_year_n: int = 20,
                ) :
    """
    Score every cell for a binary `outcome`, with all four defences applied.

    outcome   : a 0/1 column. "break_up" is derived here if requested.
    by        : the conditioners defining a cell.
    magnitude : optional numeric column summarised per cell (median / p90), so a
                direction call can be judged alongside how far it travels.

    Returns (per-cell table, diagnostics dict).
    """
    d = df.copy()
    # Only ranges with full bar coverage, a full forward horizon, and an
    # adjacent previous range. Dropping this filter is what let session-boundary
    # truncation masquerade as signal.
    if "usable" in d.columns:
        d = d[d["usable"]].copy()
    if outcome == "break_up":
        d = d[d["broke"]].copy()
        d["break_up"] = (d["break_side"] == "High").astype("float64")
    elif outcome == "broke":
        d["broke"] = d["broke"].astype("float64")
    elif outcome.startswith(("rev_", "cont_", "up_and_", "dn_and_")):
        # all of these are defined only conditional on a break
        d = d[d["broke"]].copy()

    d = d[np.isfinite(d[outcome].to_numpy(dtype="float64"))]
    for c in by:
        d = d[d[c].notna()]
    if d.empty:
        raise ValueError("no rows left after filtering - check `by` and `outcome`")

    y = d[outcome].to_numpy(dtype="float64")
    codes, idx = _codes(d, by)
    k = len(idx)
    base = float(y.mean())

    rate, cnt = _rates(codes, y, k)
    out = pd.DataFrame({"n": cnt.astype(int), "rate": rate}, index=idx)
    out["base"] = base
    out["edge"] = out["rate"] - base

    # --- train / test ------------------------------------------------------
    if split_date is not None:
        is_tr = (pd.to_datetime(d["sess"].astype(str)) < pd.Timestamp(split_date)).to_numpy()
        for tag, m in (("train", is_tr), ("test", ~is_tr)):
            r_, c_ = _rates(codes[m], y[m], k)
            out[f"{tag}_n"] = c_.astype(int)
            out[f"{tag}_rate"] = r_
        # shrink the TRAIN rate toward the pooled curve of the last `by` level,
        # with the pool itself estimated on train only
        pool_key = by[-1]
        pool = (pd.Series(y[is_tr]).groupby(d[pool_key].to_numpy()[is_tr]).mean())
        pooled = idx.get_level_values(pool_key).map(pool).to_numpy(dtype="float64")
        pooled = np.where(np.isfinite(pooled), pooled, base)
        w = out["train_n"].to_numpy() / (out["train_n"].to_numpy() + shrink_k)
        out["train_pooled"] = pooled
        out["train_shrunk"] = w * np.nan_to_num(out["train_rate"].to_numpy()) + (1 - w) * pooled
        out["holds_oos"] = (np.sign(out["train_shrunk"] - base) ==
                            np.sign(out["test_rate"] - base)) & (out["test_n"] >= 30)

    # --- per-year stability ------------------------------------------------
    yr = pd.to_datetime(d["sess"].astype(str)).dt.year.to_numpy()
    years = np.unique(yr)
    sign_obs = np.sign(out["edge"].to_numpy())
    agree = np.zeros(k); seen = np.zeros(k)
    for Y in years:
        m = yr == Y
        r_, c_ = _rates(codes[m], y[m], k)
        ok = c_ >= min_year_n
        seen += ok
        agree += ok & (np.sign(r_ - base) == sign_obs)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["years_seen"] = seen.astype(int)
        out["year_agree_frac"] = agree / np.where(seen > 0, seen, np.nan)

    # --- magnitude ---------------------------------------------------------
    if magnitude and magnitude in d.columns:
        g = pd.Series(d[magnitude].to_numpy(), index=codes)
        out[f"{magnitude}_med"] = g.groupby(level=0).median().reindex(range(k)).to_numpy()
        out[f"{magnitude}_p90"] = g.groupby(level=0).quantile(.9).reindex(range(k)).to_numpy()

    # --- day-block bootstrap ----------------------------------------------
    rng = np.random.default_rng(seed)
    blk = pd.factorize(d[block_col].astype(str))[0]
    lo, hi = _block_bootstrap(codes, y, blk, k, n_boot, rng)
    out["boot_lo"], out["boot_hi"] = lo, hi
    wl, wh = wilson(out["rate"].to_numpy(), out["n"].to_numpy())
    out["wilson_lo"], out["wilson_hi"] = wl, wh
    with np.errstate(invalid="ignore", divide="ignore"):
        out["ci_inflation"] = (out["boot_hi"] - out["boot_lo"]) / (out["wilson_hi"] - out["wilson_lo"])

    # --- within-day permutation null --------------------------------------
    keep = out["n"].to_numpy() >= min_n
    p_cell, max_null = _permutation_null(codes, y, blk, k, n_perm, base, keep, rng)
    out["p_perm"] = p_cell
    rej, q = bh_fdr(np.where(keep, p_cell, np.nan), alpha=alpha)
    out["q_value"] = q
    out["fdr_pass"] = rej

    obs_max = np.nanmax(np.abs(out["edge"].to_numpy()[keep])) if keep.any() else np.nan
    diag = {
        "outcome": outcome,
        "by": list(by),
        "rows": int(len(d)),
        "cells": int(k),
        "cells_tested": int(keep.sum()),
        "base_rate": base,
        "observed_best_edge": float(obs_max),
        "null_best_edge_p50": float(np.nanpercentile(max_null, 50)),
        "null_best_edge_p95": float(np.nanpercentile(max_null, 95)),
        "selection_pctile": float((max_null < obs_max).mean() * 100) if np.isfinite(obs_max) else np.nan,
        "median_ci_inflation": float(out["ci_inflation"].median(skipna=True)),
        "n_perm": n_perm, "n_boot": n_boot, "min_n": min_n, "alpha": alpha,
    }
    out = out.reset_index()
    out["min_n"] = min_n
    return out, diag


def _block_bootstrap(codes, y, blk, k, n_boot, rng):
    """Percentile CI from resampling whole blocks (trading days) with replacement."""
    nb = blk.max() + 1
    order = np.argsort(blk, kind="stable")
    blk_s, codes_s, y_s = blk[order], codes[order], y[order]
    starts = np.searchsorted(blk_s, np.arange(nb), side="left")
    stops = np.searchsorted(blk_s, np.arange(nb), side="right")
    acc = np.empty((n_boot, k), dtype="float64")
    for b in range(n_boot):
        pick = rng.integers(0, nb, nb)
        cc = np.concatenate([codes_s[starts[i]:stops[i]] for i in pick])
        yy = np.concatenate([y_s[starts[i]:stops[i]] for i in pick])
        r_, _ = _rates(cc, yy, k)
        acc[b] = r_
    return (np.nanpercentile(acc, 2.5, axis=0), np.nanpercentile(acc, 97.5, axis=0))


def _permutation_null(codes, y, blk, k, n_perm, base, keep, rng):
    """
    Shuffle `y` among rows sharing a block, recompute every cell, and record
    both per-cell exceedance counts and the grid-wide maximum |edge|.
    """
    obs_rate, _ = _rates(codes, y, k)
    obs_edge = np.abs(obs_rate - base)
    exceed = np.zeros(k, dtype="float64")
    max_null = np.empty(n_perm, dtype="float64")
    n = len(y)
    for b in range(n_perm):
        # argsort on (block, random) permutes strictly within each block
        perm = np.lexsort((rng.random(n), blk))
        inv = np.lexsort((np.arange(n), blk))
        yy = np.empty(n, dtype="float64")
        yy[inv] = y[perm]
        r_, _ = _rates(codes, yy, k)
        e = np.abs(r_ - base)
        exceed += np.nan_to_num(e >= obs_edge, nan=0.0)
        max_null[b] = np.nanmax(e[keep]) if keep.any() else np.nan
    p = (exceed + 1.0) / (n_perm + 1.0)
    return p, max_null


def shortlist(cells: pd.DataFrame, diag: dict, *, min_n=None, require_fdr=True,
              require_oos=True, min_year_agree=0.6, top=25) -> pd.DataFrame:
    """
    Keep only cells that clear every hurdle, ranked by the shrunk train edge.

    This is deliberately strict. The point is not to find as many cells as
    possible but to end up with a handful you can defend.
    """
    c = cells.copy()
    mn = min_n if min_n is not None else diag["min_n"]
    m = c["n"] >= mn
    if require_fdr and "fdr_pass" in c:
        m &= c["fdr_pass"].fillna(False)
    if require_oos and "holds_oos" in c:
        m &= c["holds_oos"].fillna(False)
    if "year_agree_frac" in c:
        m &= c["year_agree_frac"].fillna(0) >= min_year_agree
    # the bootstrap interval must exclude the base rate
    if "boot_lo" in c:
        m &= (c["boot_lo"] > c["base"]) | (c["boot_hi"] < c["base"])
    out = c[m].copy()
    key = "train_shrunk" if "train_shrunk" in out else "rate"
    out["abs_edge"] = (out[key] - out["base"]).abs()
    return out.sort_values("abs_edge", ascending=False).head(top)

# ========================================================================
# pipeline.py
# ========================================================================
# The outcome families worth asking about, given the goal of picking ranges.
FAMILIES = {
    "break_up":       "P(breaks the HIGH first | it broke) - directional breakout",
    "broke":          "P(breaks either edge at all within the horizon)",
    "rev_mid":        "P(returns to the range mid | it broke)",
    "rev_range_open": "P(returns to the range's own open | it broke)",
    "rev_far_edge":   "P(returns to the far edge | it broke)",
    # magnitude-conditioned: did the break actually go anywhere?
    "cont_1p0R":      "P(travels >= 1 range-width past the edge | it broke)",
    "cont_not_rev":   "P(travels >= 1R AND never returns to mid | it broke)",
}


def build(csv, symbol="NQ", durations=(15, 30, 60, 120, 240), start=None, end=None,
          horizon_mult=3.0, min_coverage=0.60, session_end=None, bars=None,
          verbose=True) -> pd.DataFrame:
    """Build the range table with features and forward outcomes, all durations."""
    if bars is None:
        bars = load_bars(csv, symbol=symbol, start=start, end=end)
    if verbose:
        print(f"{symbol}: {len(bars):,} bars  {bars['Datetime'].iloc[0]} -> {bars['Datetime'].iloc[-1]}")
    frames = []
    for d in durations:
        r = build_ranges(bars, d, min_coverage=min_coverage)
        o = forward_outcomes(bars, r, horizon_mult=horizon_mult, session_end=session_end)
        if verbose:
            br = o["broke"].mean()
            print(f"  {d:>4}-min: {len(o):>7,} ranges   broke {100*br:5.1f}%")
        frames.append(o)
    out = pd.concat(frames, ignore_index=True)
    out.attrs["symbol"] = symbol
    return out


def run(csv=None, symbol="NQ", durations=(15, 30, 60, 120, 240), *,
        by=("duration", "slot", "op_bucket"), families=tuple(FAMILIES),
        min_n=50, n_boot=200, n_perm=200, split_date="2019-06-01",
        alpha=0.10, seed=0, bars=None, ranges=None, verbose=True, **kw) -> dict:
    """Run the whole thing and return ranges, per-family cells, diagnostics, shortlist."""
    rng_tbl = ranges if ranges is not None else build(
        csv, symbol=symbol, durations=durations, bars=bars, verbose=verbose, **kw)

    cells, diag, shorts = {}, {}, []
    for fam in families:
        c, d = score_cells(rng_tbl, fam, by=list(by), min_n=min_n, n_boot=n_boot,
                           n_perm=n_perm, split_date=split_date, alpha=alpha, seed=seed)
        cells[fam], diag[fam] = c, d
        s = shortlist(c, d)
        s.insert(0, "family", fam)
        shorts.append(s)
        if verbose:
            print(f"\n{fam:16s} base {100*d['base_rate']:5.1f}%  "
                  f"tested {d['cells_tested']:>4}  "
                  f"best edge {100*d['observed_best_edge']:5.1f}pp vs null p95 "
                  f"{100*d['null_best_edge_p95']:5.1f}pp  -> survived {len(s)}")
    return {"ranges": rng_tbl, "cells": cells, "diag": diag,
            "shortlist": pd.concat(shorts, ignore_index=True) if shorts else pd.DataFrame()}


def diagnostics_table(diag: dict) -> pd.DataFrame:
    """One row per outcome family: is the best cell better than luck?"""
    rows = []
    for fam, d in diag.items():
        rows.append({
            "family": fam,
            "rows": d["rows"],
            "cells_tested": d["cells_tested"],
            "base_rate_%": round(100 * d["base_rate"], 1),
            "best_edge_pp": round(100 * d["observed_best_edge"], 1),
            "null_best_p50_pp": round(100 * d["null_best_edge_p50"], 1),
            "null_best_p95_pp": round(100 * d["null_best_edge_p95"], 1),
            "best_vs_null": round(d["observed_best_edge"] / d["null_best_edge_p95"], 2)
                            if d["null_best_edge_p95"] else np.nan,
            "beats_null_%ile": round(d["selection_pctile"], 1),
            "ci_inflation": round(d["median_ci_inflation"], 2),
        })
    return pd.DataFrame(rows)


def describe_cell(ranges: pd.DataFrame, row: pd.Series, by=("duration", "slot", "op_bucket")
                  ) -> pd.DataFrame:
    """
    Everything about one shortlisted cell: the probabilities AND the magnitudes,
    because a direction call without follow-through is not a trade idea.
    """
    m = pd.Series(True, index=ranges.index)
    for c in by:
        m &= ranges[c].astype(str) == str(row[c])
    sub = ranges[m]
    brk = sub[sub["broke"]]
    out = {
        **{c: row[c] for c in by},
        "ranges": len(sub),
        "broke_%": round(100 * sub["broke"].mean(), 1),
        "break_high_%": round(100 * (brk["break_side"] == "High").mean(), 1),
        "median_mins_to_break": round(brk["mins_to_break"].median(), 0),
        "cont_R_med": round(brk["cont_R"].median(), 2),
        "cont_R_p90": round(brk["cont_R"].quantile(.9), 2),
        "mae_to_mid_R_med": round(brk["mae_to_mid_R"].median(), 2),
    }
    for t in ("mid", "range_open", "far_edge"):
        out[f"rev_{t}_%"] = round(100 * brk[f"rev_{t}"].mean(), 1)
        out[f"mins_to_{t}_med"] = round(brk[f"mins_to_{t}"].median(), 0)
    return pd.DataFrame([out])