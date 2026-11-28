"""Key levels per 1-minute bar, using only bars that closed before that bar opened (no lookahead).

Every column is the level's value as known at the OPEN of the bar on that row.

    pdh pdl          prior session's RTH (09:30-15:59) high / low                 (NaN where pd_ok is False)
    pwh pwl          prior week's high / low, all hours                           (NaN where pw_ok is False)
    asia_h asia_l    18:00-01:59 high / low, from 02:00
    lon_h lon_l      02:00-07:59 high / low, from 08:00
    on_h on_l        18:00-09:29 high / low, from 09:30
    vwap_rth         VWAP anchored 09:30 (through the previous bar), RTH only
    vwap_day         VWAP anchored 18:00 (Globex day)
    vwap_wk          VWAP anchored Sunday 18:00
    ib_h ib_l        09:30-10:29 high / low, from 10:30
    ph_h ph_l        previous completed clock hour high / low
    p4_h p4_l        previous completed 4h candle (anchored 18:00: 18, 22, 02, 06, 10, 14)
plus helpers: hm (minute of day), rth (bool), atr (14-session mean RTH range, known at session start).
"""
import numpy as np
import pandas as pd

LEVELS = ["pdh", "pdl", "pwh", "pwl", "asia_h", "asia_l", "lon_h", "lon_l", "on_h", "on_l",
          "vwap_rth", "vwap_day", "vwap_wk", "ib_h", "ib_l", "ph_h", "ph_l", "p4_h", "p4_l"]

RTH0, RTH1 = 9 * 60 + 30, 16 * 60


def _window_hl(df, mask, key):
    g = df[mask].groupby(key)
    return g.high.max(), g.low.min()


def _anchored_vwap(df, key, mask=None):
    """VWAP from the first bar of each `key` group through the PREVIOUS bar (NaN on the group's first bar)."""
    tp = (df.high + df.low + df.close) / 3.0
    v = df.volume.astype(float)
    if mask is not None:
        v = v.where(mask, 0.0)
    pv = (tp * v).groupby(key).cumsum() - tp * v
    vv = v.groupby(key).cumsum() - v
    out = pv / vv.replace(0, np.nan)
    if mask is not None:
        out = out.where(mask)
    return out


def _prev_block(df, block_key):
    """High/low of the previous block (consecutive blocks in data order)."""
    g = df.groupby(block_key, sort=False)
    hi, lo = g.high.max(), g.low.min()
    order = pd.Index(block_key.drop_duplicates())
    prev_hi = pd.Series(hi.reindex(order).shift(1).to_numpy(), index=order)
    prev_lo = pd.Series(lo.reindex(order).shift(1).to_numpy(), index=order)
    return block_key.map(prev_hi).to_numpy(), block_key.map(prev_lo).to_numpy()


def build_levels(df):
    df = df.sort_values("ts").reset_index(drop=True)
    wall = df.ts.dt.tz_localize(None)
    hm = (wall.dt.hour * 60 + wall.dt.minute).to_numpy()
    out = pd.DataFrame(index=df.index)
    out["hm"] = hm
    rth = (hm >= RTH0) & (hm < RTH1)
    out["rth"] = rth
    sess = df.sess

    # ---- prior day RTH + ATR -----------------------------------------------------------------------------
    rh, rl = _window_hl(df, rth, sess)
    sessions = pd.Index(sorted(sess.unique()))
    rh, rl = rh.reindex(sessions), rl.reindex(sessions)
    # skip sessions without RTH bars when shifting (carry the last real RTH)
    pdh, pdl = rh.ffill().shift(1), rl.ffill().shift(1)
    partial = df.groupby(sess).partial.first().reindex(sessions)
    rng = (rh - rl).where(~partial)
    atr = rng.rolling(14, min_periods=10).mean().shift(1).ffill()
    out["pdh"] = sess.map(pdh).where(df.pd_ok).to_numpy()
    out["pdl"] = sess.map(pdl).where(df.pd_ok).to_numpy()
    out["atr"] = sess.map(atr).to_numpy()

    # ---- prior week ----------------------------------------------------------------------------------------
    wk = sess - pd.to_timedelta(sess.dt.weekday, unit="D")
    wh, wl = df.groupby(wk).high.max(), df.groupby(wk).low.min()
    out["pwh"] = wk.map(wh.shift(1)).where(df.pw_ok).to_numpy()
    out["pwl"] = wk.map(wl.shift(1)).where(df.pw_ok).to_numpy()

    # ---- overnight windows -------------------------------------------------------------------------------------
    eve = hm >= 18 * 60
    for name, m, start in (("asia", eve | (hm < 120), 120),
                           ("lon", (hm >= 120) & (hm < 480), 480),
                           ("on", eve | (hm < RTH0), RTH0)):
        h, l = _window_hl(df, m, sess)
        active = (~eve) & (hm >= start)
        out[f"{name}_h"] = np.where(active, sess.map(h), np.nan)
        out[f"{name}_l"] = np.where(active, sess.map(l), np.nan)

    # ---- VWAPs ---------------------------------------------------------------------------------------------
    out["vwap_rth"] = _anchored_vwap(df, sess, pd.Series(rth, index=df.index)).to_numpy()
    out["vwap_day"] = _anchored_vwap(df, sess).to_numpy()
    out["vwap_wk"] = _anchored_vwap(df, wk).to_numpy()

    # ---- IB --------------------------------------------------------------------------------------------------
    ibm = (hm >= RTH0) & (hm < 10 * 60 + 30)
    h, l = _window_hl(df, ibm, sess)
    active = (hm >= 10 * 60 + 30) & ~eve
    out["ib_h"] = np.where(active, sess.map(h), np.nan)
    out["ib_l"] = np.where(active, sess.map(l), np.nan)

    # ---- previous hour / previous 4h ---------------------------------------------------------------------------
    hour_key = wall.dt.floor("h")
    out["ph_h"], out["ph_l"] = _prev_block(df, hour_key)
    mins_since_open = (hm - 18 * 60) % 1440
    blk4 = sess.dt.strftime("%Y%m%d") + "_" + pd.Series(mins_since_open // 240, index=df.index).astype(str)
    out["p4_h"], out["p4_l"] = _prev_block(df, blk4)
    return df, out
