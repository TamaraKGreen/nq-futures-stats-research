"""Data access for the blueprint study: dev / oot / lockbox, one consistent frame.

Columns returned
    ts (tz-aware NY, bar open)  sess (trading date, 18:00 anchor)  open high low close volume
    delta (NaN on lockbox)  partial (holiday / early close / outage session)
    pd_ok  pw_ok   False where prior-day / prior-week levels would straddle an unadjusted contract roll (lockbox)
    roll_adj  constant per session: raw traded price = price - roll_adj (0 on lockbox); contract (blank on lockbox)
    split  'dev' | 'oot' | 'lockbox' | 'warmup' (lookback bars kept only so levels/ATR exist on day one)

Prices
    dev / oot : back-adjusted (raw + roll_adj). Inside one session this equals raw traded prices shifted by a
                constant, so every point distance is exact and multi-day levels are roll-consistent.
    lockbox   : raw front-month as exported (no contract column, so no adjustment). Prior-day levels are masked in
                expiry week, prior-week levels in expiry week and the week after (pd_ok / pw_ok).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "scripts"))

TICK = 0.25
CONTRACTS = 5
PT_VALUE = 2.0 * CONTRACTS          # MNQ $2/pt/contract
COST_PTS = 6.50 / PT_VALUE          # $6.50 round trip for the 5-lot
SLIP = TICK                         # per market entry / stop / time exit

DEV_END = pd.Timestamp("2025-07-09")
LOCKBOX = (pd.Timestamp("2012-01-01"), pd.Timestamp("2019-12-31"))


def _third_friday(year, month):
    d = pd.Timestamp(year=year, month=month, day=15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


def _expiry_weeks(years):
    """Monday of each quarterly expiry week."""
    out = []
    for y in years:
        for m in (3, 6, 9, 12):
            f = _third_friday(y, m)
            out.append(f - pd.Timedelta(days=4))
    return pd.DatetimeIndex(out)


def _research(split):
    from load import load
    d = load("all", adjusted=True)
    d = d.rename(columns={"partial_sess": "partial"})
    d["pd_ok"] = True
    d["pw_ok"] = True
    if split == "dev":
        keep = d.sess <= DEV_END
        d["split"] = np.where(keep, "dev", "oot")
        d = d[keep]
    elif split == "oot":
        # keep the last 30 dev sessions as warm-up for ATR / weekly levels
        first_oot = d.loc[d.split == "oot", "sess"].min()
        warm = sorted(d.loc[d.sess < first_oot, "sess"].unique())[-30]
        d = d[d.sess >= warm].copy()
        d["split"] = np.where(d.sess >= first_oot, "oot", "warmup")
    return d[["ts", "sess", "open", "high", "low", "close", "volume", "delta", "partial",
              "pd_ok", "pw_ok", "split", "roll_adj", "contract"]].reset_index(drop=True)


def _lockbox():
    from build_15y import read_raw, add_session
    d = read_raw("nq")
    d = d.drop_duplicates("ts").sort_values("ts")
    d = add_session(d)
    wall = d.ts.dt.tz_localize(None)
    d = d[(wall.dt.hour != 17)]                                     # 17:00-17:59 maintenance prints
    d = d[(d.sess >= LOCKBOX[0] - pd.Timedelta(days=45)) & (d.sess <= LOCKBOX[1])].copy()
    nbars = d.groupby("sess").ts.transform("size")
    d["partial"] = nbars < 1200
    d["delta"] = np.nan
    wk = d.sess - pd.to_timedelta(d.sess.dt.weekday, unit="D")      # Monday of the session's week
    exp = _expiry_weeks(range(2011, 2021))
    d["pd_ok"] = ~wk.isin(exp)
    d["pw_ok"] = ~(wk.isin(exp) | wk.isin(exp + pd.Timedelta(days=7)))
    d["split"] = np.where(d.sess >= LOCKBOX[0], "lockbox", "warmup")
    d["roll_adj"] = 0.0
    d["contract"] = ""
    return d[["ts", "sess", "open", "high", "low", "close", "volume", "delta", "partial",
              "pd_ok", "pw_ok", "split", "roll_adj", "contract"]].reset_index(drop=True)


def load_split(split="dev"):
    if split in ("dev", "oot"):
        return _research(split)
    if split == "lockbox":
        return _lockbox()
    raise ValueError(split)
