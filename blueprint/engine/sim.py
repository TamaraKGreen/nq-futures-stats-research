"""1-minute trade simulator with the plan's conservative fill rules (PLAN.md §1).

- Market entry at the OPEN of the entry bar, +1 tick adverse slippage.
- Limit entry fills only if price trades THROUGH the limit by 1 tick (or opens beyond it: fill at the open);
  the order is cancelled if the target is touched first or the window ends. No target fill on the fill bar.
- Stop: exit at stop -1 tick, or at the open -1 tick if the bar gaps beyond it.
- Target (limit): fills only if price trades through by 1 tick; no slippage.
- Stop and target inside one bar -> stop.
- Time stop: exit at the close of the last allowed bar, -1 tick. Hold <= max_hold minutes, and never past 15:59.
- Skip if risk <= 0 or risk > MAX_STOP; target distance capped at MAX_TARGET.
- Net P&L subtracts COST_PTS (the $6.50 round trip on 5 MNQ).
"""
import numpy as np
import pandas as pd

from .data import TICK, SLIP, COST_PTS

MAX_STOP = 50.0
MAX_TARGET = 50.0
MAX_HOLD = 30
RTH_LAST = 15 * 60 + 59


class Bars:
    """Numpy views of a 1m frame, plus the index of each bar's session-last RTH bar."""

    def __init__(self, df, hm):
        self.o, self.h, self.l, self.c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.ts = df.ts.to_numpy()
        self.hm = np.asarray(hm)
        idx = np.arange(len(df))
        last = pd.Series(np.where(self.hm <= RTH_LAST, idx, -1)).groupby(df.sess.to_numpy()).transform("max")
        self.last_rth = last.to_numpy()

    def index_at(self, t):
        """Index of the 1m bar opening exactly at t (or the first after it)."""
        return np.searchsorted(self.ts, np.asarray(t, dtype=self.ts.dtype))


def trade(B, e, side, stop, target, max_hold=MAX_HOLD, limit=None, window=0):
    """Simulate one trade. e = signal/entry bar index. Returns dict or None (skipped / not filled)."""
    max_hold = min(max_hold, MAX_HOLD)
    last = B.last_rth[e]
    if last < e:
        return None
    o, h, l, c = B.o, B.h, B.l, B.c
    # ---- entry -------------------------------------------------------------------------------------------------
    if limit is None:
        fill_i = e
        fill = o[e] + side * SLIP
        first_check = e                       # entry at the open: the whole entry bar counts
    else:
        fill_i = None
        for k in range(e, min(e + window, last + 1)):
            if side == 1:
                if o[k] <= limit - TICK:               # opens through the limit: filled at the open
                    fill_i, fill = k, o[k]
                elif h[k] >= target:                   # target touched without a gap fill -> cancel (missed)
                    return None
                elif l[k] <= limit - TICK:
                    fill_i, fill = k, limit
            else:
                if o[k] >= limit + TICK:
                    fill_i, fill = k, o[k]
                elif l[k] <= target:
                    return None
                elif h[k] >= limit + TICK:
                    fill_i, fill = k, limit
            if fill_i is not None:
                break
        if fill_i is None:
            return None
        first_check = fill_i
    risk = side * (fill - stop)
    if risk <= 0 or risk > MAX_STOP:
        return None
    reward = side * (target - fill)
    if reward <= 0:
        return None
    if reward > MAX_TARGET:
        target = fill + side * MAX_TARGET
    end = min(fill_i + max_hold - 1, last)
    mae = mfe = 0.0
    exit_px, why, k = None, None, first_check
    for k in range(first_check, end + 1):
        lim_bar = (limit is not None and k == fill_i)
        if side == 1:
            mae = min(mae, l[k] - fill); mfe = max(mfe, h[k] - fill)
            if k > fill_i and o[k] <= stop:
                exit_px, why = o[k] - SLIP, "stop"
            elif l[k] <= stop:
                exit_px, why = stop - SLIP, "stop"
            elif not lim_bar and h[k] >= target + TICK:
                exit_px, why = target, "target"
        else:
            mae = min(mae, fill - h[k]); mfe = max(mfe, fill - l[k])
            if k > fill_i and o[k] >= stop:
                exit_px, why = o[k] + SLIP, "stop"
            elif h[k] >= stop:
                exit_px, why = stop + SLIP, "stop"
            elif not lim_bar and l[k] <= target - TICK:
                exit_px, why = target, "target"
        if exit_px is not None:
            break
    if exit_px is None:
        k = end
        exit_px, why = c[end] - side * SLIP, ("eod" if end == last and fill_i + max_hold - 1 > last else "time")
    gross = side * (exit_px - fill)
    net = gross - COST_PTS
    return dict(i_fill=fill_i, i_exit=k, side=side, fill=fill, stop=stop, target=target, exit=exit_px,
                why=why, risk=risk, gross=gross, net=net, R=net / risk, mins=k - fill_i + 1, mae=mae, mfe=mfe)


def run(B, cands, **kw):
    """cands: DataFrame with columns e, side, stop, target (+ optional limit, window, max_hold). Extra columns kept."""
    rows = []
    for r in cands.itertuples(index=False):
        d = r._asdict()
        res = trade(B, int(d["e"]), int(d["side"]), float(d["stop"]), float(d["target"]),
                    max_hold=int(d.get("max_hold", kw.get("max_hold", MAX_HOLD))),
                    limit=(None if pd.isna(d.get("limit", np.nan)) else float(d["limit"])),
                    window=int(d.get("window", kw.get("window", 0)) or 0))
        if res is not None:
            d.update(res)
            rows.append(d)
    return pd.DataFrame(rows)
