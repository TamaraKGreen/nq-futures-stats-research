import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.data import COST_PTS, TICK, SLIP          # noqa: E402
from engine import fvg, sim                            # noqa: E402
from engine.levels import build_levels, LEVELS         # noqa: E402


def frame(rows, start="2024-03-05 09:30", sess="2024-03-05"):
    """rows: list of (o,h,l,c). 1-minute bars from `start`."""
    ts = pd.date_range(start, periods=len(rows), freq="1min", tz="America/New_York")
    d = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    d["ts"] = ts
    d["sess"] = pd.Timestamp(sess)
    d["volume"] = 100
    return d


def bars(rows, **kw):
    d = frame(rows, **kw)
    hm = d.ts.dt.hour * 60 + d.ts.dt.minute
    return d, sim.Bars(d, hm)


# ---------------------------------------------------------------- FVG / iFVG
def test_bullish_fvg_then_close_inversion():
    rows = [(100, 101, 99, 100.5), (101, 105, 100.5, 104.5), (104.5, 106, 103, 105),   # gap (101, 103)
            (105, 105.5, 102, 102.5),                                                    # into gap, no close below
            (102.5, 102.6, 100, 100.25)]                                                 # close < 101 -> bearish iFVG
    d = frame(rows)
    f, iv = fvg.detect(fvg.resample(d, 1), 1)
    assert len(iv) == 1
    r = iv.iloc[0]
    assert r.dir == -1 and r.lo == 101 and r.hi == 103
    assert r.t_known == d.ts.iloc[4] + pd.Timedelta(minutes=1)     # known at the close of the inversion bar
    assert f.iloc[0].end == "inv"


def test_wick_mode_inverts_earlier():
    rows = [(100, 101, 99, 100.5), (101, 105, 100.5, 104.5), (104.5, 106, 103, 105),
            (105, 105.5, 100.5, 102.0),     # wick below 101, close inside
            (102, 102.5, 100, 100.25)]
    d = frame(rows)
    _, iv_c = fvg.detect(fvg.resample(d, 1), 1, mode="close")
    _, iv_w = fvg.detect(fvg.resample(d, 1), 1, mode="wick")
    assert iv_w.iloc[0].t_known < iv_c.iloc[0].t_known


def test_bearish_fvg_inversion_is_long():
    rows = [(110, 111, 109, 109.5), (109.5, 109.6, 105, 105.5), (105.5, 107, 104, 106),  # gap (107, 109)
            (106, 110, 106, 109.75)]                                                       # close > 109
    _, iv = fvg.detect(fvg.resample(frame(rows), 1), 1)
    assert len(iv) == 1 and iv.iloc[0].dir == 1


def test_fvg_expiry():
    rows = [(100, 101, 99, 100.5), (101, 105, 100.5, 104.5), (104.5, 106, 103, 105)] + \
           [(105, 105.5, 104.5, 105)] * 5 + [(105, 105, 99, 99.5)]
    _, iv = fvg.detect(fvg.resample(frame(rows), 1), 1, expiry=3)
    assert len(iv) == 0


def test_resample_5m_clock_aligned_and_known_at_close():
    rows = [(100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(12)]
    d = frame(rows, start="2024-03-05 09:31")
    b = fvg.resample(d, 5)
    assert list(b.ts.dt.strftime("%H:%M")) == ["09:30", "09:35", "09:40"]
    assert b.t_close.iloc[1] == pd.Timestamp("2024-03-05 09:40", tz="America/New_York")
    assert b.high.iloc[1] == max(101 + i for i in range(4, 9))


# ---------------------------------------------------------------- simulator
def test_market_long_target_needs_trade_through():
    rows = [(100, 100.5, 99.5, 100), (100, 110.0, 99.75, 109)] + [(109, 109.5, 108, 109)] * 3
    _, B = bars(rows)
    r = sim.trade(B, 0, 1, stop=95, target=110, max_hold=5)
    assert r["why"] != "target"                      # touched 110 but not 110.25
    rows[1] = (100, 110.25, 99.75, 109)
    _, B = bars(rows)
    r = sim.trade(B, 0, 1, stop=95, target=110, max_hold=5)
    assert r["why"] == "target" and r["fill"] == 100 + SLIP
    assert r["net"] == pytest.approx(110 - 100.25 - COST_PTS)


def test_stop_first_when_both_in_bar():
    rows = [(100, 100.5, 99.5, 100), (100, 120, 90, 100)]
    _, B = bars(rows)
    r = sim.trade(B, 0, 1, stop=95, target=110, max_hold=5)
    assert r["why"] == "stop" and r["exit"] == 95 - SLIP


def test_gap_through_stop_exits_at_open():
    rows = [(100, 100.5, 99.5, 100), (93, 94, 92, 93)]
    _, B = bars(rows)
    r = sim.trade(B, 0, 1, stop=95, target=110, max_hold=5)
    assert r["exit"] == 93 - SLIP


def test_short_symmetry():
    rows = [(100, 100.5, 99.5, 100), (100, 100.2, 89.75, 90)]
    _, B = bars(rows)
    r = sim.trade(B, 0, -1, stop=105, target=90, max_hold=5)
    assert r["why"] == "target" and r["gross"] == pytest.approx(100 - SLIP - 90)


def test_time_stop_and_hold_cap():
    rows = [(100, 100.5, 99.5, 100)] * 40
    _, B = bars(rows)
    r = sim.trade(B, 0, 1, stop=90, target=120, max_hold=45)    # capped at 30
    assert r["why"] == "time" and r["mins"] == 30


def test_eod_exit_at_1559():
    rows = [(100, 100.5, 99.5, 100)] * 10
    _, B = bars(rows, start="2024-03-05 15:55")
    r = sim.trade(B, 0, 1, stop=90, target=120, max_hold=30)
    assert r["why"] == "eod" and B.hm[r["i_exit"]] == 15 * 60 + 59


def test_skip_wide_stop_and_cap_target():
    rows = [(100, 100.5, 99.5, 100)] * 5
    _, B = bars(rows)
    assert sim.trade(B, 0, 1, stop=40, target=200) is None
    r = sim.trade(B, 0, 1, stop=90, target=300)
    assert r["target"] == pytest.approx(100.25 + 50)


def test_limit_fill_requires_through_and_cancel_on_target():
    rows = [(105, 106, 104, 105), (105, 105, 100, 101), (101, 103, 99.5, 102)] + [(102, 103, 101, 102)] * 5
    _, B = bars(rows)
    r = sim.trade(B, 0, 1, stop=95, target=115, limit=100, window=5)
    assert r["i_fill"] == 2 and r["fill"] == 100            # bar 1 only touched 100, bar 2 traded through
    rows2 = [(105, 116, 104, 115)] + rows[1:]
    _, B = bars(rows2)
    assert sim.trade(B, 0, 1, stop=95, target=115, limit=100, window=5) is None


def test_limit_and_target_in_same_bar_is_cancelled():
    rows = [(105, 105, 104, 104.5), (104.5, 120, 99, 110)] + [(110, 110.5, 109.5, 110)] * 3
    _, B = bars(rows)
    assert sim.trade(B, 0, 1, stop=90, target=115, limit=100, window=5) is None


# ---------------------------------------------------------------- levels: no lookahead
def synth_days(n_days=6, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    day = pd.Timestamp("2024-03-04")
    price = 18000.0
    while len(parts) < n_days:
        if day.weekday() < 5:
            start = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d") + " 18:00"
            ts = pd.date_range(start, periods=23 * 60, freq="1min", tz="America/New_York")
            r = rng.normal(0, 2, len(ts)).cumsum() + price
            price = r[-1]
            o = np.r_[r[0], r[:-1]]
            d = pd.DataFrame({"ts": ts, "open": o, "close": r})
            d["high"] = np.maximum(d.open, d.close) + rng.uniform(0, 1, len(d))
            d["low"] = np.minimum(d.open, d.close) - rng.uniform(0, 1, len(d))
            d["volume"] = rng.integers(10, 500, len(d))
            d["sess"] = day
            parts.append(d)
        day += pd.Timedelta(days=1)
    d = pd.concat(parts, ignore_index=True)
    d["partial"], d["pd_ok"], d["pw_ok"] = False, True, True
    return d


def test_levels_have_no_lookahead():
    d = synth_days()
    _, L = build_levels(d)
    cut = d.index[(d.sess == d.sess.unique()[-2]) & (d.ts.dt.strftime("%H:%M") == "11:17")][0]
    d2 = d.copy()
    rng = np.random.default_rng(1)
    for k in ("open", "high", "low", "close"):
        d2.loc[cut:, k] = d2.loc[cut:, k] + rng.normal(0, 50, len(d2) - cut)
    d2["high"] = d2[["open", "high", "low", "close"]].max(axis=1)
    d2["low"] = d2[["open", "high", "low", "close"]].min(axis=1)
    d2.loc[cut:, "volume"] = rng.integers(1, 1000, len(d2) - cut)
    _, L2 = build_levels(d2)
    a, b = L.loc[:cut, LEVELS], L2.loc[:cut, LEVELS]
    pd.testing.assert_frame_equal(a, b)


def test_level_values():
    d = synth_days()
    _, L = build_levels(d)
    s = d.sess.unique()
    prev, cur = d[d.sess == s[-2]], d[d.sess == s[-1]]
    hm = cur.ts.dt.hour * 60 + cur.ts.dt.minute
    prth = prev[(prev.ts.dt.hour * 60 + prev.ts.dt.minute).between(570, 959)]
    i = cur.index[hm == 11 * 60 + 5][0]
    assert L.at[i, "pdh"] == prth.high.max() and L.at[i, "pdl"] == prth.low.min()
    ib = cur[hm.between(570, 629)]
    assert L.at[i, "ib_h"] == ib.high.max()
    assert np.isnan(L.at[cur.index[hm == 10 * 60 + 29][0], "ib_h"])           # IB not active before 10:30
    ph = cur[hm.between(600, 659)]
    assert L.at[i, "ph_h"] == ph.high.max()                                      # 11:05 -> 10:00 hour
    p4 = cur[hm.between(360, 599)]                                                # 06:00-09:59 block
    assert L.at[i, "p4_l"] == p4.low.min()
    rth_before = cur[hm.between(570, 11 * 60 + 4)]
    tp = (rth_before.high + rth_before.low + rth_before.close) / 3
    assert L.at[i, "vwap_rth"] == pytest.approx((tp * rth_before.volume).sum() / rth_before.volume.sum())
    day_before = cur[cur.index < i]
    tp = (day_before.high + day_before.low + day_before.close) / 3
    assert L.at[i, "vwap_day"] == pytest.approx((tp * day_before.volume).sum() / day_before.volume.sum())
    on = cur[(hm >= 1080) | (hm < 570)]
    assert L.at[i, "on_h"] == on.high.max()


def test_r_multiple_target_set_from_fill():
    rows = [(100, 100.5, 99.5, 100), (100, 116.0, 99.75, 115)]
    _, B = bars(rows)
    r = sim.trade(B, 0, 1, stop=90, r_mult=1.5, max_hold=5)
    assert r["target"] == pytest.approx(100.25 + 1.5 * 10.25)
    assert r["why"] == "target"


def test_run_one_position_at_a_time():
    rows = [(100, 100.5, 99.5, 100)] * 40
    _, B = bars(rows)
    c = pd.DataFrame(dict(e=[0, 5, 31], side=[1, 1, 1], stop=[90.0] * 3, r_mult=[2.0] * 3))
    out = sim.run(B, c, max_hold=30)
    assert list(out.e) == [0, 31]


def test_uncapped_comparison_mode_holds_past_30min():
    rows = [(100, 100.5, 99.5, 100)] * 50 + [(100, 200, 99.5, 190)]
    _, B = bars(rows)
    r = sim.trade(B, 0, 1, stop=90, target=None, max_hold=51, capped=False)
    assert r["mins"] == 51 and r["why"] == "time" and r["exit"] == 190 - SLIP
    assert sim.trade(B, 0, 1, stop=90, target=None, max_hold=60)["mins"] == 30     # capped: no target -> time at 30
