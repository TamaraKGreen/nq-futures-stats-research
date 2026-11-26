"""
Kuant Gapline ORB as a DAY-TRADING strategy (no overnight holding), MNQ.

Same rules and costs as gap_orb_ema.py (matched to the PowerLanguage code) with the overnight leg removed,
plus switches for the pre-declared exploration:
  FLAT        minute (ET) whose bar OPEN is the market exit: 1000 = 16:40 (as coded), 960 = 16:00
  RR          target in multiples of the stop distance; None = no target (hold to FLAT)
  LAST_SIG    last 5-min bar CLOSE time allowed to signal (ET minutes): 995 = 16:35 (as coded)
  GAP         ("pts", 0.5) as coded | ("atr", 0.15) gap >= 0.15 x ATR14 | None = every day
  SIDES       "long" (as coded) | "both" = also the mirror short on gap-down days
              (close < range low and < EMA200; stop distance = range high - close)
  MAX_TRADES  per day (None = unlimited, as coded)
  RANDOM_P    control: enter on a random completed 5-min bar (probability per bar) in the gap direction,
              same stop-distance definition and exits
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from research.strategies.gap_orb_ema import prep as _prep, stats, by_year, PT, TICK, COMM

OUT = ROOT / "research" / "output"


def prep():
    d, D = _prep()
    rth = d[(d["mod"] >= 570) & (d["mod"] < 960)]
    g = rth.groupby("day")
    D["rng"] = (g.high.max() - g.low.min()).reindex(D.index)
    D["atr"] = D.rng.where(D.n >= 380).rolling(14, min_periods=10).mean().shift(1)
    return d, D


def run(d, D, FLAT=1000, RR=1.5, LAST_SIG=995, GAP=("pts", 0.5), SIDES="long", MAX_TRADES=None,
        RANDOM_P=None, SEED=0):
    rng = np.random.default_rng(SEED)
    O, H, L, C = (d[c].to_numpy() for c in ("open", "high", "low", "close"))
    M, E5, END5 = d["mod"].to_numpy(), d["ema5"].to_numpy(), d["end5"].to_numpy()
    DAY = d["day"].to_numpy()
    if GAP is None:
        up_ok = D.gap.notna(); dn_ok = D.gap.notna()
    elif GAP[0] == "pts":
        up_ok = D.gap >= GAP[1]; dn_ok = D.gap <= -GAP[1]
    else:
        up_ok = D.gap >= GAP[1] * D.atr; dn_ok = D.gap <= -GAP[1] * D.atr
    good = D.on >= 12
    UP, DN = (up_ok & good).to_dict(), (dn_ok & good & (SIDES == "both")).to_dict()
    GAPSIGN = np.sign(D.gap).to_dict()
    orh, orl = D.orh.to_dict(), D.orl.to_dict()
    n = len(O)
    trades = []
    pos = None
    cur_day, ok_up, ok_dn, R_H, R_L, n_day, gsign = None, False, False, np.nan, np.nan, 0, 0
    sig_at, sig_sd, sig_dir = -1, np.nan, 0

    def close(i, px, how):
        nonlocal pos
        pts = pos["dir"] * (px - pos["entry"])
        trades.append({"day": pd.Timestamp(DAY[pos["i_entry"]]), "dir": "long" if pos["dir"] > 0 else "short",
                       "qty": pos["qty"], "entry_min": int(M[pos["i_entry"]]), "how": how, "pts": pts,
                       "R": pts / pos["risk"], "pnl": pos["qty"] * (pts * PT - COMM), "exit_day": pd.Timestamp(DAY[i]),
                       "risk_pts": pos["risk"]})
        pos = None

    for i in range(n):
        m = M[i]
        if DAY[i] != cur_day:
            cur_day = DAY[i]; key = pd.Timestamp(cur_day)
            ok_up, ok_dn = bool(UP.get(key, False)), bool(DN.get(key, False))
            R_H, R_L = orh.get(key, np.nan), orl.get(key, np.nan)
            n_day = 0; gsign = GAPSIGN.get(key, 0)
        if pos is None and sig_at == i:
            sd = sig_sd
            entry = O[i] + sig_dir * TICK
            q = min(int(500 // (sd * PT)), int(1000 // (sd * PT)))
            qty = max(1, min(40, q))
            pos = dict(dir=sig_dir, entry=entry, stop=entry - sig_dir * sd,
                       target=(entry + sig_dir * RR * sd) if RR else None, qty=qty, i_entry=i, risk=sd)
            n_day += 1
        sig_at = -1
        if pos is not None:
            dr = pos["dir"]
            if (L[i] <= pos["stop"]) if dr > 0 else (H[i] >= pos["stop"]):
                close(i, pos["stop"] - dr * TICK, "stop")
            elif pos["target"] is not None and ((H[i] >= pos["target"]) if dr > 0 else (L[i] <= pos["target"])):
                close(i, pos["target"] - dr * TICK, "target")
        if pos is not None and m == FLAT:
            close(i, O[i] - pos["dir"] * TICK, "flat")
            continue
        if m >= FLAT and m < 1080:
            continue
        if pos is None and END5[i] and 590 <= m + 1 <= LAST_SIG and np.isfinite(R_H) and (ok_up or ok_dn):
            if MAX_TRADES is not None and n_day >= MAX_TRADES:
                continue
            if i + 1 >= n or M[i + 1] != m + 1:
                continue
            if RANDOM_P is not None:
                if rng.random() < RANDOM_P:
                    dr = 1 if (ok_up and gsign > 0) or (ok_up and not ok_dn) else -1
                    sd = C[i] - R_L if dr > 0 else R_H - C[i]
                    if sd > 0:
                        sig_at, sig_sd, sig_dir = i + 1, sd, dr
            else:
                if ok_up and C[i] > R_H and C[i] > E5[i] and C[i] - R_L > 0:
                    sig_at, sig_sd, sig_dir = i + 1, C[i] - R_L, 1
                elif ok_dn and C[i] < R_L and C[i] < E5[i] and R_H - C[i] > 0:
                    sig_at, sig_sd, sig_dir = i + 1, R_H - C[i], -1
    return pd.DataFrame(trades)
