"""
Kuant Gapline ORB (gap-up ORB + EMA200, long only), MNQ - matched to KuantGaplineORB PowerLanguage
signal (research/prior_work/kuant_gapline/).

CT -> ET: 08:30 CT = 09:30 ET.
  1 gap        09:30 open - prior RTH close (15:59 bar close) >= 0.5 pts, else no NEW entries today
  2 range      high/low of 09:30-09:44 (first three 5-min bars)
  3 signal     5-min bar closing 09:50-16:35 ET (LastSigTime 1535 CT) with close > range high AND
               close > EMA200(5-min closes)
  4 entry      next 1-min bar's open; Sd = signal close - range low;
               qty = floor(500 / (Sd x $2)), also <= floor(1000 / (Sd x $2)), capped at 40, at least 1
  5 stop       fill - Sd           6 target  fill + 1.5 x Sd
               no time exit; can trade again the same day after an exit
  7 16:40 ET   flatten (close of the 16:39 bar); no new entries until 18:00
  8 18:00 ET   re-enter the flattened position at the 18:00 open with the same stop and target,
               REENTRY_QTY contracts (1 by default; 0 = same size). If the open is already beyond the
               stop or target the trade closes at that open.
Assumptions: one position at a time; EMA200 runs over all 5-min bars (24h) on back-adjusted
closes, and the whole simulation uses back-adjusted prices, so stops/targets carried over a roll
stay consistent; exits on 1-min bars; a bar touching stop and target = stop; stop fills 1 tick
worse; costs as in the code header: $0.50 per contract per side and 1 tick slippage on EVERY fill
(entry, stop, target, flatten, re-entry).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from research.scripts.load import load

PT, TICK, COMM = 2.0, 0.25, 1.00          # $0.50/side
OUT = ROOT / "research" / "output"


def prep():
    d = load("all")
    for c in ("open", "high", "low", "close"):
        d[c] = d[c] + d["roll_adj"]                       # back-adjusted
    naive = d.ts.dt.tz_localize(None)
    d["mod"] = naive.dt.hour * 60 + naive.dt.minute
    d["day"] = naive.dt.normalize()
    k5 = naive.dt.floor("5min")
    c5 = d.groupby(k5).close.last()
    ema = c5.ewm(span=200, adjust=False).mean()
    d["ema5"] = k5.map(ema).to_numpy()                   # EMA value at the close of this bar's 5-min bucket
    d["end5"] = (d["mod"] % 5 == 4)                       # last minute of a 5-min bar
    rth = d[(d["mod"] >= 570) & (d["mod"] < 960)]
    g = rth.groupby("day")
    D = pd.DataFrame({"o": g.open.first(), "c": g.close.last(), "n": g.size()})
    orb = d[(d["mod"] >= 570) & (d["mod"] < 585)].groupby("day").agg(orh=("high", "max"), orl=("low", "min"), on=("close", "size"))
    D = D.join(orb)
    D["prev_c"] = D.c.shift(1).where(D.n.shift(1) >= 380)
    D["gap"] = D.o - D.prev_c
    D["eligible"] = (D.gap >= 0.5) & (D.on >= 12)
    return d, D


def run(d, D, REENTRY_QTY=1, OVERNIGHT=True, GAP_MIN=0.5, RR=1.5, LAST_SIG=995,
        USE_ORB=True, USE_EMA=True, RANDOM_P=None, SEED=0):
    """Controls: GAP_MIN=None trades every day; USE_ORB / USE_EMA switch each entry condition off;
    RANDOM_P = enter on a random completed 5-min bar in the window with this probability (instead of the
    signal), same stop distance definition (close - range low, must be > 0), same exits."""
    rng = np.random.default_rng(SEED)
    O, H, L, C = (d[c].to_numpy() for c in ("open", "high", "low", "close"))
    M, E5, END5 = d["mod"].to_numpy(), d["ema5"].to_numpy(), d["end5"].to_numpy()
    DAY = d["day"].to_numpy()
    elig = (((D.gap >= GAP_MIN) if GAP_MIN is not None else D.gap.notna()) & (D.on >= 12)).to_dict()
    orh, orl = D.orh.to_dict(), D.orl.to_dict()
    n = len(O)
    trades = []
    pos = None                    # dict(entry, stop, target, qty, kind, i_entry, check_entry_bar)
    pending = None                # flattened position waiting for 18:00
    cur_day, ok_day, R_H, R_L = None, False, np.nan, np.nan
    sig_at = -1                   # index of the bar whose open we enter at
    sig_sd = np.nan

    def close(i, px, how):
        nonlocal pos
        pts = px - pos["entry"]
        trades.append({"day": pd.Timestamp(DAY[pos["i_entry"]]), "kind": pos["kind"], "qty": pos["qty"],
                       "entry": pos["entry"], "exit": px, "how": how, "pts": pts,
                       "R": pts / pos["risk"], "pnl": pos["qty"] * (pts * PT - COMM),
                       "exit_day": pd.Timestamp(DAY[i])})
        pos = None

    for i in range(n):
        m = M[i]
        if DAY[i] != cur_day:
            cur_day = DAY[i]; key = pd.Timestamp(cur_day)
            ok_day = bool(elig.get(key, False)); R_H, R_L = orh.get(key, np.nan), orl.get(key, np.nan)
        # 18:00 re-entry of a flattened position
        if pending is not None and m == 1080:
            p = pending; pending = None
            qty = p["qty"] if REENTRY_QTY == 0 else REENTRY_QTY
            entry = O[i] + TICK
            pos = dict(entry=entry, stop=p["stop"], target=p["target"], qty=qty, kind="overnight re-entry",
                       i_entry=i, risk=p["risk"])
            if O[i] <= pos["stop"]:
                close(i, O[i] - TICK, "gap through stop"); continue
            if O[i] >= pos["target"]:
                close(i, O[i] - TICK, "gap through target"); continue
        # entry at this bar's open (signal on the previous bar)
        if pos is None and sig_at == i:
            entry = O[i] + TICK
            sd = sig_sd
            q = min(int(500 // (sd * PT)), int(1000 // (sd * PT)))
            qty = max(1, min(40, q))
            pos = dict(entry=entry, stop=entry - sd, target=entry + RR * sd, qty=qty, kind="RTH", i_entry=i, risk=sd)
        sig_at = -1
        # exits
        if pos is not None:
            if L[i] <= pos["stop"]:
                close(i, pos["stop"] - TICK, "stop")
            elif H[i] >= pos["target"]:
                close(i, pos["target"] - TICK, "target")
        # 16:40 flatten
        if pos is not None and m == 1000:          # market exit at the open of the 16:40 ET bar
            if OVERNIGHT:
                pending = dict(stop=pos["stop"], target=pos["target"], qty=pos["qty"], risk=pos["risk"])
            close(i, O[i] - TICK, "flat 16:40")
            continue
        # signal on a completed 5-min bar closing 09:50-16:30
        if pos is None and pending is None and ok_day and END5[i] and 590 <= m + 1 <= LAST_SIG and np.isfinite(R_H):
            if RANDOM_P is not None:
                go = rng.random() < RANDOM_P
            else:
                go = (C[i] > R_H or not USE_ORB) and (C[i] > E5[i] or not USE_EMA)
            if go and C[i] - R_L > 0 and i + 1 < n and M[i + 1] == m + 1:
                sig_at = i + 1; sig_sd = C[i] - R_L
    return pd.DataFrame(trades)


def stats(t, days):
    if not len(t):
        return pd.Series({"trades": 0})
    daily = t.groupby("exit_day").pnl.sum().reindex(days, fill_value=0.0)
    eq = daily.cumsum()
    return pd.Series({"trades": len(t), "win %": 100 * (t.pnl > 0).mean(), "avg R": t.R.mean(),
                      "net $": t.pnl.sum(), "$/trade": t.pnl.mean(),
                      "PF": t.pnl[t.pnl > 0].sum() / max(-t.pnl[t.pnl < 0].sum(), 1e-9),
                      "sharpe": daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan,
                      "max DD $": (eq.cummax() - eq).max()})


def by_year(t, days):
    days = pd.DatetimeIndex(days)
    out = {y: stats(t[t.exit_day.dt.year == y], days[days.year == y]) for y in sorted(set(days.year))}
    out["ALL"] = stats(t, days)
    return pd.DataFrame(out).T


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    d, D = prep()
    days = D.index[D.n >= 380]
    print(f"days {len(days)}; gap-up >= 0.5 pts days: {int(D.eligible.sum())} ({100 * D.eligible.mean():.0f}%)\n")
    for label, kw in (("AS DESCRIBED (overnight re-entry 1 MNQ)", {}),
                      ("re-entry with the SAME size (ReEntryQty = 0)", dict(REENTRY_QTY=0)),
                      ("NO overnight hold (flat at 16:40, nothing at 18:00)", dict(OVERNIGHT=False))):
        t = run(d, D, **kw)
        t.to_parquet(OUT / f"gap_orb_ema_{'_'.join(f'{k}{v}' for k, v in kw.items()) or 'base'}.parquet", index=False)
        print("=" * 110, f"\n{label}\n")
        print(by_year(t, days).astype(float).round(2).to_string())
        print("\nby trade type:")
        print(t.groupby("kind").apply(lambda g: stats(g, days), include_groups=False).astype(float).round(2).to_string())
        print("\nhow trades end:", t.how.value_counts().to_dict(), "\n")
