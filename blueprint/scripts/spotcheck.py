"""Chart a few sessions with every level and the 5m / 15m iFVGs, in RAW contract prices, to compare by eye with
TradingView / NinjaTrader (use the specific contract, e.g. NQZ2024, not a back-adjusted continuous chart).

    python3 blueprint/scripts/spotcheck.py 2024-10-15 2025-02-20     -> blueprint/spotcheck/<date>.png + .csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from engine.data import load_split          # noqa: E402
from engine.levels import build_levels, LEVELS   # noqa: E402
from engine import fvg                      # noqa: E402

OUT = HERE / "spotcheck"
COLORS = {"pd": "#d62728", "pw": "#9467bd", "asia": "#8c564b", "lon": "#e377c2", "on": "#7f7f7f",
          "vwap": "#1f77b4", "ib": "#2ca02c", "ph": "#bcbd22", "p4": "#17becf"}


def main(dates):
    OUT.mkdir(exist_ok=True)
    d, L = build_levels(load_split("dev"))
    tfs = {tf: fvg.detect(fvg.resample(d, tf), tf)[1] for tf in (5, 15)}
    for ds in dates:
        day = pd.Timestamp(ds)
        m = (d.sess == day) & L.rth
        x, lv = d[m].copy(), L[m].copy()
        adj = x.roll_adj.iloc[0]
        for k in ("open", "high", "low", "close"):
            x[k] -= adj
        lv[LEVELS] -= adj
        fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
        t = np.arange(len(x))
        up = x.close >= x.open
        ax.vlines(t, x.low, x.high, color="#444", lw=0.6)
        ax.bar(t, (x.close - x.open).abs().clip(lower=0.25), bottom=np.minimum(x.open, x.close),
               color=np.where(up, "#26a69a", "#ef5350"), width=0.8)
        for name in LEVELS:
            fam = {"pdh": "pd", "pdl": "pd", "pwh": "pw", "pwl": "pw"}.get(name, name.split("_")[0])
            col = COLORS.get(fam, "#000")
            ls = "-" if name.startswith("vwap") else "--"
            ax.plot(t, lv[name].to_numpy(), color=col, lw=1.0 if ls == "-" else 0.8, ls=ls, alpha=0.85)
            last = lv[name].dropna()
            if len(last):
                ax.text(len(t) + 2, last.iloc[-1], name, color=col, fontsize=7, va="center", clip_on=True)
        ts0 = x.ts.iloc[0]
        rows = []
        for tf, iv in tfs.items():
            ev = iv[(iv.t_known > ts0) & (iv.t_known <= x.ts.iloc[-1])]
            for r in ev.itertuples():
                k = int((r.t_known - ts0) / pd.Timedelta(minutes=1))
                col = "#26a69a" if r.dir == 1 else "#ef5350"
                ax.add_patch(plt.Rectangle((k - tf, r.lo - adj), tf, r.hi - r.lo, color=col,
                                           alpha=0.25 if tf == 5 else 0.45, lw=0))
                ax.annotate(f"{tf}m", (k, (r.hi if r.dir == -1 else r.lo) - adj), fontsize=6, color=col,
                            ha="center", va="bottom" if r.dir == -1 else "top")
                rows.append(dict(tf=tf, dir="bull iFVG" if r.dir == 1 else "bear iFVG",
                                 known_at=r.t_known.strftime("%H:%M"), fvg_formed=r.t_fvg.strftime("%H:%M"),
                                 zone_lo=r.lo - adj, zone_hi=r.hi - adj))
        ticks = np.arange(0, len(t), 30)
        ax.set_xticks(ticks, x.ts.iloc[ticks].dt.strftime("%H:%M"))
        lo, hi = x.low.min(), x.high.max()
        pad = (hi - lo) * 0.25
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-2, len(t) + 25)
        ax.set_title(f"{ds}  {x.contract.iloc[0]} (raw prices)   levels: dashed, VWAPs: solid   "
                     f"boxes = 5m (light) / 15m (dark) iFVG zones, drawn at the bar they were confirmed")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(OUT / f"{ds}.png")
        plt.close(fig)
        tab = lv[LEVELS].iloc[[0, 60, 120, -1]].round(2).T
        tab.columns = [f"@{s}" for s in x.ts.iloc[[0, 60, 120, -1]].dt.strftime("%H:%M")]
        tab.to_csv(OUT / f"{ds}_levels.csv")
        pd.DataFrame(rows).sort_values(["known_at", "tf"]).to_csv(OUT / f"{ds}_ifvgs.csv", index=False)
        print(ds, "written")


if __name__ == "__main__":
    main(sys.argv[1:] or ["2024-10-15", "2025-02-20", "2023-08-10"])
