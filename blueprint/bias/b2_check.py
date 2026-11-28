"""B2 robustness: how much is IB width alone, and does skill survive an ATR-scaled trend label?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
from bias.models import logit, folds, summarise
days = pd.read_parquet(HERE / "cache/days_dev.parquet").sort_values("sess").reset_index(drop=True)
days["ib_close_fold"] = (days.ib_close_pos - .5).abs(); days["abs_gap"] = days.gap_atr.abs()
days["out_pd"] = np.maximum(days.open_pos_pd - 1, -days.open_pos_pd).clip(lower=0)
days["out_va"] = np.maximum(days.va_pos - 1, -days.va_pos).clip(lower=0)
days["abs_prior_body"] = days.prior_body.abs(); days["abs_vwap_1030"] = days.vwap_rth_dist_1030.abs()
F = ["ib_w_atr", "orb15_w_atr", "ib_close_fold", "on_range_atr", "abs_gap", "out_pd", "out_va", "atr_ratio",
     "prior_range_atr", "abs_prior_body", "abs_vwap_1030", "dow"]
ext_atr = days.ib_ext * days.ib_w_atr          # extension beyond IB in ATR units
med = ext_atr.median()
lines = []
for lab, y in (("B2 (IB-width units)", days.y_b2),
               (f"B2-ATR: extension beyond IB > median ({med:.2f} ATR)", (ext_atr > med).astype(float))):
    df = days.assign(y=y)[y.notna()].reset_index(drop=True)
    P = {"base": [], "ib_width_only": [], "logit_all": [], "logit_no_ibw": []}
    idx = []
    for yr, tr_d, te_d in folds(days):
        cut = days.sess[tr_d].max(); tr = (df.sess <= cut).values; te = (df.sess.dt.year == yr).values
        ytr = df.y[tr].values; idx.append(df.index[te])
        P["base"].append(np.full(te.sum(), ytr.mean()))
        for name, f in (("ib_width_only", ["ib_w_atr"]), ("logit_all", F), ("logit_no_ibw", [c for c in F if c not in ("ib_w_atr", "orb15_w_atr")])):
            P[name].append(logit().fit(df.loc[tr, f], ytr).predict_proba(df.loc[te, f])[:, 1])
    rows = df.loc[np.concatenate(idx)]
    res = summarise(rows, {k: np.concatenate(v) for k, v in P.items()})
    lines += ["", lab, res.round(4).to_string()]
m = logit().fit(days.loc[days.y_b2.notna(), F], days.y_b2.dropna())
coef = pd.Series(m[-1].coef_[0], index=F).sort_values()
lines += ["", "B2 logit coefficients, all Dev (standardised features; + = more likely trend day):", coef.round(3).to_string()]
txt = "\n".join(lines); print(txt)
with open(HERE / "results/phase1_bias.txt", "a") as fh: fh.write("\n\n" + "=" * 110 + "\nB2 ROBUSTNESS" + txt)
