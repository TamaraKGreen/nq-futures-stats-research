"""Phase 1 - bias layer walk-forward on Dev.   python3 blueprint/bias/run_phase1.py
Writes blueprint/results/phase1_bias.txt and blueprint/cache/bias_oof_*.parquet (out-of-fold predictions)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from bias.models import (hmm_filtered, logit, knn, gbm, mlp, table_predict, summarise, calibration, folds,
                         EPS)   # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
CACHE, RES = HERE / "cache", HERE / "results"
RES.mkdir(exist_ok=True)

days = pd.read_parquet(CACHE / "days_dev.parquet").sort_values("sess").reset_index(drop=True)
races = pd.read_parquet(CACHE / "races_dev.parquet")
hours = pd.read_parquet(CACHE / "hours_dev.parquet")

# ---------------------------------------------------------------------------------------- feature sets (fixed)
F = {
    "B1_0930": ["geom", "d_up", "d_dn", "open_pos_pd", "gap_atr", "pos8", "on_range_atr", "va_pos",
                "vwap_day_dist", "vwap_wk_dist", "day_up_fill", "day_dn_fill", "wk_up_fill", "wk_dn_fill",
                "prior_body", "prior_range_atr", "atr_ratio", "dow"],
    "B1_1100": ["geom", "d_up", "d_dn", "ib_pos", "ib_w_atr", "vwap_rth_dist", "vwap_day_dist_now", "move_open",
                "range_so_far", "open_pos_pd", "va_pos", "gap_atr", "wk_up_fill", "wk_dn_fill", "atr_ratio"],
    "B2": ["ib_w_atr", "orb15_w_atr", "ib_close_fold", "on_range_atr", "abs_gap", "out_pd", "out_va",
           "atr_ratio", "prior_range_atr", "abs_prior_body", "abs_vwap_1030", "dow"],
    "B3": ["pos_fold", "ph_range_atr", "hour", "prev_close_fold", "range_so_far", "vwap_rth_or", "vwap_day_or",
           "move_open_or", "ib_w_atr", "dow"],
}
KNN_F = {
    "B1_0930": ["geom", "open_pos_pd", "pos8", "va_pos", "gap_atr", "vwap_day_dist", "wk_pos", "atr_ratio"],
    "B1_1100": ["geom", "ib_pos", "ib_w_atr", "vwap_rth_dist", "move_open", "range_so_far", "open_pos_pd", "va_pos"],
    "B2": ["ib_w_atr", "orb15_w_atr", "ib_close_fold", "on_range_atr", "abs_gap", "atr_ratio", "prior_range_atr",
           "abs_vwap_1030"],
    "B3": ["pos_fold", "ph_range_atr", "hour", "prev_close_fold", "range_so_far", "vwap_rth_or", "move_open_or",
           "ib_w_atr"],
}
DELTA = {"B1_0930": ["on_delta"], "B1_1100": ["d_open_delta"], "B2": ["ib_delta"], "B3": ["h_delta"]}

# ---------------------------------------------------------------------------------------- derived columns
days["ib_close_fold"] = (days.ib_close_pos - 0.5).abs()
days["abs_gap"] = days.gap_atr.abs()
days["out_pd"] = np.maximum(days.open_pos_pd - 1, -days.open_pos_pd).clip(lower=0)
days["out_va"] = np.maximum(days.va_pos - 1, -days.va_pos).clip(lower=0)
days["abs_prior_body"] = days.prior_body.abs()
days["abs_vwap_1030"] = days.vwap_rth_dist_1030.abs()
days["wk_pos"] = days.wk_up_fill - days.wk_dn_fill
days["opd_b"] = pd.cut(days.open_pos_pd, [-np.inf, 0, 0.5, 1, np.inf], labels=False)
days["pos8_b"] = pd.cut(days.pos8, [-np.inf, 1 / 3, 2 / 3, np.inf], labels=False)

DAYCOLS = [c for c in days.columns if c not in ("split",)]
r = races.merge(days[DAYCOLS].drop(columns=[c for c in ("atr",) if c in DAYCOLS]), on="sess", how="left",
                suffixes=("", "_day"))
r = r[r.y.notna()].reset_index(drop=True)
hrs = hours.merge(days[["sess", "on_delta"]], on="sess", how="left")

TASKS = {
    "B1_0930": r[r.t == "0930"].reset_index(drop=True),
    "B1_1100": r[r.t == "1100"].reset_index(drop=True),
    "B2": days[days.y_b2.notna()].assign(y=lambda x: x.y_b2).reset_index(drop=True),
    "B3_far": hrs.assign(y=hrs.y_far).reset_index(drop=True),
    "B3_near": hrs.assign(y=hrs.y_near).reset_index(drop=True),
}
for k in ("B1_0930", "B1_1100"):
    TASKS[k]["ib_pos_b"] = pd.cut(TASKS[k].get("ib_pos", pd.Series(0.5, index=TASKS[k].index)),
                                  [-np.inf, 1 / 3, 2 / 3, np.inf], labels=False)
    TASKS[k]["mo_b"] = np.sign(TASKS[k].get("move_open", pd.Series(0.0, index=TASKS[k].index)))


def feats(task):
    return F["B3" if task.startswith("B3") else task]


def knnf(task):
    return KNN_F["B3" if task.startswith("B3") else task]


def deltaf(task):
    return DELTA["B3" if task.startswith("B3") else task]


# ---------------------------------------------------------------------------------------- walk-forward
def run_task(task, df):
    preds = {}
    for yr, tr_d, te_d in folds(days):
        # HMM on the days table, fold-specific, then mapped onto the task rows
        hcols = {}
        for ns in (2, 3):
            pr = hmm_filtered(days, tr_d, ns)
            for j in range(ns):
                hcols[f"hmm{ns}_{j}"] = pd.Series(pr[:, j], index=days.sess)
        H = pd.DataFrame(hcols)
        x = df.join(H, on="sess")
        cut = days.sess[tr_d].max()
        tr = (x.sess <= cut).to_numpy()
        te = (x.sess.dt.year == yr).to_numpy()
        if te.sum() == 0:
            continue
        Xtr, Xte, ytr = x[tr], x[te], x.y[tr].to_numpy()
        P = {}
        is_b1 = task.startswith("B1")
        if is_b1:
            P["base"] = Xte.rw_p.clip(EPS, 1 - EPS).to_numpy()
            P["geo_fit"] = logit(1e3).fit(Xtr[["geom"]], ytr).predict_proba(Xte[["geom"]])[:, 1]
            keys = ["opd_b", "pos8_b"] if task == "B1_0930" else ["ib_pos_b", "mo_b"]
            P["table"] = table_predict(Xtr, Xte, keys, "y", Xtr.rw_p.to_numpy(), Xte.rw_p.to_numpy())
        else:
            P["base"] = np.full(te.sum(), ytr.mean())
            if task == "B2":
                tq = np.nanquantile(Xtr.ib_w_atr, [1 / 3, 2 / 3])
                oq = np.nanquantile(Xtr.on_range_atr, [1 / 3, 2 / 3])
                for X_ in (Xtr, Xte):
                    X_.loc[:, "ibw_b"] = np.digitize(X_.ib_w_atr, tq)
                    X_.loc[:, "onr_b"] = np.digitize(X_.on_range_atr, oq)
                keys = ["ibw_b", "onr_b"]
            else:
                rq = np.nanquantile(Xtr.ph_range_atr, [1 / 3, 2 / 3])
                for X_ in (Xtr, Xte):
                    X_.loc[:, "pos_b"] = pd.cut(X_.pos_fold, [-np.inf, 0.625, 0.75, 1.0, np.inf], labels=False)
                    X_.loc[:, "rng_b"] = np.digitize(X_.ph_range_atr, rq)
                keys = ["hour", "pos_b", "rng_b"]
            P["table"] = table_predict(Xtr, Xte, keys, "y", np.full(tr.sum(), ytr.mean()), P["base"])
        f = feats(task)
        P["logit"] = logit().fit(Xtr[f], ytr).predict_proba(Xte[f])[:, 1]
        f2 = f + ["hmm3_0", "hmm3_1", "hmm3_2"]
        P["logit+hmm3"] = logit().fit(Xtr[f2], ytr).predict_proba(Xte[f2])[:, 1]
        f2b = f + ["hmm2_0", "hmm2_1"]
        P["logit+hmm2"] = logit().fit(Xtr[f2b], ytr).predict_proba(Xte[f2b])[:, 1]
        fd = f + deltaf(task)
        P["logit+delta"] = logit().fit(Xtr[fd], ytr).predict_proba(Xte[fd])[:, 1]
        for k in (25, 50, 100):
            P[f"knn{k}"] = np.clip(knn(k).fit(Xtr[knnf(task)], ytr).predict_proba(Xte[knnf(task)])[:, 1], 0.02, 0.98)
        # HMM state table (regime gate on its own)
        st_tr = Xtr[["hmm3_0", "hmm3_1", "hmm3_2"]].to_numpy().argmax(1)
        st_te = Xte[["hmm3_0", "hmm3_1", "hmm3_2"]].to_numpy().argmax(1)
        prior_tr = Xtr.rw_p.to_numpy() if is_b1 else np.full(tr.sum(), ytr.mean())
        P["hmm3_table"] = table_predict(Xtr.assign(st=st_tr), Xte.assign(st=st_te), ["st"], "y", prior_tr,
                                        P["base"])
        if task.startswith("B3"):
            P["gbm"] = gbm().fit(Xtr[f], ytr).predict_proba(Xte[f])[:, 1]
            P["mlp"] = mlp().fit(Xtr[f], ytr).predict_proba(Xte[f])[:, 1]
        for k, v in P.items():
            preds.setdefault(k, []).append(pd.Series(v, index=x.index[te]))
    out = pd.DataFrame({k: pd.concat(v) for k, v in preds.items()})
    return df.loc[out.index], out


lines = []


def emit(s=""):
    print(s)
    lines.append(str(s))


emit("PHASE 1 - BIAS LAYER, DEV WALK-FORWARD (test years 2022, 2023, 2024, 2025H1; expanding train, 5-session purge)")
emit("skill_pct = % reduction in log-loss vs base; t_vs_base = session-clustered t of the per-row log-loss gain;")
emit("years_better = test years where the model beat base. B1 base = random-walk geometry (d_dn / (d_up + d_dn)).")
for task, df in TASKS.items():
    df = df.copy()
    rows, P = run_task(task, df)
    res = summarise(rows, {k: P[k].to_numpy() for k in P.columns})
    emit("\n" + "=" * 110)
    emit(f"{task}   test rows {len(rows):,}   label mean {rows.y.mean():.3f}")
    emit(res.round(4).to_string())
    best = res.drop(index="base").skill_pct.idxmax()
    emit(f"\nbest: {best} - calibration (deciles of predicted p):")
    emit(calibration(rows.y.to_numpy(), P[best].to_numpy()).round(3).to_string())
    pb = P[best].to_numpy()
    y = rows.y.to_numpy()
    for th in (0.6, 0.65, 0.7):
        m = (pb >= th) | (pb <= 1 - th)
        if m.sum():
            acc = np.mean((pb[m] >= 0.5) == (y[m] == 1))
            emit(f"   confident |p-0.5| >= {th - 0.5:.2f}: coverage {100 * m.mean():.1f}%  accuracy {100 * acc:.1f}%  n {m.sum()}")
    out = rows[[c for c in ("sess", "hour", "t") if c in rows.columns]].copy()
    for c in P.columns:
        out[f"p_{c}"] = P[c].to_numpy()
    out["y"] = y
    out.to_parquet(CACHE / f"bias_oof_{task}.parquet", index=False)

(RES / "phase1_bias.txt").write_text("\n".join(lines))
