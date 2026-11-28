"""Phase 1 models and walk-forward evaluation (PLAN §5).

Folds: expanding window, test years 2022, 2023, 2024, 2025 (to 2025-07-09); training = everything before the test
year minus a 5-session purge. HMM, scalers and tables are refitted inside each fold.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
EPS = 1e-4
TEST_YEARS = [2022, 2023, 2024, 2025]
PURGE = 5
HMM_OBS = ["rth_ret_atr", "rth_range_atr", "on_range_atr"]


# ------------------------------------------------------------------------------------------------- HMM
def hmm_filtered(days, train_mask, n_states, seed=0):
    """Fit a Gaussian HMM on training days; return P(state today | info up to yesterday) for EVERY day
    (forward filter + one transition step - never smoothed). States sorted by mean RTH range."""
    from hmmlearn.hmm import GaussianHMM
    X = days[HMM_OBS].to_numpy(float)
    mu, sd = X[train_mask].mean(0), X[train_mask].std(0)
    Z = (X - mu) / sd
    m = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=seed)
    m.fit(Z[train_mask])
    order = np.argsort(m.means_[:, 1])
    ll = m._compute_log_likelihood(Z)
    A = m.transmat_
    alpha = m.startprob_.copy()
    prior = np.empty((len(Z), n_states))
    for t in range(len(Z)):
        prior[t] = alpha @ A if t else m.startprob_
        post = prior[t] * np.exp(ll[t] - ll[t].max())
        alpha = post / post.sum()
    return prior[:, order]


# ------------------------------------------------------------------------------------------------- models
def _pipe(est):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


def logit(C=1.0):
    return _pipe(LogisticRegression(C=C, max_iter=2000))


def knn(k):
    return _pipe(KNeighborsClassifier(n_neighbors=k, weights="distance"))


def gbm():
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                        l2_regularization=1.0, min_samples_leaf=40,
                                                        random_state=0))


def mlp():
    return _pipe(MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2, early_stopping=True,
                               max_iter=500, random_state=0))


def table_predict(train, test, keys, y, prior_tr, prior_te, shrink=50):
    """Conditional table: residual (y - prior) averaged per cell, shrunk toward 0 by n/(n+shrink)."""
    res = (train[y] - prior_tr).groupby([train[k] for k in keys])
    adj = res.mean() * res.size() / (res.size() + shrink)
    idx = pd.MultiIndex.from_frame(test[keys]) if len(keys) > 1 else pd.Index(test[keys[0]])
    a = pd.Series(adj.reindex(idx).to_numpy(), index=test.index).fillna(0.0)
    return np.clip(prior_te + a.to_numpy(), EPS, 1 - EPS)


# ------------------------------------------------------------------------------------------------- metrics
def logloss_rows(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def summarise(df, preds, base="base"):
    """preds: dict name -> array aligned with df (test rows only). Session-clustered t-stat of LL improvement."""
    y = df.y.to_numpy()
    ll0 = logloss_rows(y, preds[base])
    out = []
    for name, p in preds.items():
        ll = logloss_rows(y, p)
        diff = pd.Series(ll0 - ll).groupby(df.sess.to_numpy()).mean()
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if name != base else np.nan
        per_year = pd.Series(ll0 - ll).groupby(df.sess.dt.year.to_numpy()).mean()
        out.append(dict(model=name, n=len(y), logloss=ll.mean(), skill_pct=100 * (1 - ll.mean() / ll0.mean()),
                        t_vs_base=t, years_better=f"{int((per_year > 0).sum())}/{len(per_year)}" if name != base else "",
                        brier=np.mean((p - y) ** 2), auc=roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan))
    return pd.DataFrame(out).set_index("model")


def calibration(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    return pd.DataFrame({"pred": pd.Series(p).groupby(q).mean(), "obs": pd.Series(y).groupby(q).mean(),
                         "n": pd.Series(y).groupby(q).size()})


def folds(df):
    for yr in TEST_YEARS:
        start = pd.Timestamp(f"{yr}-01-01")
        sess = np.sort(df.sess.unique())
        before = sess[sess < start]
        if len(before) <= PURGE:
            continue
        cut = before[-PURGE]
        tr = df.sess < cut
        te = df.sess.dt.year == yr
        yield yr, tr.to_numpy(), te.to_numpy()
