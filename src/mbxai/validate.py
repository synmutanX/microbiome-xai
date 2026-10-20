"""
Validation schemes.

* within-cohort: repeated stratified k-fold cross-validation
* cross-cohort transfer: train on all samples of cohort A, test on cohort B
* leave-one-study-out (LOSO): train on all cohorts but one, test on the one left out

Within-cohort accuracy answers "is there a signal in this population?";
transfer and LOSO answer the question industry actually cares about: "will the
biomarker work on people from a different study, country or lab?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold

from .io import Cohort, align_features, shared_feature_space
from .models import make_model


def auc_ci(y, p, n_boot: int = 1000, seed: int = 0, level: float = 0.95):
    """AUROC with a stratified bootstrap percentile confidence interval."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boots.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(roc_auc_score(y, p)), float(lo), float(hi)


def within_cohort_cv(cohort: Cohort, model_name: str, n_splits: int = 5, n_repeats: int = 10,
                     seed: int = 42, **model_kw) -> dict:
    """Repeated stratified CV. Returns fold AUCs and out-of-fold probabilities averaged over repeats."""
    X, y = cohort.X, cohort.y
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    oof = np.zeros((n_repeats, len(y)))
    fold_aucs = []
    base = make_model(model_name, seed=seed, **model_kw)
    for k, (tr, te) in enumerate(cv.split(X, y)):
        model = clone(base).fit(X.iloc[tr], y.iloc[tr])
        prob = model.predict_proba(X.iloc[te])[:, 1]
        oof[k // n_splits, te] = prob
        fold_aucs.append(roc_auc_score(y.iloc[te], prob))
    prob = oof.mean(axis=0)
    auc, lo, hi = auc_ci(y, prob, seed=seed)
    return {"auc_mean_fold": float(np.mean(fold_aucs)), "auc_sd_fold": float(np.std(fold_aucs)),
            "auc_oof": auc, "ci_low": lo, "ci_high": hi,
            "oof_prob": pd.Series(prob, index=y.index), "fold_aucs": fold_aucs}


def _stack(cohorts: list[Cohort]):
    cols = shared_feature_space(cohorts)
    X = pd.concat([align_features(c.X, cols) for c in cohorts])
    y = pd.concat([c.y for c in cohorts])
    return X, y


def train_test(train: list[Cohort], test: Cohort, model_name: str, seed: int = 42, **model_kw) -> dict:
    """Fit on all samples of the training cohort(s), evaluate on `test`."""
    X_tr, y_tr = _stack(train)
    model = make_model(model_name, seed=seed, **model_kw).fit(X_tr, y_tr)
    X_te = align_features(test.X, X_tr.columns)
    prob = model.predict_proba(X_te)[:, 1]
    auc, lo, hi = auc_ci(test.y, prob, seed=seed)
    # how many of the model's taxa are actually observed in the test cohort?
    used = model["filter"].features_
    observed = (test.X.reindex(columns=used, fill_value=0) > 0).any(axis=0).mean()
    return {"auc": auc, "ci_low": lo, "ci_high": hi, "prob": pd.Series(prob, index=test.y.index),
            "model": model, "feature_coverage": float(observed)}


def transfer_matrix(cohorts: list[Cohort], model_name: str, cv_results: dict | None = None,
                    seed: int = 42, **model_kw) -> tuple[pd.DataFrame, dict]:
    """
    AUROC matrix: rows = training cohort, columns = test cohort.
    The diagonal is the within-cohort cross-validated AUROC.
    """
    names = [c.name for c in cohorts]
    mat = pd.DataFrame(np.nan, index=names, columns=names)
    details = {}
    for a in cohorts:
        for b in cohorts:
            if a.name == b.name:
                r = cv_results[a.name] if cv_results else within_cohort_cv(a, model_name, seed=seed, **model_kw)
                mat.loc[a.name, b.name] = r["auc_oof"]
            else:
                r = train_test([a], b, model_name, seed=seed, **model_kw)
                mat.loc[a.name, b.name] = r["auc"]
                details[(a.name, b.name)] = r
    return mat, details


def leave_one_study_out(cohorts: list[Cohort], model_name: str, seed: int = 42, **model_kw) -> pd.DataFrame:
    """Only meaningful with three or more cohorts."""
    rows = []
    for held in cohorts:
        rest = [c for c in cohorts if c.name != held.name]
        r = train_test(rest, held, model_name, seed=seed, **model_kw)
        rows.append({"held_out": held.name, "trained_on": "+".join(c.name for c in rest),
                     "auc": r["auc"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return pd.DataFrame(rows)


def roc_points(y, prob):
    fpr, tpr, _ = roc_curve(y, prob)
    return fpr, tpr
