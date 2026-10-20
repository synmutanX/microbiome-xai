"""
Model definitions. Each model is a single scikit-learn Pipeline, so the whole
thing (filtering, transformation, scaling, classifier) is refitted inside every
cross-validation fold.

* ``lasso`` : L1-penalised logistic regression on standardised log10 abundances,
              penalty tuned by inner cross-validation. Sparse and interpretable.
* ``rf``    : random forest on log10 abundances. Captures non-linear effects and
              interactions; explained with TreeSHAP.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline

from .preprocess import CLRTransform, LogTransform, PrevalenceFilter, Standardize

MODELS = ("lasso", "rf")


def make_model(name: str, seed: int = 42, min_prevalence: float = 0.1, transform: str = "log",
               n_jobs: int = 1) -> Pipeline:
    trans = CLRTransform() if transform == "clr" else LogTransform()
    if name == "lasso":
        clf = LogisticRegressionCV(
            Cs=10, cv=5, l1_ratios=[1.0], solver="liblinear", scoring="roc_auc",
            class_weight="balanced", max_iter=5000, random_state=seed, n_jobs=n_jobs,
        )
        steps = [("filter", PrevalenceFilter(min_prevalence)), ("transform", trans),
                 ("scale", Standardize()), ("clf", clf)]
    elif name == "rf":
        clf = RandomForestClassifier(
            n_estimators=500, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=seed, n_jobs=n_jobs,
        )
        steps = [("filter", PrevalenceFilter(min_prevalence)), ("transform", trans), ("clf", clf)]
    else:
        raise ValueError(f"unknown model '{name}', choose from {MODELS}")
    return Pipeline(steps)


def preprocess_only(pipe: Pipeline, X):
    """Run every step except the classifier (used to feed SHAP)."""
    return pipe[:-1].transform(X)
