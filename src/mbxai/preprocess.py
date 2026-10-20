"""
Preprocessing as scikit-learn transformers.

Every step that learns something from the data (which taxa are prevalent,
the pseudocount) is fitted on training samples only and then applied to the
test samples. Doing this outside the cross-validation loop is one of the most
common sources of inflated accuracy in microbiome machine-learning papers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PrevalenceFilter(BaseEstimator, TransformerMixin):
    """Keep taxa detected (abundance > `min_abundance`) in at least `min_prevalence` of training samples."""

    def __init__(self, min_prevalence: float = 0.1, min_abundance: float = 1e-5):
        self.min_prevalence = min_prevalence
        self.min_abundance = min_abundance

    def fit(self, X, y=None):
        X = _as_frame(X)
        prevalence = (X > self.min_abundance).mean(axis=0)
        self.features_ = list(prevalence.index[prevalence >= self.min_prevalence])
        if not self.features_:
            raise ValueError("PrevalenceFilter removed every feature; lower min_prevalence")
        return self

    def transform(self, X):
        return _as_frame(X).reindex(columns=self.features_, fill_value=0.0)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.features_, dtype=object)


class LogTransform(BaseEstimator, TransformerMixin):
    """
    log10(x + pseudocount).

    The default pseudocount (1e-5, as in SIAMCAT's log.std normalisation) keeps
    zeros close to the detection limit; a very small pseudocount makes the
    zero / non-zero split dominate and hurts linear models. With
    `pseudocount=None` it is learned as a fraction of the smallest non-zero
    training value.
    """

    def __init__(self, pseudocount: float | None = 1e-5, fraction: float = 0.5):
        self.pseudocount = pseudocount
        self.fraction = fraction

    def fit(self, X, y=None):
        X = _as_frame(X)
        if self.pseudocount is not None:
            self.pseudocount_ = float(self.pseudocount)
        else:
            nonzero = X.to_numpy()[X.to_numpy() > 0]
            self.pseudocount_ = float(nonzero.min() * self.fraction) if nonzero.size else 1e-6
        self.features_ = list(X.columns)
        return self

    def transform(self, X):
        X = _as_frame(X)
        return np.log10(X + self.pseudocount_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.features_, dtype=object)


class CLRTransform(LogTransform):
    """Centred log-ratio: log(x + pc) minus the per-sample mean of the logs (compositionally coherent)."""

    def transform(self, X):
        logx = np.log(_as_frame(X) + self.pseudocount_)
        return logx.sub(logx.mean(axis=1), axis=0)


class Standardize(BaseEstimator, TransformerMixin):
    """Like StandardScaler, but keeps the DataFrame and its column names."""

    def fit(self, X, y=None):
        X = _as_frame(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0, ddof=0).replace(0, 1.0)
        self.features_ = list(X.columns)
        return self

    def transform(self, X):
        return (_as_frame(X) - self.mean_) / self.scale_

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.features_, dtype=object)


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X))
