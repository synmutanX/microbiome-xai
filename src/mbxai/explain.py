"""
Explainability with SHAP (Lundberg & Lee 2017; Lundberg et al. 2020).

Out-of-fold SHAP
----------------
SHAP values computed on the same samples the model was trained on describe
what the model memorised as much as what generalises. Here every sample is
explained by a model that did *not* see it: the cohort is split into folds,
each fold's model explains the held-out fold, and the per-sample SHAP values
are collected. Feature selection (prevalence filtering) happens inside each
fold, so a taxon that a fold's model did not use contributes exactly 0.

Outputs per taxon: mean |SHAP| (importance), direction (does higher abundance
push predictions toward case or control?) and stability (in how many folds is
it among the top features?).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from .io import Cohort
from .models import make_model


def shap_for_model(model, X_explain: pd.DataFrame, X_background: pd.DataFrame) -> pd.DataFrame:
    """SHAP values (case class, samples x model features) for a fitted Pipeline."""
    Xt = model[:-1].transform(X_explain)
    clf = model[-1]
    if isinstance(clf, RandomForestClassifier):
        values = shap.TreeExplainer(clf).shap_values(Xt.to_numpy())
        values = np.asarray(values)
        if values.ndim == 3:      # (n, features, classes) in recent SHAP, (classes, n, features) in older ones
            values = values[..., 1] if values.shape[-1] == 2 else values[1]
    else:
        bg = model[:-1].transform(X_background)
        explainer = shap.LinearExplainer(clf, bg.to_numpy())
        values = np.asarray(explainer.shap_values(Xt.to_numpy()))
    return pd.DataFrame(values, index=X_explain.index, columns=Xt.columns)


def oof_shap(cohort: Cohort, model_name: str, n_splits: int = 5, seed: int = 42, top_k: int = 20,
             **model_kw) -> dict:
    X, y = cohort.X, cohort.y
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    parts, top_sets = [], []
    for tr, te in folds.split(X, y):
        model = make_model(model_name, seed=seed, **model_kw).fit(X.iloc[tr], y.iloc[tr])
        sv = shap_for_model(model, X.iloc[te], X.iloc[tr])
        parts.append(sv)
        top_sets.append(set(sv.abs().mean().sort_values(ascending=False).index[:top_k]))
    shap_df = pd.concat(parts).reindex(X.index).fillna(0.0)
    shap_df = shap_df.loc[:, shap_df.abs().sum() > 0]
    stability = pd.Series({f: np.mean([f in s for s in top_sets]) for f in shap_df.columns})
    return {"shap": shap_df, "stability": stability}


def importance_table(shap_df: pd.DataFrame, X: pd.DataFrame, stability: pd.Series | None = None) -> pd.DataFrame:
    """mean |SHAP|, direction and prevalence per taxon, sorted by importance."""
    logx = np.log10(X[shap_df.columns] + 1e-6)
    rows = []
    for f in shap_df.columns:
        rho = spearmanr(logx[f], shap_df[f]).statistic if logx[f].nunique() > 1 else np.nan
        rows.append({"taxon": f, "mean_abs_shap": shap_df[f].abs().mean(),
                     "direction": "higher in case" if rho > 0 else "higher in control" if rho < 0 else "unclear",
                     "shap_value_corr": rho, "prevalence": float((X[f] > 0).mean())})
    out = pd.DataFrame(rows).set_index("taxon")
    if stability is not None:
        out["top_k_stability"] = stability.reindex(out.index)
    return out.sort_values("mean_abs_shap", ascending=False)


def compare_importance(tables: dict[str, pd.DataFrame], top_k: int = 20) -> dict:
    """How similar are the explanations learned in different cohorts?"""
    names = list(tables)
    out = {"pairs": []}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ta, tb = tables[a], tables[b]
            shared = ta.index.intersection(tb.index)
            rho = spearmanr(ta.loc[shared, "mean_abs_shap"], tb.loc[shared, "mean_abs_shap"]).statistic \
                if len(shared) > 2 else np.nan
            top_a, top_b = set(ta.index[:top_k]), set(tb.index[:top_k])
            both = [f for f in ta.index[:top_k] if f in top_b]
            same_dir = [f for f in both if ta.loc[f, "direction"] == tb.loc[f, "direction"]]
            out["pairs"].append({
                "cohort_a": a, "cohort_b": b, "shared_taxa": int(len(shared)),
                "importance_spearman": float(rho), f"top{top_k}_jaccard": len(top_a & top_b) / len(top_a | top_b),
                f"top{top_k}_overlap": both, "overlap_same_direction": same_dir,
            })
    return out
