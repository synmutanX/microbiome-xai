"""
Is the microbiome signal really about the disease?

Drug treatment, BMI, diet or sequencing batch can change the microbiome and
also correlate with disease status. A classical example is metformin in type 2
diabetes: it shifts the gut microbiome strongly, so a "T2D signature" learned
from treated patients may partly be a "metformin signature" (Forslund et al.
2015).

Three checks:
1. treatment_effects    : among cases, which taxa differ between treated and untreated?
2. sensitivity_analysis : retrain without treated cases; does accuracy hold and do the
                          explanations stay the same?
3. added_value          : does the microbiome score predict the outcome beyond clinical
                          covariates (e.g. BMI, age)? Likelihood-ratio test + AUROCs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .io import Cohort
from .stats import compare_groups


def exposure_mask(cohort: Cohort, column: str, positive_values) -> pd.Series:
    values = cohort.meta[column].astype(str).str.strip()
    return values.isin([str(v) for v in np.atleast_1d(positive_values)])


def treatment_effects(cohort: Cohort, column: str, positive_values, min_prevalence: float = 0.1) -> pd.DataFrame:
    """Within cases: exposed vs unexposed. Positive gFC = higher in exposed."""
    exposed = exposure_mask(cohort, column, positive_values)
    cases = cohort.y == 1
    return compare_groups(cohort.X, cases & exposed, cases & ~exposed, min_prevalence)


def drop_exposed_cases(cohort: Cohort, column: str, positive_values) -> Cohort:
    exposed = exposure_mask(cohort, column, positive_values) & (cohort.y == 1)
    keep = ~exposed
    return Cohort(f"{cohort.name} (no {column})", cohort.X[keep], cohort.y[keep], cohort.meta[keep],
                  cohort.case_label, cohort.control_label)


def added_value(y: pd.Series, score: pd.Series, covariates: pd.DataFrame, seed: int = 42) -> dict:
    """
    Compare clinical covariates alone, the microbiome score alone, and both together.
    `score` must be out-of-fold (e.g. from within_cohort_cv) so that it is not overfitted.
    """
    df = pd.concat([y.rename("y"), score.rename("score"), covariates.apply(pd.to_numeric, errors="coerce")], axis=1)
    df = df.dropna()
    cov_cols = list(covariates.columns)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    def cv_auc(cols):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        prob = cross_val_predict(model, df[cols], df["y"], cv=cv, method="predict_proba")[:, 1]
        return float(roc_auc_score(df["y"], prob))

    null = sm.Logit(df["y"], sm.add_constant(df[cov_cols])).fit(disp=0)
    full = sm.Logit(df["y"], sm.add_constant(df[cov_cols + ["score"]])).fit(disp=0)
    lr = 2 * (full.llf - null.llf)
    from scipy.stats import chi2
    return {"n": int(len(df)), "covariates": cov_cols,
            "auc_covariates": cv_auc(cov_cols), "auc_microbiome": cv_auc(["score"]),
            "auc_combined": cv_auc(cov_cols + ["score"]),
            "lrt_stat": float(lr), "lrt_p": float(chi2.sf(lr, 1)),
            "odds_ratio_per_sd_score": float(np.exp(full.params["score"] * df["score"].std()))}
