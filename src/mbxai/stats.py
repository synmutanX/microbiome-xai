"""
Univariate differential-abundance tests, as a model-free cross-check of what
the classifiers find.

Effect size is the generalised fold change (gFC) used by SIAMCAT
(Wirbel et al. 2021): the mean difference between case and control quantiles
(10%..90%) of log10 abundance. It is robust to the many zeros in microbiome
data, unlike a ratio of means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .io import Cohort

PSEUDOCOUNT = 1e-6
QUANTILES = np.arange(0.1, 0.91, 0.1)


def bh_fdr(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def gfc(case: np.ndarray, control: np.ndarray) -> float:
    return float(np.mean(np.quantile(case, QUANTILES) - np.quantile(control, QUANTILES)))


def compare_groups(X: pd.DataFrame, mask_a: pd.Series, mask_b: pd.Series, min_prevalence: float = 0.1) -> pd.DataFrame:
    """Wilcoxon rank-sum test of group a vs group b for every sufficiently prevalent taxon."""
    sub = X[mask_a | mask_b]
    prevalent = sub.columns[(sub > 0).mean() >= min_prevalence]
    logx = np.log10(X[prevalent] + PSEUDOCOUNT)
    a, b = logx[mask_a], logx[mask_b]
    rows = []
    for f in prevalent:
        stat, p = mannwhitneyu(a[f], b[f], alternative="two-sided")
        rows.append({"taxon": f, "gfc": gfc(a[f].to_numpy(), b[f].to_numpy()), "auc": stat / (len(a) * len(b)),
                     "p": p, "prevalence_a": float((X.loc[mask_a, f] > 0).mean()),
                     "prevalence_b": float((X.loc[mask_b, f] > 0).mean())})
    out = pd.DataFrame(rows).set_index("taxon")
    out["fdr"] = bh_fdr(out["p"]) if len(out) else []
    return out.sort_values("p")


def differential_abundance(cohort: Cohort, min_prevalence: float = 0.1) -> pd.DataFrame:
    """Cases vs controls. Positive gFC = more abundant in cases."""
    return compare_groups(cohort.X, cohort.y == 1, cohort.y == 0, min_prevalence)


def consistency(da_tables: dict[str, pd.DataFrame], fdr: float = 0.1) -> pd.DataFrame:
    """Taxa tested in every cohort, with per-cohort gFC/FDR and whether the direction agrees everywhere."""
    names = list(da_tables)
    shared = set.intersection(*(set(t.index) for t in da_tables.values()))
    rows = []
    for f in sorted(shared):
        row = {"taxon": f}
        signs = []
        for n in names:
            row[f"gfc_{n}"] = da_tables[n].loc[f, "gfc"]
            row[f"fdr_{n}"] = da_tables[n].loc[f, "fdr"]
            signs.append(np.sign(row[f"gfc_{n}"]))
        row["same_direction"] = len(set(signs)) == 1 and signs[0] != 0
        row["significant_in"] = sum(row[f"fdr_{n}"] < fdr for n in names)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.set_index("taxon").sort_values(["significant_in", "same_direction"], ascending=False)
