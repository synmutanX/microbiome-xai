"""
Simulated multi-cohort microbiome data for tests and demos.

Each cohort has its own baseline community (study / country effect), its own
sequencing depth, a shared disease signal in a few taxa, a cohort-specific
signal in a few others, and a "drug" given to some cases that shifts a further
set of taxa (a confounder). Written in MLRepo format so the real loader is
exercised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def simulate(outdir: str | Path, n_taxa: int = 80, n_per_cohort: int = 80, cohorts=("A", "B", "C"),
             n_shared_signal: int = 6, n_drug: int = 4, effect: float = 1.2, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(outdir)
    taxa = [f"k__Bacteria;p__Simulata;c__Sim;o__Sim;f__Sim;g__Simgenus{i // 4:02d};"
            f"s__Simgenus{i // 4:02d}_species_{i:03d}" for i in range(n_taxa)]
    log_base = rng.normal(0, 1.5, n_taxa)
    shared = rng.choice(n_taxa, n_shared_signal, replace=False)
    shared_eff = rng.choice([-1, 1], n_shared_signal) * effect
    rest = np.setdiff1d(np.arange(n_taxa), shared)
    drug = rng.choice(rest, n_drug, replace=False)
    truth = {"shared_signal": [taxa[i].split("s__")[1] for i in shared], "drug_taxa": [taxa[i].split("s__")[1] for i in drug]}

    for name in cohorts:
        n = n_per_cohort
        y = rng.permutation(np.r_[np.ones(n // 2, int), np.zeros(n - n // 2, int)])
        on_drug = (y == 1) & (rng.random(n) < 0.4)
        log_mu = log_base + rng.normal(0, 0.7, n_taxa)            # cohort-specific community
        own = rng.choice(rest, 3, replace=False)                   # cohort-specific signal
        counts = np.zeros((n_taxa, n), dtype=int)
        for j in range(n):
            lm = log_mu + rng.normal(0, 0.8, n_taxa)
            lm[shared] += y[j] * shared_eff
            lm[own] += y[j] * effect
            lm[drug] += on_drug[j] * 1.5
            dropout = 0.6 / (1 + np.exp(lm + 1))                  # rarer taxa are more often absent
            p = np.exp(lm) * (rng.random(n_taxa) > dropout)
            p /= p.sum()
            counts[:, j] = rng.multinomial(int(rng.integers(20_000, 80_000)), p)
        ids = [f"{name}_{j:03d}" for j in range(n)]
        d = out / f"cohort_{name}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(counts, index=pd.Index(taxa, name="#OTU ID"), columns=ids).to_csv(d / "taxatable.txt", sep="\t")
        pd.DataFrame({"Var": np.where(y == 1, "Case", "Control")}, index=pd.Index(ids, name="#SampleID")) \
            .to_csv(d / "task-case-control.txt", sep="\t")
        pd.DataFrame({"Drug": np.where(on_drug, "yes", "no"), "BMI": np.round(rng.normal(25 + y, 3, n), 1),
                      "Age": rng.integers(30, 70, n)}, index=pd.Index(ids, name="#SampleID")) \
            .to_csv(d / "mapping-orig.txt", sep="\t")
    return truth
