"""Reproduce Rynazal et al. 2023 (Genome Biology 24:21) across all five LODO folds.

Settings follow the paper's Methods and the original SHAPMAT code:
- abundance filter: values < 1e-15 set to 0; drop species with > 90% zeros
  (computed on the TRAINING studies only, to avoid leakage)
- model: RandomForestClassifier(random_state=0), default 100 trees
- SHAP: TreeExplainer with the test set as background (interventional),
  using ALL test samples as background (newer shap versions subsample to 100)
- keep only correctly predicted subjects (CRC and control) for PCA
- PCA: 2 components, unscaled, random_state=0, species with all-zero SHAP removed
- k-means on PC1/PC2 of CRC subjects only: k-means++, max_iter=1000, n_init=10,
  random_state=0, k = 4; cluster labels numbered 1..k as in the paper

Run from the repo root:
    python scripts/reproduce_rynazal2023.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DATA = Path("../shapmat_tutorial/data/curatedCRC.csv")
OUT = Path("results/rynazal2023")
META_COLS = ["study_name", "CRC", "ajcc_stage"]

SETTINGS = {
    "abundance_threshold": 1e-15,
    "max_zero_frac": 0.9,
    "rf_seed": 0,
    "n_clusters": 4,
    "kmeans_seed": 0,
}


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, meta): species abundances, CRC labels (1/0), and metadata.

    TODO: read the CSV with the sample ID as index (see notebooks/01_data_check.py),
    split off META_COLS, and set abundances below SETTINGS["abundance_threshold"] to 0.
    """
    d = pd.read_csv(path, index_col=0)                 # sample IDs become the index
    meta = d[META_COLS]                                # study_name, CRC, ajcc_stage
    y = d["CRC"]                                       # 1 = CRC, 0 = control
    X = d.drop(columns=META_COLS)                      # species only
    X = X.mask(X < SETTINGS["abundance_threshold"], 0) # values below 1e-15 become 0
    return X, y, meta

    raise NotImplementedError


def prevalent_species(X_train: pd.DataFrame, max_zero_frac: float) -> pd.Index:
    """Species with at most `max_zero_frac` zeros in the TRAINING data.

    TODO: see the `prevalent()` function in notebooks/05_leakage.py.
    """
    raise NotImplementedError


def fit_fold(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, test_study: str):
    """Train on the other four studies, return (model, X_test, y_test, species).

    TODO: build the train/test masks from meta["study_name"], filter species on the
    training part only, and fit RandomForestClassifier(random_state=SETTINGS["rf_seed"]).
    """
    raise NotImplementedError


def shap_for_crc(model, X_test: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Interventional SHAP values for class 1 (CRC), plus the base value.

    TODO:
    - masker = shap.maskers.Independent(X_test, max_samples=len(X_test))
    - explainer = shap.TreeExplainer(model, data=masker)
    - shap_values(X_test) has shape (n_samples, n_species, 2): take [:, :, 1]
    - base value: np.atleast_1d(explainer.expected_value)[-1]
    Return a DataFrame with the same index and columns as X_test.
    """
    raise NotImplementedError


def keep_correct(model, shap_df: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series):
    """Keep only subjects whose prediction equals their label (both classes).

    TODO: see notebooks/08_pca.py. Return (shap_correct, y_correct).
    """
    raise NotImplementedError


def pca_2d(shap_correct: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """PC1/PC2 scores (DataFrame, same index) and explained variance ratios.

    TODO: drop species whose SHAP values are all zero, then
    PCA(n_components=2, random_state=0) without scaling.
    """
    raise NotImplementedError


def elbow_table(scores_crc: pd.DataFrame, ks=range(1, 9)) -> pd.DataFrame:
    """Inertia for each k, to reproduce the elbow plots (Additional file 2, Fig. S1).

    TODO: see the elbow loop in notebooks/09_kmeans.py; use the same KMeans
    settings as cluster_crc().
    """
    raise NotImplementedError


def cluster_crc(scores: pd.DataFrame, y_correct: pd.Series, k: int) -> pd.Series:
    """k-means labels (1..k) for the CRC subjects only, indexed by sample ID.

    TODO: KMeans(n_clusters=k, init="k-means++", max_iter=1000, n_init=10,
    random_state=SETTINGS["kmeans_seed"]); add 1 to the labels.
    """
    raise NotImplementedError


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, meta = load_data(DATA)

    auc_rows = []
    for study in sorted(meta["study_name"].unique()):
        model, X_test, y_test, species = fit_fold(X, y, meta, study)
        p_crc = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
        auc = roc_auc_score(y_test, p_crc)

        shap_df, base = shap_for_crc(model, X_test)
        # Additivity: base + sum of SHAP must equal the predicted score for everyone
        assert np.allclose(base + shap_df.sum(axis=1), p_crc, atol=1e-6), study

        shap_correct, y_correct = keep_correct(model, shap_df, X_test, y_test)
        scores, var_ratio = pca_2d(shap_correct)
        scores_crc = scores[y_correct == 1]
        clusters = cluster_crc(scores, y_correct, SETTINGS["n_clusters"])

        # TODO (session R2): save per-study tables for the figures, e.g.
        #   scores.assign(label=y_correct, p_crc=p_crc).to_csv(OUT / f"pcs_{study}.tsv", sep="\t")
        #   clusters.to_csv(OUT / f"clusters_{study}.tsv", sep="\t")
        #   elbow_table(scores_crc).to_csv(OUT / f"elbow_{study}.tsv", sep="\t")

        auc_rows.append({
            "test_study": study,
            "n_test": len(y_test),
            "n_species": len(species),
            "lodo_auc": round(auc, 3),
            "n_correct": len(y_correct),
            "pc1_var": round(var_ratio[0], 3),
            "pc2_var": round(var_ratio[1], 3),
            "cluster_sizes": clusters.value_counts().sort_index().to_dict(),
        })
        print(auc_rows[-1])

    pd.DataFrame(auc_rows).to_csv(OUT / "lodo_summary.tsv", sep="\t", index=False)
    print("saved", OUT / "lodo_summary.tsv")


if __name__ == "__main__":
    main()
