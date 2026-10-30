import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

d = pd.read_csv("../shapmat_tutorial/data/curatedCRC.csv", index_col=0)
meta_cols = ["study_name", "CRC", "ajcc_stage"]
X_all = d.drop(columns=meta_cols)
y_all = d["CRC"]

# Same LODO fold as Session 4: train on 4 studies, test on Yachida
is_test = d["study_name"] == "YachidaS_2019"
zero_frac = (X_all[~is_test] == 0).mean()
species = zero_frac[zero_frac <= 0.9].index
X_train, y_train = X_all.loc[~is_test, species], y_all[~is_test]
X_test, y_test = X_all.loc[is_test, species], y_all[is_test]

model = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1).fit(X_train, y_train)
p_crc = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
sv = shap.TreeExplainer(model).shap_values(X_test)
shap_crc = pd.DataFrame(sv[:, :, 1], index=X_test.index, columns=species)

# As in the paper: keep only people the model predicted correctly
correct = model.predict(X_test) == y_test.values
S = shap_crc[correct]                 # SHAP profiles (people x species)
A = X_test[correct]                   # abundance profiles, same people
lab = y_test[correct]
print("correctly predicted:", correct.sum(), "of", len(y_test),
      "| CRC:", int(lab.sum()), " healthy:", int((lab == 0).sum()))

# --- PCA on SHAP values and on abundances, 2 components each ---
pca_shap = PCA(n_components=2).fit(S)
pca_abun = PCA(n_components=2).fit(A)
Z_shap = pd.DataFrame(pca_shap.transform(S), index=S.index, columns=["PC1", "PC2"])
Z_abun = pd.DataFrame(pca_abun.transform(A), index=A.index, columns=["PC1", "PC2"])
print("variance explained, SHAP PCA:    ", pca_shap.explained_variance_ratio_.round(2))
print("variance explained, abundance PCA:", pca_abun.explained_variance_ratio_.round(2))

# Which species define each axis? (largest loadings)
for name, pca in [("SHAP", pca_shap), ("abundance", pca_abun)]:
    load = pd.DataFrame(pca.components_.T, index=species, columns=["PC1", "PC2"])
    for pc in ["PC1", "PC2"]:
        top = load[pc].abs().sort_values(ascending=False).index[:3]
        print(f"{name} {pc} top loadings:", [f"{sp} ({load.loc[sp, pc]:+.2f})" for sp in top])

# How much does SHAP-PC1 simply track the model's score?
r = np.corrcoef(Z_shap["PC1"], p_crc[correct])[0, 1]
print("correlation of SHAP PC1 with predicted P(CRC):", round(abs(r), 2))

# --- Plot the two maps side by side ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, Z, title in [(axes[0], Z_abun, "PCA on abundance"), (axes[1], Z_shap, "PCA on SHAP values")]:
    for value, color, name in [(0, "tab:blue", "healthy"), (1, "tab:red", "CRC")]:
        z = Z[lab == value]
        ax.scatter(z["PC1"], z["PC2"], s=12, alpha=0.6, c=color, label=name)
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
out = Path("results/figures"); out.mkdir(parents=True, exist_ok=True)
plt.tight_layout(); plt.savefig(out / "pca_abundance_vs_shap.png", dpi=150); plt.close()
print("figure saved in", out)