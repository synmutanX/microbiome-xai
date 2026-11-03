import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier

d = pd.read_csv("../shapmat_tutorial/data/curatedCRC.csv", index_col=0)
meta_cols = ["study_name", "CRC", "ajcc_stage"]
X_all = d.drop(columns=meta_cols)
y_all = d["CRC"]

# Same setup as Session 5.1: LODO with Yachida as test, SHAP, correct predictions, PCA
is_test = d["study_name"] == "YachidaS_2019"
zero_frac = (X_all[~is_test] == 0).mean()
species = zero_frac[zero_frac <= 0.9].index
X_train, y_train = X_all.loc[~is_test, species], y_all[~is_test]
X_test, y_test = X_all.loc[is_test, species], y_all[is_test]

model = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1).fit(X_train, y_train)
sv = shap.TreeExplainer(model).shap_values(X_test)
shap_crc = pd.DataFrame(sv[:, :, 1], index=X_test.index, columns=species)

correct = model.predict(X_test) == y_test.values
S = shap_crc[correct]
lab = y_test[correct]
Z = pd.DataFrame(PCA(n_components=2).fit_transform(S), index=S.index, columns=["PC1", "PC2"])

# As in the paper: cluster only the CRC patients, on the 2 PCs
Z_crc = Z[lab == 1]
print("CRC patients to cluster:", len(Z_crc))

# --- Choosing k: elbow (inertia) and silhouette ---
rows = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z_crc)
    sil = silhouette_score(Z_crc, km.labels_) if k > 1 else np.nan
    rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
choose = pd.DataFrame(rows).set_index("k")
print(choose.round(4))

# --- Fit k = 4, as in the paper ---
k = 4
km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z_crc)
clusters = pd.Series(km.labels_, index=Z_crc.index, name="cluster")
print("\ncluster sizes:", clusters.value_counts().sort_index().to_dict())

# What drives each cluster? Mean SHAP per species within the cluster, top 3
S_crc = S.loc[Z_crc.index]
for c in range(k):
    mean_shap = S_crc[clusters == c].mean().sort_values(ascending=False)
    print(f"cluster {c}: " + ", ".join(f"{sp} ({v:+.3f})" for sp, v in mean_shap.head(3).items()))

# --- Stability: re-cluster random 80% subsets and compare with the full-data clusters ---
rng = np.random.default_rng(0)
aris = []
for rep in range(50):
    idx = rng.choice(Z_crc.index, size=int(0.8 * len(Z_crc)), replace=False)
    km_sub = KMeans(n_clusters=k, n_init=10, random_state=rep).fit(Z_crc.loc[idx])
    aris.append(adjusted_rand_score(clusters.loc[idx], km_sub.labels_))
print(f"\nstability (ARI over 50 subsamples): mean {np.mean(aris):.2f}, min {np.min(aris):.2f}")

# --- Figures: elbow and clusters on the SHAP map ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(choose.index, choose["inertia"], marker="o")
axes[0].set_xlabel("k (number of clusters)"); axes[0].set_ylabel("inertia"); axes[0].set_title("Elbow plot")
axes[1].scatter(Z[lab == 0]["PC1"], Z[lab == 0]["PC2"], s=10, c="lightgrey", label="healthy")
for c in range(k):
    z = Z_crc[clusters == c]
    axes[1].scatter(z["PC1"], z["PC2"], s=14, label=f"CRC cluster {c}")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[1].set_title("k-means on SHAP-PCA (CRC only)")
axes[1].legend(fontsize=8)
out = Path("results/figures"); out.mkdir(parents=True, exist_ok=True)
plt.tight_layout(); plt.savefig(out / "kmeans_shap_pca.png", dpi=150); plt.close()
print("figure saved in", out)

# --- Who are the clusters? Predicted score and cancer stage per cluster ---
p = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
print(p.loc[Z_crc.index].groupby(clusters).median().round(2))      # median score per cluster
stage = d.loc[Z_crc.index, "ajcc_stage"]
print(pd.crosstab(clusters, stage))                                # stage counts per cluster

# --- Is each cluster enriched for F. nucleatum by abundance (the paper's measure)? ---
fn = X_test.loc[Z_crc.index, "Fusobacterium_nucleatum"]
print(pd.DataFrame({"detected": (fn > 0).groupby(clusters).mean().round(2),
                    "mean_pct": (fn * 100).groupby(clusters).mean().round(3)}))