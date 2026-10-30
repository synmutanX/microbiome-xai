import numpy as np
import pandas as pd
import shap
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 10)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

d = pd.read_csv("../shapmat_tutorial/data/curatedCRC.csv", index_col=0)
meta_cols = ["study_name", "CRC", "ajcc_stage"]
X_all = d.drop(columns=meta_cols)
y_all = d["CRC"]

# Same LODO fold as Session 4.1: train on 4 studies, test on Yachida
is_test = d["study_name"] == "YachidaS_2019"
zero_frac = (X_all[~is_test] == 0).mean()
species = zero_frac[zero_frac <= 0.9].index
X_train, y_train = X_all.loc[~is_test, species], y_all[~is_test]
X_test = X_all.loc[is_test, species]

model = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1).fit(X_train, y_train)
sv = shap.TreeExplainer(model).shap_values(X_test)
shap_crc = pd.DataFrame(sv[:, :, 1], index=X_test.index, columns=species)

# --- Global importance from SHAP: average size of the push, over all test people ---
mean_abs = shap_crc.abs().mean()

# Direction: does more of this species push toward CRC (+) or toward healthy (-)?
direction = pd.Series({sp: spearmanr(X_test[sp], shap_crc[sp])[0] for sp in species})

# --- RF's built-in (Gini) importance: computed during training, no direction ---
gini = pd.Series(model.feature_importances_, index=species)

table = pd.DataFrame({
    "mean_abs_shap": mean_abs,
    "shap_rank": mean_abs.rank(ascending=False).astype(int),
    "gini_rank": gini.rank(ascending=False).astype(int),
    "direction": np.where(direction > 0, "higher in CRC", "higher in healthy"),
}).sort_values("mean_abs_shap", ascending=False)
print(table.head(15).round(4))

top_shap = set(table.index[:20])
top_gini = set(gini.sort_values(ascending=False).index[:20])
print("\noverlap of top-20 (SHAP vs Gini):", len(top_shap & top_gini), "of 20")
print("rank correlation over all species:", round(spearmanr(mean_abs, gini)[0], 2))

# --- Summary (beeswarm) plot: one dot per person per species ---
out = Path("results/figures"); out.mkdir(parents=True, exist_ok=True)
shap.summary_plot(shap_crc.values, X_test, max_display=15, show=False)
plt.savefig(out / "shap_summary_yachida.png", dpi=150, bbox_inches="tight")
plt.close()
print("summary plot saved in", out)