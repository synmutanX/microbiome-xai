import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")                     # save figures to files instead of opening windows
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

d = pd.read_csv("../shapmat_tutorial/data/curatedCRC.csv", index_col=0)
meta_cols = ["study_name", "CRC", "ajcc_stage"]
X_all = d.drop(columns=meta_cols)
y_all = d["CRC"]

# LODO fold from the paper: train on 4 studies, test on Yachida
is_test = d["study_name"] == "YachidaS_2019"
zero_frac = (X_all[~is_test] == 0).mean()
species = zero_frac[zero_frac <= 0.9].index          # prevalence filter, training studies only
X_train, y_train = X_all.loc[~is_test, species], y_all[~is_test]
X_test, y_test = X_all.loc[is_test, species], y_all[is_test]

model = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1).fit(X_train, y_train)
p_crc = model.predict_proba(X_test)[:, 1]

# --- SHAP values for every test person ---
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_test)                  # shape: (people, species, 2 classes)
print("raw SHAP array shape:", np.shape(sv))
shap_crc = pd.DataFrame(sv[:, :, 1], index=X_test.index, columns=species)   # class 1 = CRC
base = explainer.expected_value[1]                  # average score before looking at any species

# --- Check additivity: base + sum of SHAP = predicted score ---
reconstructed = base + shap_crc.sum(axis=1)
print("base value:", round(base, 3))
print("additivity holds for everyone:", np.allclose(reconstructed, p_crc))

# --- The "receipt" for two people ---
out = Path("results/figures"); out.mkdir(parents=True, exist_ok=True)
for person in ["SAMD00165003", "SAMD00114766"]:
    i = X_test.index.get_loc(person)
    contrib = shap_crc.loc[person]
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)
    print(f"\n{person}  true label = {y_test[person]}  predicted P(CRC) = {p_crc[i]:.2f}")
    print(f"  base {base:.3f} + sum of all SHAP {contrib.sum():+.3f} = {base + contrib.sum():.3f}")
    print("  top 5 contributions:")
    for sp, v in top.items():
        print(f"    {sp:45s} {v:+.3f}   (abundance {X_test.loc[person, sp] * 100:.3f}%)")

    expl = shap.Explanation(values=contrib.values, base_values=base,
                            data=X_test.loc[person].values, feature_names=list(species))
    shap.plots.waterfall(expl, max_display=10, show=False)
    plt.savefig(out / f"waterfall_{person}.png", dpi=150, bbox_inches="tight")
    plt.close()
print("\nwaterfall plots saved in", out)