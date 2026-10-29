import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Species that are all zero inside a training fold make the F-test print warnings; they are harmless
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

d = pd.read_csv("../shapmat_tutorial/data/curatedCRC.csv", index_col=0)
meta_cols = ["study_name", "CRC", "ajcc_stage"]

# --- Part 1: shuffled labels (no real signal); feature selection outside vs inside CV ---
s = d[d["study_name"] == "YachidaS_2019"]
X = s.drop(columns=meta_cols)
X = X.loc[:, (X > 0).any()]     # drop species never detected in Yachida
X = np.log10(X + 1e-5)      # log scale, so a few extreme values don't dominate the test
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

rows = []
for rep in range(10):
    rng = np.random.default_rng(rep)
    y_fake = pd.Series(rng.permutation(s["CRC"].values), index=s.index)     # CRC labels shuffled: no real signal
    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)

    # WRONG: choose the 20 species most associated with the labels using ALL people, then run CV
    selector = SelectKBest(f_classif, k=20).fit(X, y_fake)
    X_top = X.loc[:, selector.get_support()]
    leaky = cross_val_score(rf, X_top, y_fake, cv=cv, scoring="roc_auc").mean()

    # RIGHT: put the selection inside a Pipeline, so it is redone on the training folds only
    pipe = Pipeline([("select", SelectKBest(f_classif, k=20)), ("rf", rf)])
    correct = cross_val_score(pipe, X, y_fake, cv=cv, scoring="roc_auc").mean()

    rows.append({"shuffle": rep, "leaky (selection outside CV)": leaky,
                 "correct (selection inside CV)": correct})

res = pd.DataFrame(rows).set_index("shuffle")
print(res.round(2))
print("mean:")
print(res.mean().round(2))

# --- Part 2: prevalence filter (use no labels) in LODO with Yachida as the test study ---
def prevalent(Xpart, max_zero_frac=0.9):
    """Species that are non-zero in at least 10% of the given samples (SHAPMAT's default)."""
    zero_frac = (Xpart == 0).mean()
    return zero_frac[zero_frac <= max_zero_frac].index

X_all = d.drop(columns=meta_cols)
y_all = d["CRC"]
is_test = d["study_name"] == "YachidaS_2019"

kept_all = prevalent(X_all)     # computed with the test study included (leaky)
kept_train = prevalent(X_all[~is_test])     # computed on the training studies only (correct)
print("species kept, filter on all studies:     ", len(kept_all))
print("species kept, filter on training studies:", len(kept_train))

for label, cols in [("filter on all (leaky)", kept_all), ("filter on train (correct)", kept_train)]:
    m = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    m.fit(X_all.loc[~is_test, cols], y_all[~is_test])
    auc = roc_auc_score(y_all[is_test], m.predict_proba(X_all.loc[is_test, cols])[:, 1])
    print(f"LODO AUC (Yachida), {label}: {auc:.3f}")
