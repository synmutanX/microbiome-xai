import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

d = pd.read_csv("../shapmat_tutorial/data/curatedCRC.csv", index_col=0)
meta_cols = ["study_name", "CRC", "ajcc_stage"]

# --- Part 1: 5-fold cross-validation inside Yachida, written out by hand ---
s = d[d["study_name"] == "YachidaS_2019"]
X = s.drop(columns=meta_cols)
y = s["CRC"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
fold_aucs = []
tested = pd.Series(0, index=X.index)    # how many times each person is tested
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    m = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1).fit(X_train, y_train)
    auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    fold_aucs.append(auc)
    tested.iloc[test_idx] += 1
    print(f"fold {fold}: train {len(train_idx)} test {len(test_idx)} AUC {auc:.2f}")
print(f"5-fold CV AUC: mean {np.mean(fold_aucs):.2f}, sd {np.std(fold_aucs):.2f}")
print("times each person was tested:", tested.value_counts().to_dict())

# --- Part 1b: the same thing in one line ---
aucs = cross_val_score(RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1),
                       X, y, cv=cv, scoring="roc_auc")
print("cross_val_score:", aucs.round(2))

# --- Part 2: leave-one-dataset-out (LODO) across the 5 studies ---
X_all = d.drop(columns=meta_cols)
y_all = d["CRC"]
rows = []
for study in sorted(d["study_name"].unique()):
    is_test = d["study_name"] == study
    m = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    m.fit(X_all[~is_test], y_all[~is_test])     # ~ means "not": train on all other studies
    auc = roc_auc_score(y_all[is_test], m.predict_proba(X_all[is_test])[:, 1])
    rows.append({"test study": study, "n train": int((~is_test).sum()),
                    "n test": int(is_test.sum()), "LODO AUC": round(auc, 2)})
print(pd.DataFrame(rows))