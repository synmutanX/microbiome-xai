import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --- Part 1: toy example you can check by hand ---
y_true = [1, 1, 1, 0, 0, 0]
score = [0.90, 0.70, 0.45, 0.60, 0.30, 0.20]

pred = [1 if s > 0.5 else 0 for s in score]
print("toy accuracy:", round(accuracy_score(y_true, pred), 2))
print("toy AUC:", round(roc_auc_score(y_true, score), 2))

# Shift every score by +0.3: the ranking stays the same
shifted = [s + 0.4 for s in score]
pred_s = [1 if s > 0.5 else 0 for s in shifted]
print("shifted accuracy:", round(accuracy_score(y_true, pred_s), 2),
    "   shifted AUC:", round(roc_auc_score(y_true, shifted), 2))

# Each treshold gives one (FPR, TPR) point on the ROC curve
fpr, tpr, thr = roc_curve(y_true, score)
print(pd.DataFrame({"threshold": thr, "FPR": fpr, "TPR": tpr}).round(2))

# --- Part 2: Yachida, accuracy vs AUC over three splits ---
d = pd.read_csv("../shapmat_tutorial/data/curatedCRC.csv", index_col=0)
s = d[d["study_name"] == "YachidaS_2019"]
X = s.drop(columns=["study_name", "CRC", "ajcc_stage"])
y = s["CRC"]

rows = []
for seed in [0, 1, 2]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed)

    models = {
        "deep tree": DecisionTreeClassifier(random_state=0),
        "random forest": RandomForestClassifier(n_estimators=500, random_state=0),
    }
    for name, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict_proba(X_test)[:, 1]   # probability score of class 1 (CRC) = P(CRC)
        rows.append({
            "seed": seed,
            "model": name,
            "accuracy": accuracy_score(y_test, (p > 0.5).astype(int)),
            "AUC": roc_auc_score(y_test, p),
            "n_unique_scores": len(set(p)),     # how many different scores the model gives
        })

print(pd.DataFrame(rows).round(2))
