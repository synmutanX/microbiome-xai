import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

d = pd.read_csv("../shapmat_tutorial/data/curatedCRC.csv", index_col=0)

# Use one study only (Yachida), to keep things simple
s = d[d["study_name"] == "YachidaS_2019"]
X = s.drop(columns=["study_name", "CRC", "ajcc_stage"]) # input : species abundances
y = s["CRC"]                                            # answer : 1 = CRC, 0 = healthy

# Split: 70% train, 30% test; stratify keeps the CRC/healthy ratio equal in both parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# 1) A small tree you can read
small = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
print(export_text(small, feature_names=list(X.columns), show_weights=True))

# 2) Compare train vs test accuracy for three models
models = {
    "small tree (depth=2)": small,
    "deep tree (no limit)": DecisionTreeClassifier(random_state=0).fit(X_train, y_train),
    "random forest (500 trees)": RandomForestClassifier(n_estimators=500, random_state=0).fit(X_train, y_train),
}
for name, m in models.items():
    print(f"{name:28s} train acc: {m.score(X_train, y_train):.2f} test acc: {m.score(X_test, y_test):.2f}")

# 3) Baseline: always guess the most common class
print("baseline (always guess CRC):", round(y_test.mean(), 2))

# 4) Probabilities for the first 5 test people
rf = models["random forest (500 trees)"]
print("predicted P(CRC) :", rf.predict_proba(X_test[:5])[:, 1].round(2))
print("true labels     :", y_test[:5].tolist())