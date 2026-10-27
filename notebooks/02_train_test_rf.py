import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

d = pd.read_csv("../shapmat_tutorial/data/curatedCRC.csv", index_col=0)

# Use one study only (Yachida), to keep things simple
s = d[d["study_name"] == "YachidaS_2019"]
X = s.drop(columns=["study_name", "CRC", "ajcc_stage"])   # input: species abundances
y = s["CRC"]                                              # answer: 1 = CRC, 0 = healthy

results = []   # one row per (split, model)

for seed in [0, 1, 2]:
    # A different seed gives a different random 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed)

    models = {
        "small tree": DecisionTreeClassifier(max_depth=2, random_state=0),
        "deep tree": DecisionTreeClassifier(random_state=0),
        "random forest": RandomForestClassifier(n_estimators=500, random_state=0),
    }
    for name, m in models.items():
        m.fit(X_train, y_train)
        results.append({
            "seed": seed,
            "model": name,
            "train_acc": m.score(X_train, y_train),
            "test_acc": m.score(X_test, y_test),
        })

    # Show the small tree for each split, with enough decimals to read the thresholds
    print(f"--- small tree, seed {seed} ---")
    print(export_text(models["small tree"], feature_names=list(X.columns),
                      show_weights=True, decimals=6))

res = pd.DataFrame(results)

# Table: rows = seed, columns = model, values = test accuracy
table = res.pivot(index="seed", columns="model", values="test_acc")
print(table.round(2))

# How much does each model's test accuracy move between splits?
print("range (max - min):")
print((table.max() - table.min()).round(2))
print("baseline (always guess CRC):", round(y.mean(), 2))