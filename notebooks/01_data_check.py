import pandas as pd

# --- load ---
# Path is relative to the repo root: run this script from ~/Research/code/microbiome-xai
d = pd.read_csv("../shapmat_tutorial/data/curatedCRC.csv", index_col=0)

print(d.shape)  # (rows, columns)
print(d.columns[-3:])  # last 3 columns = metadata
print(d.head(3))  # first 3 rows

# --- Q1: cases and controls per study ---
counts = pd.crosstab(d["study_name"], d["CRC"], margins=True)
print(counts)

# --- Q2: does each person's abundance sum to 1? ---
meta_cols = ["study_name", "CRC", "ajcc_stage"]
X = d.drop(columns=meta_cols)   # species only
row_sums = X.sum(axis=1)    # one number per person
print(row_sums.describe())  # min, max, mean, std, etc.

# --- Q3: how many species are present in each person or sample? show median of each study ---
# species are present if abundance > 0
species_present = (X > 0).sum(axis=1)  # one number per person
species_present_df = pd.DataFrame({"study_name": d["study_name"], "species_present": species_present})
median_species_per_study = species_present_df.groupby("study_name")["species_present"].median()
print(median_species_per_study)

# --- Mini example: 3 people from Yachida, 3 from Zeller ---
small = pd.concat([d[d["study_name"] == "YachidaS_2019"].head(3),
                   d[d["study_name"] == "ZellerG_2014"].head(3)])
sp = (small.drop(columns=meta_cols) > 0).sum(axis=1)

print(sp)
print(pd.DataFrame({"study_name": small["study_name"], "species_present": sp}))   # your version, step 1
print(sp.groupby(small["study_name"]).median())                                    # my version
print(sp.groupby(small["study_name"].values[::-1]).median())                       # the silent error