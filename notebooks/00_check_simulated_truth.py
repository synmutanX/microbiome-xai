import json
import pandas as pd

truth = json.load(open("data/simulated/truth.json"))    # ground truth from the simulation
s = json.load(open("results/simulated/summary.json"))   # what the model found
truth_shared = [t.replace("_species_", " species ") for t in truth["shared_signal"]]
truth_drug   = [t.replace("_species_", " species ") for t in truth["drug_taxa"]]

# Part 1: membership in the top 20
for cohort, top20 in s["top_taxa"].items():
    found_shared = [t for t in truth_shared if t in top20]
    found_drug = [t for t in truth_drug if t in top20]
    print(cohort, f"{len(found_shared)}/{len(truth_shared)}", found_shared)
    print(cohort, f"{len(found_drug)}/{len(truth_drug)}", found_drug)

# Part 2: ranks (1 = largest mean |SHAP|)
def label(taxon):
    if taxon in truth_shared:
        return "shared"
    if taxon in truth_drug:
        return "drug"
    return "other"

print()
for cohort in s["top_taxa"]:
    tbl = pd.read_csv(f"results/simulated/tables/shap_importance_{cohort.lower()}.tsv",
                      sep="\t", index_col=0)
    rank = {taxon: i+1 for i, taxon in enumerate(tbl.index)}

    shared_ranks = sorted(rank[t] for t  in truth_shared)
    drug_ranks = {t: rank[t] for t in truth_drug}
    top1 = tbl.index[0]

    print(f"Cohort {cohort} ({len(tbl)} taxa)")
    print("  shared ranks:", shared_ranks)
    print("  drug ranks:  ", drug_ranks)
    print(f"  rank 1: {top1} ({label(top1)})")
    print("  rank of Simgenus15 species 060:", rank["Simgenus15 species 060"])