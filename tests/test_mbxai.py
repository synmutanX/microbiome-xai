import json

import numpy as np
import pandas as pd
import pytest
import yaml

from mbxai.analysis import run
from mbxai.explain import importance_table, oof_shap
from mbxai.io import Cohort, collapse_to_rank, load_cohort, load_mlrepo, parse_lineage, to_relative
from mbxai.preprocess import CLRTransform, LogTransform, PrevalenceFilter
from mbxai.simulate import simulate
from mbxai.stats import bh_fdr, gfc
from mbxai.validate import auc_ci, within_cohort_cv


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    d = tmp_path_factory.mktemp("sim")
    truth = simulate(d, n_taxa=60, n_per_cohort=60, seed=3)
    return d, truth


# ------------------------------------------------------------------- io
def test_parse_lineage_both_separators():
    assert parse_lineage("k__Bacteria;g__Escherichia;s__Escherichia_coli")["s"] == "Escherichia_coli"
    assert parse_lineage("k__Bacteria|p__Proteobacteria|s__Escherichia_coli")["p"] == "Proteobacteria"


def test_collapse_strains_to_species():
    tab = pd.DataFrame({"S1": [1, 2, 3]}, index=[
        "k__B;g__Esch;s__Escherichia_coli;t__K12",
        "k__B;g__Esch;s__Escherichia_coli;t__O157",
        "k__B;g__Bact;s__;t__x",          # unnamed species -> genus fallback
    ])
    out = collapse_to_rank(tab, "s")
    assert out.loc["Escherichia coli", "S1"] == 3
    assert out.loc["unclassified Bact", "S1"] == 3


def test_collapse_hierarchical_metaphlan_does_not_double_count():
    tab = pd.DataFrame({"S1": [100, 60, 60, 40]}, index=[
        "k__Bacteria", "k__Bacteria|g__Esch", "k__Bacteria|g__Esch|s__Escherichia_coli", "k__Bacteria|g__X|s__X_y"])
    out = collapse_to_rank(tab, "s")
    assert out["S1"].sum() == 100 and len(out) == 2


def test_to_relative_drops_empty_samples():
    X = pd.DataFrame({"a": [1, 0], "b": [3, 0]}, index=["s1", "s2"])
    r = to_relative(X)
    assert list(r.index) == ["s1"] and r.loc["s1"].sum() == pytest.approx(1)


def test_load_mlrepo(sim):
    d, _ = sim
    c = load_mlrepo("A", d / "cohort_A", "task-case-control.txt", ["Case"], ["Control"])
    assert c.n == 60 and c.y.sum() == 30
    assert np.allclose(c.X.sum(axis=1), 1)
    assert "Drug" in c.meta.columns


def test_load_table_with_subset(tmp_path):
    ab = pd.DataFrame({"s1": [10, 90], "s2": [50, 50], "s3": [80, 20]},
                      index=["k__B|s__Sp_one", "k__B|s__Sp_two"])
    ab.index.name = "taxon"
    ab.to_csv(tmp_path / "ab.tsv", sep="\t")
    md = pd.DataFrame({"sample_id": ["s1", "s2", "s3"], "study_name": ["X", "X", "Y"],
                       "study_condition": ["CRC", "control", "CRC"]})
    md.to_csv(tmp_path / "md.tsv", sep="\t", index=False)
    spec = {"type": "table", "name": "X", "abundance": "ab.tsv", "metadata": "md.tsv", "sample_column": "sample_id",
            "label_column": "study_condition", "case_values": ["CRC"], "control_values": ["control"],
            "subset": {"study_name": ["X"]}}
    c = load_cohort(spec, tmp_path)
    assert list(c.y.index) == ["s1", "s2"] and list(c.y) == [1, 0]
    assert set(c.X.columns) == {"Sp one", "Sp two"}


# ------------------------------------------------------------ preprocessing
def test_prevalence_filter_learns_on_train_only():
    train = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0, 0, 0.1]})
    test = pd.DataFrame({"a": [0.5], "c": [0.5]})
    f = PrevalenceFilter(min_prevalence=0.5).fit(train)
    assert f.features_ == ["a"]
    assert list(f.transform(test).columns) == ["a"]


def test_clr_rows_sum_to_zero():
    X = pd.DataFrame(np.random.default_rng(0).dirichlet(np.ones(5), 4))
    assert np.allclose(CLRTransform().fit(X).transform(X).sum(axis=1), 0)


def test_log_transform_pseudocount():
    X = pd.DataFrame({"a": [0.0, 0.01]})
    out = LogTransform().fit(X).transform(X)
    assert out.iloc[0, 0] == pytest.approx(-5)


# -------------------------------------------------------------- validation
def test_random_labels_give_chance_auc(sim):
    """Guard against data leakage: with shuffled labels, cross-validated AUROC must be near 0.5."""
    d, _ = sim
    c = load_mlrepo("A", d / "cohort_A", "task-case-control.txt", ["Case"], ["Control"])
    y = pd.Series(np.random.default_rng(1).permutation(c.y.to_numpy()), index=c.y.index)
    shuffled = Cohort("shuffled", c.X, y, c.meta)
    r = within_cohort_cv(shuffled, "rf", n_splits=5, n_repeats=2, seed=0)
    assert 0.3 < r["auc_oof"] < 0.7


def test_auc_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(50), np.zeros(50)]
    p = y + rng.normal(0, 1, 100)
    auc, lo, hi = auc_ci(y, p, n_boot=300)
    assert lo < auc < hi


# ---------------------------------------------------------------- stats
def test_bh_fdr_matches_reference():
    from statsmodels.stats.multitest import multipletests
    p = np.random.default_rng(0).uniform(size=50) ** 2
    assert np.allclose(bh_fdr(p), multipletests(p, method="fdr_bh")[1])


def test_gfc_sign():
    assert gfc(np.array([-2.0, -1.5, -1.0]), np.array([-5.0, -4.0, -4.5])) > 0


# --------------------------------------------------------------- explain
def test_oof_shap_recovers_simulated_signal(sim):
    d, truth = sim
    c = load_mlrepo("A", d / "cohort_A", "task-case-control.txt", ["Case"], ["Control"])
    sh = oof_shap(c, "rf", n_splits=5, seed=0)
    imp = importance_table(sh["shap"], c.X, sh["stability"])
    expected = {t.replace("_", " ") for t in truth["shared_signal"]}
    assert len(expected & set(imp.index[:10])) >= 3


# ------------------------------------------------------------- end to end
def test_run_end_to_end(sim, tmp_path):
    d, _ = sim
    cohorts = [{"name": n, "type": "mlrepo", "directory": str(d / f"cohort_{n}"), "task": "task-case-control.txt",
                "case_values": ["Case"], "control_values": ["Control"]} for n in ("A", "B", "C")]
    config = {"title": "test", "cv": {"n_splits": 3, "n_repeats": 1}, "models": ["rf"], "cohorts": cohorts,
              "confounders": [{"cohort": "A", "column": "Drug", "positive_values": ["yes"], "label": "drug"}],
              "covariates": [{"cohort": "A", "columns": ["BMI", "Age"]}]}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(config))
    summary = run(config, tmp_path / "out")
    assert (tmp_path / "out" / "report.html").stat().st_size > 10_000
    s = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert set(s["models"]["rf"]["transfer_matrix"]) == {"A", "B", "C"}
    assert len(s["models"]["rf"]["loso"]) == 3
    assert summary["confounders"][0]["n_exposed_cases"] > 0
