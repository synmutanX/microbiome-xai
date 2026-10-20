"""End-to-end analysis driven by a YAML config: validate, explain, check confounders, report."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import report as rp
from .confounders import added_value, drop_exposed_cases, treatment_effects
from .explain import compare_importance, importance_table, oof_shap
from .io import load_cohort
from .stats import consistency, differential_abundance
from .validate import leave_one_study_out, roc_points, transfer_matrix, within_cohort_cv

log = logging.getLogger("mbxai")


def run(config: dict, outdir: str | Path, data_root: str | Path = ".") -> dict:
    out = Path(outdir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    seed = config.get("seed", 42)
    cvp = config.get("cv", {})
    n_splits, n_repeats = cvp.get("n_splits", 5), cvp.get("n_repeats", 10)
    model_kw = {"min_prevalence": config.get("min_prevalence", 0.1), "transform": config.get("transform", "log"),
                "n_jobs": config.get("n_jobs", -1)}
    models = config.get("models", ["lasso", "rf"])
    explain_model = config.get("explain_model", "rf")
    top_k = config.get("top_k", 20)

    cohorts = [load_cohort(spec, data_root) for spec in config["cohorts"]]
    if config.get("shared_taxa_only") and len(cohorts) > 1:
        # keep only taxa detected in every cohort: removes features that cannot transfer for technical reasons
        shared = set.intersection(*({t for t in c.X.columns if (c.X[t] > 0).any()} for c in cohorts))
        for c in cohorts:
            c.X = c.X[sorted(shared)]
        log.info("shared_taxa_only: %d taxa detected in all cohorts", len(shared))
    by_name = {c.name: c for c in cohorts}
    summary = {"title": config.get("title", "Microbiome biomarker report"),
               "cohorts": [c.summary() for c in cohorts], "models": {}}
    sections = []
    if config.get("description"):
        sections.append(f"<p class='note'>{config['description']}</p>")
    sections.append("<h2>Cohorts</h2>" + rp.table_html(pd.DataFrame(summary["cohorts"]), index=False))

    # ---------------------------------------------------------------- validation
    cv_all = {}
    for m in models:
        log.info("model %s: within-cohort CV", m)
        cv = {c.name: within_cohort_cv(c, m, n_splits, n_repeats, seed, **model_kw) for c in cohorts}
        cv_all[m] = cv
        res = {"within_cohort": {k: {kk: v[kk] for kk in ("auc_oof", "ci_low", "ci_high", "auc_mean_fold", "auc_sd_fold")}
                                 for k, v in cv.items()}}
        if len(cohorts) > 1:
            log.info("model %s: cross-cohort transfer", m)
            mat, details = transfer_matrix(cohorts, m, cv, seed, **model_kw)
            mat.to_csv(out / "tables" / f"transfer_auc_{m}.tsv", sep="\t")
            res["transfer"] = {f"{a}->{b}": {"auc": d["auc"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                                            "feature_coverage": d["feature_coverage"]} for (a, b), d in details.items()}
            res["transfer_matrix"] = mat.round(3).to_dict()
        if len(cohorts) > 2:
            loso = leave_one_study_out(cohorts, m, seed, **model_kw)
            loso.to_csv(out / "tables" / f"loso_{m}.tsv", sep="\t", index=False)
            res["loso"] = loso.to_dict(orient="records")
        summary["models"][m] = res

    kpi = "".join(
        f"<div><span>{c.name} · {m} · within-cohort</span><b>{summary['models'][m]['within_cohort'][c.name]['auc_oof']:.2f}</b>"
        f"<span>AUROC</span></div>" for m in models for c in cohorts)
    if len(cohorts) > 1:
        kpi += "".join(f"<div><span>{pair} · {m} · transfer</span><b>{r['auc']:.2f}</b><span>AUROC</span></div>"
                       for m in models for pair, r in summary["models"][m]["transfer"].items())
    sections.insert(1, f"<div class='kpi'>{kpi}</div>")

    sec = "<h2>1. Does the signal hold within each cohort, and does it transfer?</h2>" \
          "<p class='note'>Diagonal: repeated 5-fold cross-validation within a cohort. Off-diagonal: model trained " \
          "on all samples of one cohort, tested on another. 0.5 = chance.</p>"
    if len(cohorts) > 1:
        for m in models:
            mat = pd.DataFrame(summary["models"][m]["transfer_matrix"]).loc[[c.name for c in cohorts]]
            b64 = rp.plot_transfer(mat, f"AUROC, {m}", out / "figures" / f"transfer_{m}.png")
            sec += rp.img(b64, f"transfer matrix {m}")
    curves = {}
    for c in cohorts:
        r = cv_all[explain_model][c.name]
        fpr, tpr = roc_points(c.y, r["oof_prob"])
        curves[c.name] = (fpr, tpr, r["auc_oof"])
    sec += rp.img(rp.plot_roc(curves, f"Within-cohort ROC ({explain_model}, out-of-fold)",
                              out / "figures" / "roc_within.png"), "ROC curves")
    rows = [{"model": m, "cohort": k, **{kk: round(v, 3) for kk, v in r.items()}}
            for m in models for k, r in summary["models"][m]["within_cohort"].items()]
    sec += rp.table_html(pd.DataFrame(rows), index=False)
    if len(cohorts) > 2:
        for m in models:
            sec += f"<p><b>Leave-one-study-out ({m})</b></p>" + rp.table_html(pd.DataFrame(summary["models"][m]["loso"]),
                                                                            index=False)
    sections.append(sec)

    # ------------------------------------------------------------ explanation
    log.info("out-of-fold SHAP (%s)", explain_model)
    imp = {}
    for c in cohorts:
        sh = oof_shap(c, explain_model, n_splits, seed, top_k, **model_kw)
        imp[c.name] = importance_table(sh["shap"], c.X, sh["stability"])
        imp[c.name].to_csv(out / "tables" / f"shap_importance_{_slug(c.name)}.tsv", sep="\t")
    da = {c.name: differential_abundance(c, model_kw["min_prevalence"]) for c in cohorts}
    for name, t in da.items():
        t.to_csv(out / "tables" / f"differential_abundance_{_slug(name)}.tsv", sep="\t")
    sec = ("<h2>2. Which taxa drive the predictions?</h2><p class='note'>Out-of-fold SHAP: each sample is explained by "
           "a model that did not see it. Stability = share of folds in which the taxon is in the top "
           f"{top_k}.</p>")
    sec += rp.img(rp.plot_importance(imp, 15, out / "figures" / "shap_importance.png"), "SHAP importance")
    for name, t in imp.items():
        merged = t.head(15).join(da[name][["gfc", "fdr"]], how="left")
        sec += f"<p><b>{name}</b></p>" + rp.table_html(merged)
    summary["top_taxa"] = {k: list(v.index[:top_k]) for k, v in imp.items()}
    if len(cohorts) > 1:
        cmp = compare_importance(imp, top_k)
        summary["explanation_agreement"] = cmp["pairs"]
        a, b = cohorts[0].name, cohorts[1].name
        sec += rp.img(rp.plot_importance_agreement(imp[a], imp[b], a, b, out / "figures" / "shap_agreement.png"),
                      "SHAP agreement")
        sec += rp.table_html(pd.DataFrame(cmp["pairs"]), index=False)
        cons = consistency(da)
        cons.to_csv(out / "tables" / "differential_abundance_consistency.tsv", sep="\t")
        summary["taxa_significant_in_all_cohorts"] = list(cons.index[(cons["significant_in"] == len(cohorts))
                                                                     & cons["same_direction"]]) if len(cons) else []
        sec += ("<p><b>Univariate cross-check</b>: taxa significant (FDR &lt; 0.1) with the same direction in every "
                f"cohort: {', '.join(summary['taxa_significant_in_all_cohorts']) or 'none'}.</p>")
    sections.append(sec)

    # ----------------------------------------------------------- confounders
    conf_specs = config.get("confounders", [])
    if conf_specs:
        sec = "<h2>3. Confounders: is it the disease or the treatment?</h2>"
        summary["confounders"] = []
        for spec in conf_specs:
            c = by_name[spec["cohort"]]
            label = spec.get("label", spec["column"])
            eff = treatment_effects(c, spec["column"], spec["positive_values"], model_kw["min_prevalence"])
            eff.to_csv(out / "tables" / f"confounder_{_slug(label)}_{_slug(c.name)}.tsv", sep="\t")
            reduced = drop_exposed_cases(c, spec["column"], spec["positive_values"])
            cv_red = within_cohort_cv(reduced, explain_model, n_splits, n_repeats, seed, **model_kw)
            sh_red = oof_shap(reduced, explain_model, n_splits, seed, top_k, **model_kw)
            imp_red = importance_table(sh_red["shap"], reduced.X, sh_red["stability"])
            top_full = list(imp[c.name].index[:top_k])
            top_red = set(imp_red.index[:top_k])
            affected = [t for t in top_full if t in eff.index and eff.loc[t, "p"] < 0.05]
            n_exposed = c.n - reduced.n
            res = {"cohort": c.name, "confounder": label, "n_exposed_cases": n_exposed,
                   "auc_all": cv_all[explain_model][c.name]["auc_oof"], "auc_without_exposed": cv_red["auc_oof"],
                   "ci_without_exposed": [cv_red["ci_low"], cv_red["ci_high"]],
                   "top_taxa_kept_after_exclusion": len(set(top_full) & top_red) / len(top_full),
                   "top_taxa_associated_with_confounder": affected}
            summary["confounders"].append(res)
            b64 = rp.plot_confounder(da[c.name], eff, top_full, c.name, label,
                                     out / "figures" / f"confounder_{_slug(label)}.png")
            sec += (f"<p><b>{c.name}, {label}</b>: {n_exposed} exposed cases. AUROC {res['auc_all']:.2f} with all "
                    f"samples vs {res['auc_without_exposed']:.2f} after removing exposed cases; "
                    f"{res['top_taxa_kept_after_exclusion']:.0%} of the top {top_k} taxa stay in the top {top_k}. "
                    f"Top taxa that also differ with {label} (p &lt; 0.05): {', '.join(affected) or 'none'}.</p>")
            sec += rp.img(b64, "confounder")
        sections.append(sec)

    # ------------------------------------------------------- added value
    cov_specs = config.get("covariates", [])
    if cov_specs:
        rows = []
        for spec in cov_specs:
            c = by_name[spec["cohort"]]
            cols = [col for col in spec["columns"] if col in c.meta.columns]
            if not cols:
                continue
            r = added_value(c.y, cv_all[explain_model][c.name]["oof_prob"], c.meta[cols], seed)
            rows.append({"cohort": c.name, **r})
        if rows:
            summary["added_value"] = rows
            sec = ("<h2>4. Does the microbiome add information beyond clinical covariates?</h2>"
                   "<p class='note'>Likelihood-ratio test of covariates + out-of-fold microbiome score vs covariates "
                   "alone.</p>")
            sec += rp.img(rp.plot_added_value(rows, out / "figures" / "added_value.png"), "added value")
            sec += rp.table_html(pd.DataFrame(rows).drop(columns=["covariates"]), index=False)
            sections.append(sec)

    sections.append("<h2>Methods</h2><p class='note'>Relative abundances collapsed to species; taxa kept if present "
                    f"in ≥{model_kw['min_prevalence']:.0%} of training samples; log10 transform with a pseudocount "
                    "learned on training data. Models: L1-logistic regression (penalty tuned by inner CV) and random "
                    "forest (500 trees). All preprocessing is refitted inside every fold. AUROC CIs: stratified "
                    "bootstrap (1,000 resamples). Differential abundance: Wilcoxon rank-sum, generalised fold change, "
                    "Benjamini-Hochberg FDR.</p>")
    rp.write_html(out / "report.html", summary["title"], sections)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    return summary


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").lower()


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
