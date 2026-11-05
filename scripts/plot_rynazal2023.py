"""Figures and comparison tables for the Rynazal et al. 2023 reproduction.

Reads the per-study tables written by scripts/reproduce_rynazal2023.py and makes
equivalents of the paper's Figs. 3, 5, 6, 7 and Additional file 2 Fig. S1, plus:
- comparison_with_paper.tsv : LODO AUC and explained variance, paper vs mine
- cluster_profiles.tsv      : per cluster: size, median score, F. nucleatum, stage I, top SHAP taxa

Run from the repo root, after reproduce_rynazal2023.py:
    python scripts/plot_rynazal2023.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path("../shapmat_tutorial/data/curatedCRC.csv")
RES = Path("results/rynazal2023")
FIG = RES / "figures"

# Panel order as in the paper's figures
STUDIES = ["YachidaS_2019", "ZellerG_2014", "WirbelJ_2018", "YuJ_2015", "VogtmannE_2016"]

# Values read from the titles and axis labels of the paper's Fig. 3
PAPER = {
    "YachidaS_2019": {"auc": 0.72, "pc1_var": 0.389, "pc2_var": 0.089},
    "ZellerG_2014": {"auc": 0.77, "pc1_var": 0.391, "pc2_var": 0.120},
    "WirbelJ_2018": {"auc": 0.89, "pc1_var": 0.472, "pc2_var": 0.097},
    "YuJ_2015": {"auc": 0.87, "pc1_var": 0.474, "pc2_var": 0.090},
    "VogtmannE_2016": {"auc": 0.77, "pc1_var": 0.294, "pc2_var": 0.100},
}


def load_study(study: str) -> dict:
    """Read the tables that reproduce_rynazal2023.py saved for one study."""
    pcs = pd.read_csv(RES / f"pcs_{study}.tsv", sep="\t", index_col=0)
    clusters = pd.read_csv(RES / f"clusters_{study}.tsv", sep="\t", index_col=0)["cluster"]
    elbow = pd.read_csv(RES / f"elbow_{study}.tsv", sep="\t", index_col=0)["inertia"]
    shap_correct = pd.read_csv(RES / f"shap_correct_{study}.tsv", sep="\t", index_col=0)
    return {"pcs": pcs, "clusters": clusters, "elbow": elbow, "shap": shap_correct}


def panel_grid(title: str):
    """A 2 x 3 grid with the sixth panel hidden, like the paper's figures."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()
    axes[-1].axis("off")
    fig.suptitle(title, fontsize=13)
    return fig, axes


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print("saved", FIG / name)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RES / "lodo_summary.tsv", sep="\t", index_col="test_study")
    d = pd.read_csv(DATA, index_col=0)
    tables = {s: load_study(s) for s in STUDIES}

    # --- Fig. 3: SHAP-PCA coloured by label ---
    fig, axes = panel_grid("SHAP-value PCA, correctly predicted subjects (cf. paper Fig. 3)")
    for ax, s in zip(axes, STUDIES):
        pcs = tables[s]["pcs"]
        for label, color, name in [(0, "tab:blue", "control"), (1, "tab:red", "CRC")]:
            z = pcs[pcs["label"] == label]
            ax.scatter(z["PC1"], z["PC2"], s=10, alpha=0.6, c=color, label=name)
        ax.set_title(f"{s} (AUC={summary.loc[s, 'lodo_auc']:.2f})", fontsize=10)
        ax.set_xlabel(f"PC1 ({summary.loc[s, 'pc1_var']:.1%})")
        ax.set_ylabel(f"PC2 ({summary.loc[s, 'pc2_var']:.1%})")
    axes[0].legend()
    save(fig, "fig3_shap_pca_label.png")

    # --- Fig. 5: SHAP-PCA coloured by predicted CRC score ---
    fig, axes = panel_grid("SHAP-value PCA coloured by predicted CRC score (cf. paper Fig. 5)")
    for ax, s in zip(axes, STUDIES):
        pcs = tables[s]["pcs"]
        points = ax.scatter(pcs["PC1"], pcs["PC2"], s=10, c=pcs["p_crc"], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(s, fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.colorbar(points, ax=axes[-1], fraction=0.5, label="predicted CRC score")
    save(fig, "fig5_shap_pca_score.png")

    # --- Fig. S1: elbow plots ---
    fig, axes = panel_grid("Elbow method (cf. Additional file 2, Fig. S1)")
    for ax, s in zip(axes, STUDIES):
        elbow = tables[s]["elbow"]
        ax.plot(elbow.index, elbow.values, marker="o")
        ax.axvline(4, color="grey", linestyle="--", linewidth=1)
        ax.set_title(s, fontsize=10)
        ax.set_xlabel("k")
        ax.set_ylabel("inertia")
    save(fig, "figS1_elbow.png")

    # --- Fig. 6: k-means clusters of CRC subjects ---
    fig, axes = panel_grid("k-means clusters of CRC subjects on SHAP-PCA (cf. paper Fig. 6)")
    for ax, s in zip(axes, STUDIES):
        pcs, clusters = tables[s]["pcs"], tables[s]["clusters"]
        controls = pcs[pcs["label"] == 0]
        ax.scatter(controls["PC1"], controls["PC2"], s=8, c="lightgrey", label="control")
        for c in sorted(clusters.unique()):
            z = pcs.loc[clusters[clusters == c].index]
            ax.scatter(z["PC1"], z["PC2"], s=12, label=f"cluster {c}")
        ax.set_title(s, fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    axes[0].legend(fontsize=7)
    save(fig, "fig6_clusters.png")

    # --- Fig. 7: predicted CRC score per cluster ---
    fig, axes = panel_grid("Predicted CRC score per cluster (cf. paper Fig. 7)")
    for ax, s in zip(axes, STUDIES):
        pcs, clusters = tables[s]["pcs"], tables[s]["clusters"]
        groups = sorted(clusters.unique())
        values = [pcs.loc[clusters[clusters == c].index, "p_crc"] for c in groups]
        ax.boxplot(values)
        ax.set_xticks(range(1, len(groups) + 1), [str(c) for c in groups])
        ax.set_title(s, fontsize=10)
        ax.set_xlabel("cluster")
        ax.set_ylabel("predicted CRC score")
    save(fig, "fig7_score_by_cluster.png")

    # --- Table: paper vs mine ---
    rows = []
    for s in STUDIES:
        rows.append({
            "study": s,
            "auc_paper": PAPER[s]["auc"], "auc_mine": summary.loc[s, "lodo_auc"],
            "pc1_var_paper": PAPER[s]["pc1_var"], "pc1_var_mine": summary.loc[s, "pc1_var"],
            "pc2_var_paper": PAPER[s]["pc2_var"], "pc2_var_mine": summary.loc[s, "pc2_var"],
            "n_clusters_paper": 4, "n_clusters_mine": tables[s]["clusters"].nunique(),
        })
    comparison = pd.DataFrame(rows).set_index("study")
    comparison.to_csv(RES / "comparison_with_paper.tsv", sep="\t")
    print(comparison.round(3).to_string())

    # --- Table: cluster profiles ---
    rows = []
    for s in STUDIES:
        pcs, clusters, shap_df = tables[s]["pcs"], tables[s]["clusters"], tables[s]["shap"]
        for c in sorted(clusters.unique()):
            ids = clusters[clusters == c].index
            fn = d.loc[ids, "Fusobacterium_nucleatum"]
            stage = d.loc[ids, "ajcc_stage"].dropna()  # VogtmannE_2016 has no stage information
            top = shap_df.loc[ids].mean().sort_values(ascending=False).index[:3]
            rows.append({
                "study": s, "cluster": c, "n": len(ids),
                "median_score": round(pcs.loc[ids, "p_crc"].median(), 2),
                "fn_detected": round((fn > 0).mean(), 2),
                "fn_mean_pct": round(fn.mean() * 100, 3),
                "stage_I_share": round((stage == "i").mean(), 2) if len(stage) else None,
                "top_shap": ", ".join(top),
            })
    profiles = pd.DataFrame(rows)
    profiles.to_csv(RES / "cluster_profiles.tsv", sep="\t", index=False)
    print(profiles.drop(columns="top_shap").to_string(index=False))


if __name__ == "__main__":
    main()
