"""Figures and a self-contained HTML report."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

# categorical slots (cohorts), diverging poles (case-up red / control-up blue), neutral inks
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
RED, BLUE, MID = "#d03b3b", "#1c5cab", "#f0efec"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
AUC_CMAP = LinearSegmentedColormap.from_list("auc", [RED, MID, BLUE])


def _style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    if grid_axis == "both":
        ax.grid(True, color=GRID, linewidth=0.8)
    elif grid_axis:
        getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_transfer(mat: pd.DataFrame, title: str, path: Path) -> str:
    n = len(mat)
    fig, ax = plt.subplots(figsize=(1.6 * n + 2.2, 1.3 * n + 1.4))
    ax.imshow(mat.to_numpy(float), cmap=AUC_CMAP, norm=TwoSlopeNorm(vmin=0.2, vcenter=0.5, vmax=1.0))
    for i in range(n):
        for j in range(n):
            v = mat.iat[i, j]
            ax.text(j, i, f"{v:.2f}" + ("\n(CV)" if i == j else ""), ha="center", va="center", fontsize=10,
                    color="white" if abs(v - 0.5) > 0.3 else INK)
    ax.set_xticks(range(n), mat.columns)
    ax.set_yticks(range(n), mat.index)
    ax.set_xlabel("tested on")
    ax.set_ylabel("trained on")
    ax.set_title(title, fontsize=11, loc="left")
    _style(ax, grid_axis=None)
    for s in ax.spines.values():
        s.set_visible(False)
    return _save(fig, path)


def plot_roc(curves: dict, title: str, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle="--")
    for (name, (fpr, tpr, auc)), c in zip(curves.items(), CAT):
        ax.plot(fpr, tpr, color=c, linewidth=2, label=f"{name} (AUROC {auc:.2f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax, "both")
    return _save(fig, path)


def plot_importance(tables: dict[str, pd.DataFrame], top: int, path: Path) -> str:
    k = len(tables)
    fig, axes = plt.subplots(1, k, figsize=(5.2 * k, 0.32 * top + 1.6), squeeze=False)
    for ax, (name, t) in zip(axes[0], tables.items()):
        t = t.head(top).iloc[::-1]
        colors = [RED if d == "higher in case" else BLUE for d in t["direction"]]
        ax.barh(range(len(t)), t["mean_abs_shap"], color=colors, height=0.7)
        ax.set_yticks(range(len(t)), [_short(x) for x in t.index], fontsize=8.5, style="italic")
        ax.set_xlabel("mean |SHAP| (out-of-fold)")
        ax.set_title(name, fontsize=11, loc="left")
        _style(ax, "x")
    from matplotlib.patches import Patch
    axes[0][0].legend(handles=[Patch(color=RED, label="higher abundance → case"),
                               Patch(color=BLUE, label="higher abundance → control")],
                      frameon=False, fontsize=8.5, loc="lower right")
    return _save(fig, path)


def plot_importance_agreement(ta: pd.DataFrame, tb: pd.DataFrame, a: str, b: str, path: Path, label_top: int = 8) -> str:
    shared = ta.index.intersection(tb.index)
    x, y = ta.loc[shared, "mean_abs_shap"], tb.loc[shared, "mean_abs_shap"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(x, y, s=14, color=CAT[0], alpha=0.55, linewidths=0)
    score = (x.rank(ascending=False) + y.rank(ascending=False))
    for f in score.nsmallest(label_top).index:
        ax.annotate(_short(f), (x[f], y[f]), fontsize=7.5, color=INK, xytext=(3, 3), textcoords="offset points",
                    style="italic")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"importance in {a} (mean |SHAP|)")
    ax.set_ylabel(f"importance in {b} (mean |SHAP|)")
    ax.set_title("Do the cohorts rely on the same taxa?", fontsize=11, loc="left")
    _style(ax, "both")
    return _save(fig, path)


def plot_confounder(disease: pd.DataFrame, drug: pd.DataFrame, top_taxa, cohort: str, label: str, path: Path,
                    n_label: int = 6) -> str:
    """x: disease effect, y: treatment effect within cases. Model taxa far from y = 0 may reflect the treatment."""
    shared = disease.index.intersection(drug.index)
    taxa = [t for t in top_taxa if t in shared]
    drug_top = [t for t in drug.loc[shared].sort_values("p").index[:3] if t not in taxa]
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.scatter(disease.loc[shared, "gfc"], drug.loc[shared, "gfc"], s=10, color=GRID, linewidths=0,
               label="all tested taxa")
    ax.scatter(disease.loc[taxa, "gfc"], drug.loc[taxa, "gfc"], s=28, color=CAT[1], linewidths=0,
               label="top model taxa (SHAP)")
    ax.scatter(disease.loc[drug_top, "gfc"], drug.loc[drug_top, "gfc"], s=28, color=CAT[2], linewidths=0,
               label=f"strongest {label} effects")
    for i, t in enumerate(taxa[:n_label] + drug_top):
        dy = 6 if i % 2 == 0 else -11
        ax.annotate(_short(t, 30), (disease.loc[t, "gfc"], drug.loc[t, "gfc"]), fontsize=7.5, style="italic",
                    xytext=(4, dy), textcoords="offset points", color=INK)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.axvline(0, color=INK2, linewidth=0.6)
    ax.set_xlabel("disease effect: case vs control (gFC)")
    ax.set_ylabel(f"{label} effect within cases (gFC)")
    ax.set_title(f"{cohort}: disease signal vs {label} signal", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax, "both")
    return _save(fig, path)


def plot_added_value(rows: list[dict], path: Path) -> str:
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    keys = [("auc_covariates", "clinical covariates"), ("auc_microbiome", "microbiome score"),
            ("auc_combined", "combined")]
    width = 0.8 / len(keys)
    for i, ((k, lab), c) in enumerate(zip(keys, CAT)):
        vals = [r[k] for r in rows]
        bars = ax.bar(np.arange(len(rows)) + (i - 1) * width, vals, width * 0.92, color=c, label=lab)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom", fontsize=8,
                        xytext=(0, 2), textcoords="offset points", color=INK)
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(rows)), [r["cohort"] for r in rows])
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("AUROC (cross-validated)")
    ax.set_title("Does the microbiome add to clinical covariates?", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    _style(ax)
    return _save(fig, path)


def _short(name: str, n: int = 38) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


# ----------------------------------------------------------------- HTML
CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#3a3936}
  img{background:#fcfcfb;border-radius:6px}}
body{background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;
  max-width:960px;margin:0 auto;padding:32px 16px}
h1{font-size:24px;margin:0 0 6px} h2{font-size:18px;margin:36px 0 8px;border-bottom:1px solid var(--rule);padding-bottom:4px}
p{margin:6px 0} .note{color:var(--ink2)}
table{border-collapse:collapse;font-size:13px;margin:8px 0;display:block;overflow-x:auto}
th,td{text-align:left;padding:5px 10px;border-bottom:1px solid var(--rule);white-space:nowrap}
th{color:var(--ink2);font-weight:600} img{max-width:100%;height:auto;margin:8px 0}
.kpi{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}
.kpi div{border:1px solid var(--rule);border-radius:8px;padding:10px 14px;min-width:150px}
.kpi b{display:block;font-size:22px} .kpi span{color:var(--ink2);font-size:12px}
"""


def table_html(df: pd.DataFrame, max_rows: int = 25, index: bool = True) -> str:
    df = df.head(max_rows)
    if index:
        df = df.reset_index()
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                txt = f"{v:.3g}" if abs(v) < 1e-3 and v != 0 else f"{v:.3f}"
            elif isinstance(v, (list, tuple)):
                txt = ", ".join(map(str, v))
            else:
                txt = str(v)
            cells.append(f"<td>{html.escape(txt)}</td>")
        body += "<tr>" + "".join(cells) + "</tr>"
    return f"<table><tr>{head}</tr>{body}</table>"


def img(b64: str, alt: str) -> str:
    return f"<img alt='{html.escape(alt)}' src='data:image/png;base64,{b64}'>"


def write_html(path: Path, title: str, sections: list[str]) -> None:
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
        f"initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>{''.join(sections)}</body></html>"
    )
