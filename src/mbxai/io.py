"""
Loading and harmonising microbiome abundance tables from several cohorts.

Every loader returns a :class:`Cohort`: a samples x taxa relative-abundance
table, binary labels and sample metadata. Taxa are collapsed to one rank
(species by default) so that cohorts profiled with the same reference can be
combined feature by feature.

Supported inputs
----------------
* ``mlrepo``   : datasets from the Microbiome Learning Repo (Vangay et al. 2019),
                 taxa x samples counts with ';'-separated lineages
* ``table``    : any taxa x samples (or samples x taxa) table, e.g. a merged
                 MetaPhlAn profile ('|'-separated lineages) or the export of
                 scripts/export_curatedMetagenomicData.R
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RANKS = ["k", "p", "c", "o", "f", "g", "s", "t"]


@dataclass
class Cohort:
    name: str
    X: pd.DataFrame                 # samples x taxa, relative abundance (rows sum to 1)
    y: pd.Series                    # 0 = control, 1 = case
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    case_label: str = "case"
    control_label: str = "control"

    def __post_init__(self):
        common = self.X.index.intersection(self.y.index)
        if len(common) == 0:
            raise ValueError(f"{self.name}: no sample IDs shared between abundance table and labels")
        self.X = self.X.loc[common]
        self.y = self.y.loc[common].astype(int)
        self.meta = self.meta.reindex(common)

    @property
    def n(self) -> int:
        return len(self.y)

    def summary(self) -> dict:
        return {"cohort": self.name, "n": self.n, "cases": int(self.y.sum()),
                "controls": int((1 - self.y).sum()), "taxa": int(self.X.shape[1])}


# ----------------------------------------------------------------- taxonomy
def parse_lineage(name: str) -> dict:
    """'k__Bacteria;p__...;s__Escherichia_coli' (or '|' separated) -> {'k': 'Bacteria', ..., 's': 'Escherichia_coli'}."""
    out = {}
    for part in re.split(r"[;|]", str(name)):
        m = re.match(r"^\s*([kpcofgst])__(.*)$", part)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def taxon_label(lineage: dict, rank: str = "s") -> str | None:
    """Readable label at `rank`; unnamed taxa fall back to the nearest named parent."""
    value = lineage.get(rank, "")
    if value:
        return value.replace("_", " ").replace("[", "").replace("]", "")
    for parent in reversed(RANKS[: RANKS.index(rank)]):
        if lineage.get(parent):
            return f"unclassified {lineage[parent].replace('_', ' ')}"
    return None


def collapse_to_rank(table: pd.DataFrame, rank: str = "s") -> pd.DataFrame:
    """
    Sum a taxa x samples table to one rank.

    For MetaPhlAn-style tables that list every rank (k, k|p, ..., k|..|s), only
    rows that end exactly at `rank` are kept, to avoid double counting. For
    tables at a finer rank (e.g. strains), rows are summed per `rank` label.
    """
    lineages = [parse_lineage(i) for i in table.index]
    deepest = [max((RANKS.index(r) for r in lin), default=-1) for lin in lineages]
    target = RANKS.index(rank)
    if any(d == target for d in deepest) and any(d < target for d in deepest):
        keep = [d == target for d in deepest]   # hierarchical table: take the rank's own rows
        table, lineages = table.loc[keep], [lin for lin, k in zip(lineages, keep) if k]
    labels = [taxon_label(lin, rank) for lin in lineages]
    if all(lab is None for lab in labels):     # plain names (e.g. 'Escherichia coli')
        labels = [str(i).replace("_", " ") for i in table.index]
    table = table.copy()
    table.index = labels
    table = table[table.index.notna()]
    return table.groupby(level=0).sum()


def to_relative(X: pd.DataFrame) -> pd.DataFrame:
    """Rows (samples) scaled to sum to 1; empty samples are dropped."""
    totals = X.sum(axis=1)
    X = X.loc[totals > 0]
    return X.div(X.sum(axis=1), axis=0)


# ------------------------------------------------------------------ loaders
def _labels(series: pd.Series, case_values, control_values, name: str) -> pd.Series:
    s = series.astype(str).str.strip()
    case_values = [str(v) for v in np.atleast_1d(case_values)]
    control_values = [str(v) for v in np.atleast_1d(control_values)]
    y = pd.Series(np.nan, index=s.index)
    y[s.isin(case_values)] = 1
    y[s.isin(control_values)] = 0
    y = y.dropna()
    if y.nunique() < 2:
        raise ValueError(f"{name}: labels {case_values} / {control_values} did not give two classes "
                         f"(values found: {sorted(s.unique())[:10]})")
    return y.astype(int)


def load_mlrepo(name: str, directory: str | Path, task: str, case_values, control_values, rank: str = "s") -> Cohort:
    """Load an MLRepo dataset folder (taxatable.txt, task-*.txt, mapping-orig.txt)."""
    d = Path(directory)
    counts = pd.read_csv(d / "taxatable.txt", sep="\t", index_col=0)
    X = to_relative(collapse_to_rank(counts, rank).T)
    task_df = pd.read_csv(d / task, sep="\t", index_col=0, dtype=str)
    y = _labels(task_df.iloc[:, 0], case_values, control_values, name)
    meta = pd.DataFrame(index=X.index)
    if (d / "mapping-orig.txt").exists():
        meta = pd.read_csv(d / "mapping-orig.txt", sep="\t", index_col=0, dtype=str)
        meta.columns = [c.strip() for c in meta.columns]
    return Cohort(name, X, y, meta, case_label=str(np.atleast_1d(case_values)[0]),
                  control_label=str(np.atleast_1d(control_values)[0]))


def load_table(name: str, abundance: str | Path, metadata: str | Path, label_column: str, case_values,
               control_values, sample_column: str | None = None, samples_as_rows: bool = False,
               rank: str = "s", subset: dict | None = None) -> Cohort:
    """
    Load a generic abundance table + metadata table.

    `subset` restricts the metadata first, e.g. {"study_name": ["YachidaS_2019"]}.
    """
    tab = pd.read_csv(abundance, sep="\t", index_col=0)
    if samples_as_rows:
        tab = tab.T
    X = to_relative(collapse_to_rank(tab, rank).T)
    meta = pd.read_csv(metadata, sep="\t", dtype=str)
    meta = meta.set_index(sample_column or meta.columns[0])
    for col, values in (subset or {}).items():
        meta = meta[meta[col].astype(str).isin([str(v) for v in np.atleast_1d(values)])]
    y = _labels(meta[label_column], case_values, control_values, name)
    return Cohort(name, X, y, meta, case_label=str(np.atleast_1d(case_values)[0]),
                  control_label=str(np.atleast_1d(control_values)[0]))


def load_cohort(spec: dict, base_dir: str | Path = ".") -> Cohort:
    """Build a Cohort from one entry of the YAML config."""
    spec = dict(spec)
    kind = spec.pop("type")
    base = Path(base_dir)
    for key in ("directory", "abundance", "metadata"):
        if key in spec:
            spec[key] = base / spec[key]
    if kind == "mlrepo":
        return load_mlrepo(**spec)
    if kind == "table":
        return load_table(**spec)
    raise ValueError(f"unknown cohort type '{kind}' (use 'mlrepo' or 'table')")


def align_features(X: pd.DataFrame, columns) -> pd.DataFrame:
    """Reorder X to `columns`; taxa absent from X are filled with 0 (not detected)."""
    return X.reindex(columns=columns, fill_value=0.0)


def shared_feature_space(cohorts: list[Cohort]) -> list[str]:
    """Union of taxa over cohorts (absent = 0 in a cohort)."""
    cols = set()
    for c in cohorts:
        cols |= set(c.X.columns)
    return sorted(cols)
