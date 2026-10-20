"""Command-line interface: `mbxai run`, `mbxai fetch-mlrepo`, `mbxai simulate`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
import warnings
from pathlib import Path

import yaml

from . import __version__

# Microbiome Learning Repo (Vangay et al. 2019, GigaScience; MIT licence), pinned commit
MLREPO_COMMIT = "4bbfee98c911d861c40664e93fa1dee28bb415da"
MLREPO_RAW = f"https://raw.githubusercontent.com/knights-lab/MLRepo/{MLREPO_COMMIT}/datasets"
MLREPO_FILES = {
    "karlsson": ["taxatable.txt", "mapping-orig.txt", "task-normal-diabetes.txt"],
    "qin2012": ["taxatable.txt", "mapping-orig.txt", "task-healthy-diabetes.txt"],
    "qin2014": ["taxatable.txt", "mapping-orig.txt", "task-healthy-cirrhosis.txt"],
}


def cmd_run(a):
    from .analysis import run

    config = yaml.safe_load(Path(a.config).read_text())
    summary = run(config, a.out, a.data_root)
    print(json.dumps({"report": str(Path(a.out) / "report.html"),
                      "within_cohort_auc": {m: {k: round(v["auc_oof"], 3) for k, v in r["within_cohort"].items()}
                                            for m, r in summary["models"].items()}}, indent=2))


def cmd_fetch(a):
    out = Path(a.out)
    for ds in a.datasets:
        if ds not in MLREPO_FILES:
            sys.exit(f"unknown dataset {ds}; choose from {sorted(MLREPO_FILES)}")
        (out / ds).mkdir(parents=True, exist_ok=True)
        for f in MLREPO_FILES[ds]:
            dest = out / ds / f
            if dest.exists() and dest.stat().st_size > 0:
                continue
            url = f"{MLREPO_RAW}/{ds}/{f}"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)
    print(f"MLRepo data (commit {MLREPO_COMMIT[:7]}) in {out}")


def cmd_simulate(a):
    from .simulate import simulate

    truth = simulate(a.out, seed=a.seed)
    print(json.dumps(truth, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="mbxai", description="Explainable, cross-cohort microbiome biomarkers")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(required=True)

    r = sub.add_parser("run", help="run an analysis from a YAML config")
    r.add_argument("config")
    r.add_argument("--out", default="results")
    r.add_argument("--data-root", default=".", help="paths in the config are relative to this directory")
    r.set_defaults(func=cmd_run)

    f = sub.add_parser("fetch-mlrepo", help="download public datasets from the Microbiome Learning Repo")
    f.add_argument("--out", default="data/mlrepo")
    f.add_argument("datasets", nargs="*", default=list(MLREPO_FILES))
    f.set_defaults(func=cmd_fetch)

    s = sub.add_parser("simulate", help="write a small simulated multi-cohort dataset")
    s.add_argument("--out", default="data/simulated")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_simulate)

    a = p.parse_args(argv)
    # scikit-learn deprecation notices and saga convergence chatter are not actionable for users
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*converge.*")
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(asctime)s %(message)s")
    a.func(a)


if __name__ == "__main__":
    main()
