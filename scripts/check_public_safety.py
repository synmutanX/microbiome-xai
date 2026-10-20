#!/usr/bin/env python3
"""
Check that the repository is safe to make public.

Scans every file git would publish (tracked + staged + untracked-but-not-ignored)
and fails if it finds:

* data files that should never be committed (raw reads, alignments, variant
  calls, R/AnnData objects, spreadsheets) or anything under data/ or results/
* large files (> 2 MB by default)
* credentials (GitHub/AWS/Slack/Anthropic/OpenAI tokens, private keys)
* e-mail addresses (except no-reply addresses)
* any term listed in `.public-safety-denylist` - a local, git-ignored file
  where you put names that must never appear publicly (e.g. your lab, internal
  project codes, collaborators, unpublished dataset names), one per line

usage: python scripts/check_public_safety.py [--max-mb 2]
Install as a pre-commit hook:
  printf '#!/bin/sh\\npython scripts/check_public_safety.py\\n' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_EXT = {".fastq", ".fq", ".fasta.gz", ".fq.gz", ".fastq.gz", ".bam", ".sam", ".cram", ".bai", ".vcf",
                 ".vcf.gz", ".bcf", ".rds", ".rdata", ".rda", ".h5ad", ".loom", ".xlsx", ".xls", ".sqlite", ".db",
                 ".pkl", ".pickle", ".joblib", ".biom"}
FORBIDDEN_DIRS = ("data/", "results/")
SECRETS = {
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}",
    "AWS access key": r"AKIA[0-9A-Z]{16}",
    "Slack token": r"xox[abpr]-[A-Za-z0-9-]{10,}",
    "Anthropic/OpenAI key": r"sk-(?:ant-)?[A-Za-z0-9_-]{20,}",
    "Private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "Google API key": r"AIza[0-9A-Za-z_-]{35}",
}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EMAIL_OK = re.compile(r"(noreply|no-reply|example\.(com|org))", re.I)
DENYLIST_FILE = ".public-safety-denylist"
SKIP_CONTENT_SCAN = {"scripts/check_public_safety.py"}  # contains the patterns themselves


def publishable_files(root: Path) -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root,
                             capture_output=True, text=True, check=True).stdout
        return sorted(set(filter(None, out.splitlines())))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".git/" not in str(p))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-mb", type=float, default=2.0)
    p.add_argument("--root", default=".")
    a = p.parse_args()
    root = Path(a.root).resolve()

    deny = []
    if (root / DENYLIST_FILE).exists():
        deny = [t.strip() for t in (root / DENYLIST_FILE).read_text().splitlines() if t.strip() and not t.startswith("#")]
    deny_re = re.compile("|".join(re.escape(t) for t in deny), re.I) if deny else None

    problems = []
    files = publishable_files(root)
    if DENYLIST_FILE in files:
        problems.append(f"{DENYLIST_FILE}: must stay private - add it to .gitignore")
    for rel in files:
        path = root / rel
        if not path.is_file():
            continue
        low = rel.lower()
        if any(low.endswith(ext) for ext in FORBIDDEN_EXT):
            problems.append(f"{rel}: data/object file type that should not be published")
        if low.startswith(FORBIDDEN_DIRS):
            problems.append(f"{rel}: files under data/ or results/ should not be committed")
        size_mb = path.stat().st_size / 1e6
        if size_mb > a.max_mb:
            problems.append(f"{rel}: {size_mb:.1f} MB (> {a.max_mb} MB)")
        if rel in SKIP_CONTENT_SCAN or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, pat in SECRETS.items():
                if re.search(pat, line):
                    problems.append(f"{rel}:{lineno}: possible {name}")
            for m in EMAIL.finditer(line):
                if not EMAIL_OK.search(m.group()):
                    problems.append(f"{rel}:{lineno}: e-mail address '{m.group()}'")
            if deny_re and deny_re.search(line):
                problems.append(f"{rel}:{lineno}: contains a denylisted term")

    if problems:
        print("Public-safety check FAILED:")
        for prob in problems:
            print("  -", prob)
        return 1
    print(f"Public-safety check passed: {len(files)} files scanned"
          + (f", {len(deny)} denylisted terms" if deny else " (no local denylist found)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
