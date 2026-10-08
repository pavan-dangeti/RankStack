#!/usr/bin/env python
"""Fail if any dataset file is tracked by git.

The Amazon Reviews 2023 files must never enter the repository (docs/design.md section 4.3): the
publisher does not license redistribution. `data/` is gitignored, and this check makes the rule
independent of a developer's local ignore rules.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which("git") or "git"
MAX_ALLOWED_BYTES = 1_000_000
DATASET_SUFFIXES = (".csv.gz", ".jsonl.gz", ".parquet", ".tsv")
DATASET_DIRECTORIES = ("data/",)


def tracked_files() -> list[str]:
    """Paths tracked by git, one per line."""
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [GIT, "-C", str(REPO_ROOT), "ls-files", "-z"],
        capture_output=True,
        check=True,
        text=True,
    )
    return [entry for entry in result.stdout.split("\0") if entry]


def violations(paths: list[str]) -> list[str]:
    """Return one message per tracked file that must not be in the repository."""
    problems: list[str] = []
    for relative in paths:
        if any(relative.startswith(directory) for directory in DATASET_DIRECTORIES):
            problems.append(f"{relative}: dataset directory is tracked")
            continue
        if relative.endswith(DATASET_SUFFIXES):
            problems.append(f"{relative}: dataset file type is tracked")
            continue
        size = (REPO_ROOT / relative).stat().st_size if (REPO_ROOT / relative).exists() else 0
        if size > MAX_ALLOWED_BYTES:
            problems.append(f"{relative}: {size} bytes exceeds the {MAX_ALLOWED_BYTES} byte limit")
    return problems


def main() -> int:
    """Fail when a dataset file or an oversized file is tracked."""
    problems = violations(tracked_files())
    if not problems:
        print("no dataset files are tracked")
        return 0
    print("dataset files must not be committed:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
