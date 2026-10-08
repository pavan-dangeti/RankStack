#!/usr/bin/env python
"""Fail if a figure in the documentation no longer matches its committed result file.

Every number in README.md and docs/design.md must come from a committed file under `results/`, with
the file named next to it. For each table row of the form

    | Label | 43,365,424 | `results/dataset_x.json#/totals/interactions` |

this script resolves the JSON pointer and compares it with the number written in the row. It is the
mechanism behind the README's claim that every published figure is reproducible, and it fails CI
when a re-run changes a number that has not been re-stated in the documentation.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
DOCUMENTS = (REPO_ROOT / "README.md", REPO_ROOT / "docs" / "design.md")

#: `results/<name>.json#/a/b` as it appears inside a Markdown code span.
REFERENCE = re.compile(
    r"results/(?P<name>[a-z0-9][a-z0-9._-]*)\.json#(?P<pointer>/[A-Za-z0-9_/.\[\]-]*)"
)
#: A number as written in a table cell: optional thousands separators, optional decimals.
NUMBER = re.compile(r"^[0-9][0-9,_]*(?:\.[0-9]+)?$")

TOLERANCE = 1e-9


def resolve(result_path: Path, pointer: str) -> Any:
    """Read a value from a result file, following a `/`-separated JSON pointer."""
    payload: Any = json.loads(result_path.read_text())
    for key in pointer.strip("/").split("/"):
        if isinstance(payload, list):
            payload = payload[int(key)]
            continue
        if not isinstance(payload, dict) or key not in payload:
            msg = f"{result_path.name}#{pointer} is not present"
            raise KeyError(msg)
        payload = payload[key]
    return payload


def stated_figure(row: str) -> str | None:
    """Return the numeric cell of a table row, if the row contains one."""
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    return next((cell for cell in cells if NUMBER.match(cell.strip("`* "))), None)


def check_row(document: Path, row: str) -> str | None:
    """Return a problem message when a table row's figure disagrees with its result file."""
    match = REFERENCE.search(row)
    stated = stated_figure(row)
    if match is None or stated is None:
        return None

    result_path = RESULTS_DIR / f"{match.group('name')}.json"
    if not result_path.exists():
        return f"{document.name}: {result_path.name} is referenced but not committed"
    try:
        expected = resolve(result_path, match.group("pointer"))
    except (KeyError, ValueError, IndexError) as exc:
        return f"{document.name}: {exc}"
    if not isinstance(expected, int | float):
        return None

    written = float(stated.strip("`* ").replace(",", ""))
    if abs(written - float(expected)) > max(TOLERANCE, abs(float(expected)) * TOLERANCE):
        return (
            f"{document.name}: figure {written:g} does not match "
            f"{result_path.name}#{match.group('pointer')} = {expected:g}"
        )
    return None


def verify_document(document: Path) -> tuple[list[str], int]:
    """Return the drift problems and the number of verified figures for one document."""
    if not document.exists():
        return [], 0
    problems: list[str] = []
    checked = 0
    for line in document.read_text().splitlines():
        if not line.lstrip().startswith("|"):
            continue
        problem = check_row(document, line)
        if problem is None:
            checked += 1 if REFERENCE.search(line) else 0
        else:
            problems.append(problem)
    return problems, checked


def main() -> int:
    """Check every referenced figure and report drift."""
    if not RESULTS_DIR.exists():
        print("results/ does not exist yet; nothing to verify")
        return 0

    problems: list[str] = []
    checked = 0
    for document in DOCUMENTS:
        document_problems, document_checked = verify_document(document)
        problems.extend(document_problems)
        checked += document_checked

    if problems:
        print("documentation drift detected:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"documentation matches {checked} committed result figure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
