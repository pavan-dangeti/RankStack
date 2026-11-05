#!/usr/bin/env python
"""Fail if a figure in the documentation no longer matches its committed result file.

Every number in README.md, CHANGELOG.md, docs/design.md, docs/runbook.md, the ADRs, the demo page
and the serving Dockerfile's comments must come from a committed file under `results/`, with the
file named next to it. These forms are checked.

A table row, where the first figure cell is compared with the pointer in the same row:

    | Label | 43,365,424 | `results/dataset_x.json#/totals/interactions` |
    | Warm user | 25.1 ms | ... | results/load_test_x.json#/latency/0/p50_ms |

A figure in prose, written directly before its pointer in parentheses:

    the re-ranker scores -0.0148 (`results/two_stage_x.json#/cohorts/.../difference`) against

A table whose header cells name sub-paths, where every cell under such a header is checked against
the row's pointer extended by that sub-path:

    | w (`/weight`) | HR@20 (`/hr@20/point`) | Figure |
    | 0.25 | 0.03250 | results/sweep_x.json#/by_weight/0.25 |

A table row that cites its figures in the cells, read exactly like prose:

    | Serving latency, warm p50 / p99 | 23.9 ms (`results/x.json#/p50`) / 34.6 ms (`...`) |

A figure on the demo page, held by the element whose `data-figure` attribute names its pointer:

    <span data-figure="results/sweep_x.json#/selected_weight">0.75</span>

A prose pointer without a figure directly before it is an error rather than being skipped, so a
sentence cannot cite a result file while stating a number the check never reads. A figure must equal
its result rounded to the precision written: `25.1 ms` matches 25.076, `25.0 ms` does not.

It is the mechanism behind the README's claim that every published figure is reproducible, and it
fails CI when a re-run changes a number that has not been re-stated in the documentation.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from functools import cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
#: Its released sections are checked against the result files at their own tag.
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
DOCUMENTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "CHANGELOG.md",
    REPO_ROOT / "docs" / "design.md",
    REPO_ROOT / "docs" / "runbook.md",
    *sorted((REPO_ROOT / "docs" / "adr").glob("*.md")),
    REPO_ROOT / "src" / "rankstack" / "serving" / "demo.html",
    REPO_ROOT / "infra" / "docker" / "Dockerfile.serve",
)

_POINTER = r"results/(?P<name>[a-z0-9][a-z0-9._-]*)\.json#(?P<pointer>/[A-Za-z0-9_/.\[\]@-]*)"
_FIGURE = r"(?P<figure>-?[0-9][0-9,_]*(?:\.[0-9]+)?)"
_UNIT = r"(?:\s?(?:ms|s|MiB|MB|GB|req/s|%))?"

#: `results/<name>.json#/a/b` as it appears inside a Markdown code span or table cell.
REFERENCE = re.compile(_POINTER)
#: A figure as written in a table cell: optional sign, thousands separators, decimals and a unit.
NUMBER = re.compile(rf"^{_FIGURE}{_UNIT}$")
#: A figure in prose, immediately followed by its pointer as a parenthesised code span.
PROSE_FIGURE = re.compile(rf"{_FIGURE}{_UNIT} \(`{_POINTER}`\)")
#: A figure on the demo page: an element whose `data-figure` names its pointer and holds the figure.
HTML_FIGURE = re.compile(rf'data-figure="{_POINTER}"[^>]*>{_FIGURE}<')
#: A sub-path declared in a table header cell; each row's pointer is then the base it extends.
COLUMN_PATH = re.compile(r"`(?P<path>/[A-Za-z0-9_/.\[\]@-]*)`")
#: The `|---|---|` line under a table header.
SEPARATOR = re.compile(r"^\s*\|(\s*:?-+:?\s*\|)+\s*$")

TOLERANCE = 1e-9


@cache
def load_result(name: str, revision: str | None) -> Any:
    """Read `results/<name>.json` from the working tree, or as committed at a git revision."""
    path = RESULTS_DIR / f"{name}.json"
    if revision is None:
        if not path.exists():
            msg = f"{path.name} is referenced but not committed"
            raise FileNotFoundError(msg)
        return json.loads(path.read_text())
    relative = path.relative_to(REPO_ROOT).as_posix()
    shown = subprocess.run(  # noqa: S603 - fixed git argv; the revision is a tag name from the file
        ["git", "-C", str(REPO_ROOT), "show", f"{revision}:{relative}"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if shown.returncode != 0:
        msg = f"{path.name} is not committed at {revision}"
        raise FileNotFoundError(msg)
    return json.loads(shown.stdout)


def tag_exists(name: str) -> bool:
    """Return True when `name` is a tag in this repository."""
    found = subprocess.run(  # noqa: S603 - fixed git argv
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", "--quiet", f"refs/tags/{name}"],  # noqa: S607
        capture_output=True,
        check=False,
    )
    return found.returncode == 0


def resolve(payload: Any, label: str, pointer: str) -> Any:
    """Follow a `/`-separated JSON pointer into a result file's payload."""
    for key in pointer.strip("/").split("/"):
        if isinstance(payload, list):
            payload = payload[int(key)]
            continue
        if not isinstance(payload, dict) or key not in payload:
            msg = f"{label}#{pointer} is not present"
            raise KeyError(msg)
        payload = payload[key]
    return payload


def matches(written: str, expected: float) -> bool:
    """Return True when `written` is `expected` rounded to the precision it is written with."""
    digits = written.replace(",", "").replace("_", "")
    decimals = len(digits.partition(".")[2])
    allowed = 0.5 * 10.0**-decimals + TOLERANCE * max(1.0, abs(expected))
    return abs(float(digits) - expected) <= allowed


def stated_figure(row: str) -> str | None:
    """Return the first figure cell of a table row, without its unit, if the row has one."""
    for cell in row.strip().strip("|").split("|"):
        match = NUMBER.match(cell.strip().strip("`* "))
        if match:
            return match.group("figure")
    return None


def _compare(
    document: Path, written: str, name: str, pointer: str, revision: str | None = None
) -> str | None:
    """Return a problem message when `written` disagrees with the value at `name#pointer`."""
    label = f"{name}.json" if revision is None else f"{name}.json at {revision}"
    try:
        expected = resolve(load_result(name, revision), label, pointer)
    except (FileNotFoundError, KeyError, ValueError, IndexError) as exc:
        return f"{document.name}: {exc}"
    if isinstance(expected, bool) or not isinstance(expected, int | float):
        return None
    if not matches(written, float(expected)):
        return f"{document.name}: figure {written} does not match {label}#{pointer} = {expected:g}"
    return None


def check_row(document: Path, row: str, revision: str | None = None) -> str | None:
    """Return a problem message when a table row's figure disagrees with its result file."""
    match = REFERENCE.search(row)
    stated = stated_figure(row)
    if match is None or stated is None:
        return None
    return _compare(document, stated, match.group("name"), match.group("pointer"), revision)


def check_prose(document: Path, line: str, revision: str | None = None) -> tuple[list[str], int]:
    """Return the problems and the number of figures checked in one line of prose."""
    figures = [*PROSE_FIGURE.finditer(line), *HTML_FIGURE.finditer(line)]
    covered = {figure.start("name") for figure in figures}
    problems = [
        f"{document.name}: {reference.group(0)} is cited in prose without a figure "
        "written directly before it"
        for reference in REFERENCE.finditer(line)
        if reference.start("name") not in covered
    ]
    for figure in figures:
        problem = _compare(
            document,
            figure.group("figure"),
            figure.group("name"),
            figure.group("pointer"),
            revision,
        )
        if problem is not None:
            problems.append(problem)
    return problems, len(figures)


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def column_paths(header: str) -> list[str | None]:
    """Return the sub-path each header cell declares, e.g. ``HR@20 (`/hr@20/point`)``."""
    return [
        match.group("path") if (match := COLUMN_PATH.search(cell)) else None
        for cell in _cells(header)
    ]


def check_columns(
    document: Path, row: str, paths: list[str | None], revision: str | None = None
) -> tuple[list[str], int]:
    """Check every cell under a header sub-path against the row's pointer plus that sub-path."""
    reference = REFERENCE.search(row)
    if reference is None:
        return [], 0
    problems: list[str] = []
    checked = 0
    for cell, path in zip(_cells(row), paths, strict=False):
        if path is None:
            continue
        figure = NUMBER.match(cell.strip("`* "))
        if figure is None:
            problems.append(
                f"{document.name}: no figure under column {path} in the row citing "
                f"{reference.group(0)}"
            )
            continue
        checked += 1
        pointer = reference.group("pointer").rstrip("/") + path
        problem = _compare(
            document, figure.group("figure"), reference.group("name"), pointer, revision
        )
        if problem is not None:
            problems.append(problem)
    return problems, checked


def verify_document(document: Path) -> tuple[list[str], int]:
    """Return the drift problems and the number of verified figures for one document."""
    if not document.exists():
        return [], 0
    problems: list[str] = []
    checked = 0
    previous = ""
    paths: list[str | None] | None = None
    # In the changelog a released section is checked against the result files at its own tag: it
    # states what was true when it shipped, and the current files have since moved on.
    revision: str | None = None
    for line in document.read_text().splitlines():
        if document == CHANGELOG and line.startswith("## "):
            section = line[3:].split()[0]
            revision = None if section.lower() == "unreleased" else section
            if revision is not None and not tag_exists(revision):
                problems.append(f"{document.name}: section {revision} has no tag to check it at")
        if not line.lstrip().startswith("|"):
            paths = None
            prose_problems, prose_checked = check_prose(document, line, revision)
            problems.extend(prose_problems)
            checked += prose_checked
        elif SEPARATOR.match(line):
            declared = column_paths(previous)
            paths = declared if any(declared) else None
        elif paths is not None:
            column_problems, column_checked = check_columns(document, line, paths, revision)
            problems.extend(column_problems)
            checked += column_checked
        elif PROSE_FIGURE.search(line):
            # A row that cites figures in its cells, `23.9 ms (`results/...`)`, is read like prose:
            # every figure is checked, and a pointer without a figure in front of it is an error.
            cell_problems, cell_checked = check_prose(document, line, revision)
            problems.extend(cell_problems)
            checked += cell_checked
        else:
            problem = check_row(document, line, revision)
            if problem is None:
                checked += 1 if REFERENCE.search(line) and stated_figure(line) else 0
            else:
                problems.append(problem)
        previous = line
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
