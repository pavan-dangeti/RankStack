"""Tests for the documentation drift check.

The check is what stands behind "every published figure comes from a committed result file", so
each way a stale figure once slipped past it is pinned here: a figure in prose, and a table cell
written with a unit.
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest
from scripts import check_docs_drift
from scripts.check_docs_drift import check_prose, matches, stated_figure, verify_document

if TYPE_CHECKING:
    from pathlib import Path

POINTER = "results/run.json#/paired/difference"


@pytest.fixture(autouse=True)
def fresh_results() -> None:
    """Forget result files read by an earlier test; each test writes its own."""
    check_docs_drift.load_result.cache_clear()


@pytest.fixture
def document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the check at one synthetic result file and return a path for the document."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "run.json").write_text(
        json.dumps({"paired": {"difference": -0.0148}, "latency": [{"p50_ms": 26.143}]})
    )
    monkeypatch.setattr(check_docs_drift, "RESULTS_DIR", results)
    return tmp_path / "README.md"


def test_a_stale_figure_in_prose_is_reported(document: Path) -> None:
    """The README intro's -0.0173 sat in prose, where the check used to read nothing."""
    problems, checked = check_prose(document, f"the re-ranker scores -0.0173 (`{POINTER}`) here")

    assert checked == 1
    assert len(problems) == 1
    assert "-0.0173 does not match" in problems[0]


def test_a_current_figure_in_prose_passes(document: Path) -> None:
    """The corrected sentence produces no problem."""
    assert check_prose(document, f"the re-ranker scores -0.0148 (`{POINTER}`) here") == ([], 1)


def test_a_prose_pointer_without_a_figure_is_an_error(document: Path) -> None:
    """Citing a result file next to an unchecked number must fail rather than pass silently."""
    problems, checked = check_prose(document, f"about -0.0173 against stage 1 (see `{POINTER}`)")

    assert checked == 0
    assert len(problems) == 1
    assert "without a figure" in problems[0]


def test_a_table_cell_with_a_unit_is_compared(document: Path) -> None:
    """`24.1 ms` used to be skipped because the old pattern accepted bare numbers only."""
    document.write_text(
        "| Path | p50 | Figure |\n"
        "|---|---|---|\n"
        "| Warm user | 24.1 ms | results/run.json#/latency/0/p50_ms |\n"
    )
    problems, _ = verify_document(document)
    assert len(problems) == 1
    assert "24.1 does not match" in problems[0]

    document.write_text("| Warm user | 26.1 ms | results/run.json#/latency/0/p50_ms |\n")
    assert verify_document(document) == ([], 1)


def test_a_figure_must_equal_the_result_rounded_to_its_written_precision() -> None:
    """Rounding to the precision shown is allowed; any other difference is drift."""
    assert matches("26.1", 26.143)
    assert matches("26.143", 26.143)
    assert not matches("26.2", 26.143)
    assert not matches("26.14", 26.149)
    assert matches("43,365,424", 43_365_424)
    assert matches("-0.0148", -0.0148)


SWEEP_TABLE = (
    "| w (`/weight`) | HR@20 (`/hr/point`) | Coverage (`/coverage/items`) | Figure |\n"
    "|---|---|---|---|\n"
    "| 0.25 | {hr} | {coverage} | results/sweep.json#/by_weight/0.25 |\n"
)


@pytest.fixture
def sweep_document(document: Path) -> Path:
    """Add a sweep-shaped result file, where each row's pointer names an object, not a figure."""
    row = {"weight": 0.25, "hr": {"point": 0.0325}, "coverage": {"items": 42}}
    (check_docs_drift.RESULTS_DIR / "sweep.json").write_text(
        json.dumps({"by_weight": {"0.25": row}})
    )
    return document


def test_every_cell_under_a_header_sub_path_is_checked(sweep_document: Path) -> None:
    """Each column is compared, not just the first figure in the row."""
    sweep_document.write_text(SWEEP_TABLE.format(hr="0.03250", coverage="42"))
    assert verify_document(sweep_document) == ([], 3)

    sweep_document.write_text(SWEEP_TABLE.format(hr="0.03250", coverage="41"))
    problems, _ = verify_document(sweep_document)
    assert len(problems) == 1
    assert "41 does not match" in problems[0]


def test_a_missing_figure_under_a_header_sub_path_is_an_error(sweep_document: Path) -> None:
    """A column that declares a sub-path must hold a figure in every row."""
    sweep_document.write_text(SWEEP_TABLE.format(hr="n/a", coverage="42"))
    problems, checked = verify_document(sweep_document)
    assert checked == 2
    assert len(problems) == 1
    assert "no figure under column /hr/point" in problems[0]


def test_every_figure_cited_in_a_table_cell_is_checked(document: Path) -> None:
    """A key-results row cites several figures in its cells; each one is compared."""
    row = "| Latency p50 / p99 | {p50} ms (`results/run.json#/latency/0/p50_ms`) / {extra} |\n"
    document.write_text(row.format(p50="26.1", extra="-0.0148 (`" + POINTER + "`)"))
    assert verify_document(document) == ([], 2)

    document.write_text(row.format(p50="25.0", extra="see `" + POINTER + "`"))
    problems, checked = verify_document(document)
    assert checked == 1
    assert any("25.0 does not match" in problem for problem in problems)
    assert any("without a figure" in problem for problem in problems)


def test_a_label_starting_with_a_digit_is_not_taken_as_the_figure() -> None:
    """A row label such as `300 (shipped)` must not be compared against the pointer."""
    assert stated_figure("| 300 (shipped) | 0.00915 | results/x.json#/a |") == "0.00915"
    assert stated_figure("| Warm user | 25.1 ms | 33.0 ms |") == "25.1"
    assert stated_figure("| Target only | p95 ≤ 250 ms |") is None


def _git(repo: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603 - fixed git argv in a temporary repository
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],  # noqa: S607
        check=True,
        capture_output=True,
    )


@pytest.fixture
def released(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a repository whose result file read 1.0 when v9 was tagged and reads 2.0 now."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "run.json").write_text(json.dumps({"value": 1.0}))
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "results/run.json")
    _git(tmp_path, "commit", "-q", "-m", "release")
    _git(tmp_path, "tag", "-a", "v9", "-m", "v9")
    (results / "run.json").write_text(json.dumps({"value": 2.0}))
    changelog = tmp_path / "CHANGELOG.md"
    monkeypatch.setattr(check_docs_drift, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check_docs_drift, "RESULTS_DIR", results)
    monkeypatch.setattr(check_docs_drift, "CHANGELOG", changelog)
    return changelog


CITE = "(`results/run.json#/value`)"


def test_a_released_section_is_checked_at_its_tag(released: Path) -> None:
    """Unreleased states today's figure, v9 the figure it shipped with; both pass."""
    released.write_text(f"## Unreleased\n\nnow 2.0 {CITE}\n\n## v9\n\nthen 1.0 {CITE}\n")
    assert verify_document(released) == ([], 2)


def test_a_released_figure_is_not_checked_against_todays_file(released: Path) -> None:
    """Rewriting a released entry to today's figure is drift, even though today's file agrees."""
    released.write_text(f"## v9\n\nthen 2.0 {CITE}\n")
    problems, _ = verify_document(released)
    assert len(problems) == 1
    assert "at v9" in problems[0]


def test_a_section_without_a_tag_is_an_error(released: Path) -> None:
    """A released heading must name a tag, or there is nothing to check it against."""
    released.write_text(f"## v10\n\nsoon 2.0 {CITE}\n")
    problems, _ = verify_document(released)
    assert any("v10 has no tag" in problem for problem in problems)
