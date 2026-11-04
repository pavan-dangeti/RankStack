"""Tests for the documentation drift check.

The check is what stands behind "every published figure comes from a committed result file", so
each way a stale figure once slipped past it is pinned here: a figure in prose, and a table cell
written with a unit.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from scripts import check_docs_drift
from scripts.check_docs_drift import check_prose, matches, stated_figure, verify_document

if TYPE_CHECKING:
    from pathlib import Path

POINTER = "results/run.json#/paired/difference"


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


def test_a_label_starting_with_a_digit_is_not_taken_as_the_figure() -> None:
    """A row label such as `300 (shipped)` must not be compared against the pointer."""
    assert stated_figure("| 300 (shipped) | 0.00915 | results/x.json#/a |") == "0.00915"
    assert stated_figure("| Warm user | 25.1 ms | 33.0 ms |") == "25.1"
    assert stated_figure("| Target only | p95 ≤ 250 ms |") is None
