"""Validation contract: every rule must pass clean data and fail bad data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rankstack.data.corpus import MAX_TIMESTAMP_MS, MIN_TIMESTAMP_MS, get_corpus
from rankstack.data.schema import (
    Check,
    CheckKind,
    Compare,
    ValidationError,
    ValidationReport,
    duplicate_check,
    interaction_checks,
    item_checks,
    report_checks,
)
from tests.helpers import scalar

if TYPE_CHECKING:
    import duckdb

STAGE = """
CREATE OR REPLACE TABLE staged AS
SELECT * FROM (VALUES
    ('U1', 'B000000001', 5.0, {t1}),
    ('U1', 'B000000002', 3.0, {t2}),
    ('U2', 'B000000001', 1.0, {t0})
) AS t(user_id, parent_asin, rating, timestamp)
"""


@pytest.fixture
def staged(connection: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    corpus = get_corpus("electronics_0core")
    connection.execute(
        STAGE.format(
            t0=corpus.train_end_ms - 86_400_000, t1=corpus.train_end_ms, t2=corpus.test_start_ms
        )
    )
    return connection


def run(connection: duckdb.DuckDBPyConnection, checks: list[Check]) -> ValidationReport:
    """Run checks and fail the test if any hard check fails."""
    report = observe(connection, checks)
    report.raise_on_failure()
    return report


def observe(connection: duckdb.DuckDBPyConnection, checks: list[Check]) -> ValidationReport:
    """Run checks without raising, so a test can assert which ones fail."""
    report = ValidationReport()
    for check in checks:
        report.add(check, scalar(connection, check.sql))
    return report


def test_clean_data_passes_every_hard_check(staged: duckdb.DuckDBPyConnection) -> None:
    report = run(staged, interaction_checks("test", expected_interactions=3))
    assert not report.failures


def test_duplicate_check_is_reported_as_a_drop_not_a_failure(
    staged: duckdb.DuckDBPyConnection,
) -> None:
    staged.execute(
        "INSERT INTO staged VALUES ('U1', 'B000000001', 5.0, "
        f"{get_corpus('electronics_0core').train_end_ms})"
    )
    report = run(staged, [duplicate_check()])
    assert report.rows_dropped == 1
    assert report.failures == []


def test_null_user_id_fails_the_build(staged: duckdb.DuckDBPyConnection) -> None:
    staged.execute("INSERT INTO staged VALUES (NULL, 'B000000003', 4.0, 1650000000000)")
    report = observe(staged, interaction_checks("test", expected_interactions=3))
    with pytest.raises(ValidationError, match="null_user_id"):
        report.raise_on_failure()


@pytest.mark.parametrize("bad_rating", [0.0, 5.5, None])
def test_rating_outside_the_scale_is_dropped_not_fatal(
    staged: duckdb.DuckDBPyConnection, bad_rating: float | None
) -> None:
    staged.execute("INSERT INTO staged VALUES ('U9', 'B000000009', ?, 1650000000000)", [bad_rating])
    report = observe(staged, interaction_checks("test", expected_interactions=4))
    report.raise_on_failure()
    assert report.rows_dropped == 1


def test_timestamp_outside_the_publisher_range_fails(
    staged: duckdb.DuckDBPyConnection,
) -> None:
    corpus = get_corpus("electronics_0core")
    staged.execute(
        "INSERT INTO staged VALUES ('U9', 'B000000009', 4.0, ?)", [corpus.test_start_ms + 10**12]
    )
    report = observe(staged, interaction_checks("test", expected_interactions=3))
    with pytest.raises(ValidationError, match="timestamp_out_of_range"):
        report.raise_on_failure()


def test_row_count_must_match_the_published_statistic(
    staged: duckdb.DuckDBPyConnection,
) -> None:
    report = observe(staged, interaction_checks("test", expected_interactions=999))
    with pytest.raises(ValidationError, match="interaction_count"):
        report.raise_on_failure()


def test_tolerance_comparison_respects_the_published_rounding() -> None:
    published_count = Check(
        name="count",
        kind=CheckKind.HARD_FAIL,
        compare=Compare.WITHIN_TOLERANCE,
        sql="SELECT 0",
        expected=43_400_000,
        detail="published count, rounded to 3 significant figures",
    )
    report = ValidationReport()
    for observed, passes in ((43_450_000, True), (42_600_000, True), (50_000_000, False)):
        evaluation = report.add(published_count, float(observed))
        assert evaluation.result.passed is passes, observed


def test_item_coverage_check_detects_missing_metadata(
    staged: duckdb.DuckDBPyConnection,
) -> None:
    staged.execute(
        "CREATE OR REPLACE TABLE staged_items AS SELECT * FROM (VALUES "
        "('B000000001', 'A product'), ('B000000002', 'Another')"
        ") AS t(parent_asin, title)"
    )
    report = run(staged, item_checks(min_metadata_coverage=0.9))
    assert not report.failures

    staged.execute("DELETE FROM staged_items WHERE parent_asin = 'B000000002'")
    report = observe(staged, item_checks(min_metadata_coverage=0.9))
    with pytest.raises(ValidationError, match="item_metadata_coverage"):
        report.raise_on_failure()


def test_report_checks_never_fail_the_build(staged: duckdb.DuckDBPyConnection) -> None:
    staged.execute(
        "CREATE OR REPLACE TABLE staged_items AS SELECT * FROM (VALUES "
        "('B000000001', '', 1.0, '', 'c'), ('B000000002', 't', NULL, 'd', '')"
        ") AS t(parent_asin, title, price, description, category_path)"
    )
    report = run(staged, report_checks())
    assert {str(check.kind) for check in report_checks()} == {"report", "drop"}
    assert "items_without_category_path" in {check.name for check in report_checks()}
    assert report.failures == []


def test_check_kinds_and_comparisons_are_distinct() -> None:
    checks = interaction_checks("t", 1) + item_checks(0.5) + report_checks() + [duplicate_check()]
    assert {str(check.kind) for check in checks} == {"hard_fail", "drop", "report"}
    assert {str(check.compare) for check in checks} == {
        "must_be_zero",
        "within_tolerance",
        "at_least",
    }
    assert MIN_TIMESTAMP_MS < MAX_TIMESTAMP_MS
