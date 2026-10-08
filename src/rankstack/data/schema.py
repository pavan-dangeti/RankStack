"""Canonical schema and the validation that gates every build.

Validation is a pipeline stage, not a notebook assertion: `build_dataset` refuses to write a
partition unless the contract below holds. Each check declares how it is compared, so "must be
zero", "must match a published count within tolerance" and "must be at least" can never be confused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

import pyarrow as pa

from rankstack.data.corpus import MAX_TIMESTAMP_MS, MIN_TIMESTAMP_MS, STAT_TOLERANCE

#: Column names and declared types of the publisher's `rating_only` files.
INTERACTION_SOURCE_COLUMNS: Final[dict[str, str]] = {
    "user_id": "VARCHAR",
    "parent_asin": "VARCHAR",
    "rating": "DOUBLE",
    "timestamp": "BIGINT",
}

#: Written interactions. `timestamp` stays authoritative; `event_time` is derived for readability.
INTERACTION_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("parent_asin", pa.string(), nullable=False),
        pa.field("rating", pa.float32(), nullable=False),
        pa.field("timestamp", pa.int64(), nullable=False),
        pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

#: Written item metadata, one row per parent_asin.
ITEM_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("parent_asin", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("main_category", pa.string(), nullable=True),
        pa.field("category_path", pa.string(), nullable=True),
        pa.field("price", pa.float64(), nullable=True),
        pa.field("store", pa.string(), nullable=True),
        pa.field("description", pa.string(), nullable=True),
    ]
)

#: Long-format feature snapshots. `as_of` is what makes point-in-time reads possible.
FEATURE_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("entity_type", pa.string(), nullable=False),
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("feature_name", pa.string(), nullable=False),
        pa.field("as_of_ms", pa.int64(), nullable=False),
        pa.field("value_double", pa.float64(), nullable=True),
    ]
)

#: Truncate item descriptions to this many characters. Item text only ever feeds a local embedding
#: model; keeping the corpus small keeps the embedding step fast and the artefacts small.
DESCRIPTION_MAX_CHARS: Final[int] = 512


class CheckKind(StrEnum):
    """What a failing check does to the build."""

    HARD_FAIL = "hard_fail"
    DROP = "drop"
    REPORT = "report"


class Compare(StrEnum):
    """How the observed value is compared with the expected value."""

    MUST_BE_ZERO = "must_be_zero"
    WITHIN_TOLERANCE = "within_tolerance"
    AT_LEAST = "at_least"


class ValidationError(RuntimeError):
    """Raised when a hard validation check fails. The build stops."""


@dataclass(frozen=True, slots=True)
class Check:
    """One validation rule over a staged table."""

    name: str
    kind: CheckKind
    compare: Compare
    sql: str
    expected: float
    detail: str
    tolerance: float = STAT_TOLERANCE


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of one check."""

    name: str
    kind: CheckKind
    compare: Compare
    observed: float
    expected: float
    detail: str
    tolerance: float = STAT_TOLERANCE

    @property
    def passed(self) -> bool:
        match self.compare:
            case Compare.MUST_BE_ZERO:
                return self.observed == 0
            case Compare.WITHIN_TOLERANCE:
                return abs(self.observed - self.expected) <= self.expected * self.tolerance
            case Compare.AT_LEAST:
                return self.observed >= self.expected


@dataclass(frozen=True, slots=True)
class CheckEvaluation:
    """A check plus its outcome."""

    check: Check
    result: CheckResult

    @property
    def failed(self) -> bool:
        return self.check.kind is CheckKind.HARD_FAIL and not self.result.passed


@dataclass(slots=True)
class ValidationReport:
    """All check outcomes for one dataset build, written into the manifest."""

    evaluations: list[CheckEvaluation] = field(default_factory=list)

    def add(self, check: Check, observed: float) -> CheckEvaluation:
        evaluation = CheckEvaluation(
            check=check,
            result=CheckResult(
                name=check.name,
                kind=check.kind,
                compare=check.compare,
                observed=observed,
                expected=check.expected,
                detail=check.detail,
                tolerance=check.tolerance,
            ),
        )
        self.evaluations.append(evaluation)
        return evaluation

    @property
    def failures(self) -> list[CheckEvaluation]:
        return [e for e in self.evaluations if e.failed]

    @property
    def rows_dropped(self) -> int:
        return sum(
            int(e.result.observed) for e in self.evaluations if e.check.kind is CheckKind.DROP
        )

    def raise_on_failure(self) -> None:
        if not self.failures:
            return
        lines = "\n".join(
            f"  - {e.check.name}: observed={e.result.observed:g} "
            f"expected={e.check.expected:g} [{e.check.compare}] ({e.check.detail})"
            for e in self.failures
        )
        names = ", ".join(e.check.name for e in self.failures)
        msg = f"{len(self.failures)} hard validation check(s) failed: {names}\n{lines}"
        raise ValidationError(msg)

    def as_dict(self) -> dict[str, object]:
        return {
            "checks": [
                {
                    "name": e.check.name,
                    "kind": str(e.check.kind),
                    "compare": str(e.check.compare),
                    "observed": e.result.observed,
                    "expected": e.check.expected,
                    "passed": e.result.passed,
                    "detail": e.check.detail,
                }
                for e in self.evaluations
            ],
            "rows_dropped": self.rows_dropped,
        }


def interaction_checks(corpus_name: str, expected_interactions: int) -> list[Check]:
    """Hard checks on the staged interaction table."""
    return [
        Check(
            name="null_user_id",
            kind=CheckKind.HARD_FAIL,
            compare=Compare.MUST_BE_ZERO,
            sql="SELECT count(*) FROM staged WHERE user_id IS NULL OR trim(user_id) = ''",
            expected=0.0,
            detail="user_id must be present and non-empty",
        ),
        Check(
            name="null_parent_asin",
            kind=CheckKind.HARD_FAIL,
            compare=Compare.MUST_BE_ZERO,
            sql="SELECT count(*) FROM staged WHERE parent_asin IS NULL OR trim(parent_asin) = ''",
            expected=0.0,
            detail="parent_asin must be present and non-empty",
        ),
        Check(
            name="rating_out_of_scale",
            kind=CheckKind.DROP,
            compare=Compare.MUST_BE_ZERO,
            sql="SELECT count(*) FROM staged WHERE rating IS NULL OR rating < 1.0 OR rating > 5.0",
            expected=0.0,
            detail=(
                "ratings are on a 1-5 scale; the publisher's Electronics 0-core file contains "
                "2 rows rated 0.0 and 1 null rating, which are dropped and counted"
            ),
        ),
        Check(
            name="timestamp_out_of_range",
            kind=CheckKind.HARD_FAIL,
            compare=Compare.MUST_BE_ZERO,
            sql=(
                "SELECT count(*) FROM staged "
                f"WHERE timestamp < {MIN_TIMESTAMP_MS} OR timestamp > {MAX_TIMESTAMP_MS}"
            ),
            expected=0.0,
            detail=f"timestamps must fall within the publisher's corpus range "
            f"{MIN_TIMESTAMP_MS}..{MAX_TIMESTAMP_MS}",
        ),
        Check(
            name="interaction_count",
            kind=CheckKind.HARD_FAIL,
            compare=Compare.WITHIN_TOLERANCE,
            sql="SELECT count(*) FROM staged",
            expected=float(expected_interactions),
            detail=f"{corpus_name} published interaction count, rounded to 3 significant figures",
        ),
    ]


def duplicate_check() -> Check:
    """Exact duplicate events: the same user, item and millisecond reviewed twice."""
    return Check(
        name="duplicate_events",
        kind=CheckKind.DROP,
        compare=Compare.MUST_BE_ZERO,
        sql=(
            "SELECT count(*) FROM ("
            "  SELECT user_id, parent_asin, timestamp, count(*) AS n FROM staged"
            "  GROUP BY ALL HAVING count(*) > 1"
            ")"
        ),
        expected=0.0,
        detail="exact duplicate (user_id, parent_asin, timestamp) rows; one copy is kept",
    )


def item_checks(min_metadata_coverage: float) -> list[Check]:
    """Hard and reported checks on the staged item table."""
    return [
        Check(
            name="null_item_parent_asin",
            kind=CheckKind.HARD_FAIL,
            compare=Compare.MUST_BE_ZERO,
            sql="SELECT count(*) FROM staged_items WHERE parent_asin IS NULL OR trim(parent_asin) = ''",
            expected=0.0,
            detail="item metadata must carry a parent_asin",
        ),
        Check(
            name="item_metadata_coverage",
            kind=CheckKind.HARD_FAIL,
            compare=Compare.AT_LEAST,
            sql="""
                WITH interacted AS (SELECT DISTINCT parent_asin FROM staged)
                SELECT 1.0 - (
                    SELECT count(*)
                    FROM interacted
                    WHERE NOT EXISTS (
                        SELECT 1 FROM staged_items WHERE staged_items.parent_asin = interacted.parent_asin
                    )
                )::DOUBLE / greatest((SELECT count(*) FROM interacted), 1)
            """,
            expected=min_metadata_coverage,
            detail="fraction of interacted items that have metadata; a partial download fails here",
        ),
    ]


def report_checks() -> list[Check]:
    """Checks recorded but never fatal, so the manifest documents what was imperfect."""
    return [
        Check(
            name="items_without_title",
            kind=CheckKind.REPORT,
            compare=Compare.AT_LEAST,
            sql="SELECT count(*) FROM staged_items WHERE title IS NULL OR trim(title) = ''",
            expected=0.0,
            detail="titled from main_category when absent",
        ),
        Check(
            name="items_without_category_path",
            kind=CheckKind.REPORT,
            compare=Compare.AT_LEAST,
            sql="SELECT count(*) FROM staged_items WHERE category_path = ''",
            expected=0.0,
            detail="category path is used for category affinity; a missing one is treated as unknown",
        ),
        Check(
            name="items_without_price",
            kind=CheckKind.REPORT,
            compare=Compare.AT_LEAST,
            sql="SELECT count(*) FROM staged_items WHERE price IS NULL",
            expected=0.0,
            detail="price is optional; the ranker treats it as missing rather than zero",
        ),
        Check(
            name="duplicate_item_metadata",
            kind=CheckKind.DROP,
            compare=Compare.MUST_BE_ZERO,
            sql="SELECT count(*) - count(DISTINCT parent_asin) FROM staged_items",
            expected=0.0,
            detail="conflicting metadata rows for one parent_asin; the first occurrence is kept",
        ),
    ]