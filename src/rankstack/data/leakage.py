"""Leakage checks that run in CI and gate every model.

A recommender can be made to look excellent by reading the future, and the failure is silent: the
number goes up and nobody notices. These checks are the guard. The negative control is the part
that proves the guard works — `check_shuffled_timeline_is_detected` breaks the timeline on purpose
and the suite must notice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from rankstack.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    import duckdb

    from rankstack.data.corpus import Corpus

_logger = get_logger(__name__)


class LeakKind(StrEnum):
    """What kind of leakage was detected."""

    PARTITION_ORDER = "partition_order"
    SPLIT_COUNTS = "split_counts"
    FEATURE_VALUE = "feature_value"
    FUTURE_ONLY_ITEM = "future_only_item"
    NEGATIVE_IS_POSITIVE = "negative_is_positive"


@dataclass(frozen=True, slots=True)
class Leak:
    """One detected leakage."""

    kind: LeakKind
    detail: str
    count: int

    def as_dict(self) -> dict[str, object]:
        """Return the leak as a JSON-serialisable mapping."""
        return {"kind": str(self.kind), "detail": self.detail, "count": self.count}


class LeakageError(RuntimeError):
    """Raised when a leakage check fails."""


@dataclass(slots=True)
class LeakageReport:
    """Every leak found in one run. Empty means clean."""

    leaks: list[Leak] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """True when no leak was found."""
        return not self.leaks

    def add(self, kind: LeakKind, detail: str, count: int = 1) -> None:
        """Record one leak; a zero count is ignored so the report carries no noise."""
        if count <= 0:
            return
        self.leaks.append(Leak(kind=kind, detail=detail, count=count))
        _logger.warning(
            "leakage.detected", extra={"kind": str(kind), "detail": detail, "count": count}
        )

    def raise_on_leak(self) -> None:
        """Raise `LeakageError` listing every leak, or return when the report is clean."""
        if self.clean:
            return
        lines = "\n".join(f"  - [{leak.kind}] {leak.detail} (x{leak.count})" for leak in self.leaks)
        msg = f"{len(self.leaks)} leakage violation(s):\n{lines}"
        raise LeakageError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the report as a JSON-serialisable mapping."""
        return {"clean": self.clean, "leaks": [leak.as_dict() for leak in self.leaks]}


def check_partition_order(
    report: LeakageReport,
    observed_ranges: dict[str, tuple[int | None, int | None]],
    corpus: Corpus,
) -> None:
    """Train must end before validation starts, and validation before test."""
    train_max = observed_ranges.get("train", (None, None))[1]
    valid_min = observed_ranges.get("valid", (None, None))[0]
    valid_max = observed_ranges.get("valid", (None, None))[1]
    test_min = observed_ranges.get("test", (None, None))[0]

    if train_max is not None and train_max >= corpus.train_end_ms:
        report.add(
            LeakKind.PARTITION_ORDER,
            f"train reaches {train_max}, boundary is {corpus.train_end_ms}",
        )
    if valid_min is not None and valid_min < corpus.train_end_ms:
        report.add(
            LeakKind.PARTITION_ORDER, f"valid starts at {valid_min}, before the train boundary"
        )
    if valid_max is not None and valid_max >= corpus.test_start_ms:
        report.add(
            LeakKind.PARTITION_ORDER,
            f"valid reaches {valid_max}, boundary is {corpus.test_start_ms}",
        )
    if test_min is not None and test_min < corpus.test_start_ms:
        report.add(
            LeakKind.PARTITION_ORDER, f"test starts at {test_min}, before the valid boundary"
        )


def check_split_counts(report: LeakageReport, observed: dict[str, int], corpus: Corpus) -> None:
    """Compare measured partition sizes with the publisher's published split sizes."""
    if corpus.split is None:
        return
    published = {
        "train": corpus.split.train,
        "valid": corpus.split.valid,
        "test": corpus.split.test,
    }
    for partition, expected in published.items():
        actual = observed.get(partition, 0)
        if abs(actual - expected) > expected * 0.02:
            report.add(
                LeakKind.SPLIT_COUNTS,
                f"{partition} has {actual:,} rows, publisher published {expected:,}",
            )


def check_feature_values(
    report: LeakageReport,
    connection: duckdb.DuckDBPyConnection,
    features_path: Path,
    interactions_relation: str,
) -> None:
    """Every stored snapshot must equal a recomputation over data strictly before its `as_of_ms`."""
    from rankstack.data.asof import count_feature_leaks  # noqa: PLC0415 - avoids an import cycle

    mismatches = count_feature_leaks(
        connection=connection,
        features_path=features_path,
        interactions_relation=interactions_relation,
    )
    report.add(
        LeakKind.FEATURE_VALUE,
        f"{mismatches} snapshot value(s) disagree with a point-in-time recomputation",
        mismatches,
    )


def check_future_only_items(
    report: LeakageReport,
    connection: duckdb.DuckDBPyConnection,
    features_path: Path,
    as_of_ms: int,
    interactions_relation: str,
) -> None:
    """Require that an item unseen before `as_of_ms` has no snapshot at `as_of_ms`.

    This is the cold-start boundary. An item that first appears after the prediction moment cannot
    appear in the trained features, so any score it receives must come from the content path.
    """
    row = connection.execute(
        f"""
        WITH stored AS (
            SELECT DISTINCT entity_id
            FROM read_parquet('{features_path.as_posix()}')
            WHERE as_of_ms = {as_of_ms} AND feature_name = 'item_interactions'
        ),
        before AS (
            SELECT DISTINCT parent_asin FROM {interactions_relation} WHERE timestamp < {as_of_ms}
        )
        SELECT count(*) FROM stored WHERE entity_id NOT IN (SELECT parent_asin FROM before)
        """
    ).fetchone()
    assert row is not None  # noqa: S101 - aggregate always returns one row
    report.add(
        LeakKind.FUTURE_ONLY_ITEM,
        f"items with a snapshot at {as_of_ms} but no interaction before it",
        int(row[0]),
    )


def check_negatives_are_not_positives(
    report: LeakageReport,
    connection: duckdb.DuckDBPyConnection,
    negatives_relation: str,
    positives_relation: str,
    train_end_ms: int,
) -> None:
    """Require that no sampled negative is a positive the user had at prediction time."""
    row = connection.execute(
        f"""
        SELECT count(*)
        FROM {negatives_relation} AS neg
        WHERE EXISTS (
            SELECT 1 FROM {positives_relation} AS pos
            WHERE pos.user_id = neg.user_id
              AND pos.parent_asin = neg.parent_asin
              AND pos.timestamp <= {train_end_ms}
        )
        """
    ).fetchone()
    assert row is not None  # noqa: S101
    report.add(
        LeakKind.NEGATIVE_IS_POSITIVE, "sampled negatives that are real positives", int(row[0])
    )


def check_shuffled_timeline_is_detected(
    connection: duckdb.DuckDBPyConnection,
    interactions_relation: str,
    features_path: Path,
) -> LeakageReport:
    """Negative control: shuffle timestamps, then require the feature check to complain.

    Timestamps are permuted across rows while item identities stay put, which is exactly the shape
    of accidental time leakage. A clean report here means the guard is not testing anything.
    """
    report = LeakageReport()
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _shuffled_interactions AS
        WITH keyed AS (SELECT row_number() OVER () AS k, * FROM {interactions_relation}),
        permuted AS (SELECT k, row_number() OVER (ORDER BY random()) AS slot FROM keyed)
        SELECT source.parent_asin, target.timestamp
        FROM permuted
        JOIN keyed AS source ON source.k = permuted.k
        JOIN keyed AS target ON target.k = permuted.slot
        """
    )
    check_feature_values(
        report,
        connection=connection,
        features_path=features_path,
        interactions_relation="_shuffled_interactions",
    )
    connection.execute("DROP TABLE IF EXISTS _shuffled_interactions")
    _logger.info("leakage.negative_control_complete", extra={"clean": report.clean})
    return report
