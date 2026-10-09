"""End-to-end: build a corpus, then prove the leakage suite catches a broken timeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import pytest

from rankstack.data.asof import AS_OF_FEATURES_PATH, read_as_of
from rankstack.data.build import build_dataset
from rankstack.data.corpus import TEST_START_MS, TRAIN_END_MS
from rankstack.data.leakage import (
    LeakageReport,
    check_feature_values,
    check_future_only_items,
    check_partition_order,
    check_shuffled_timeline_is_detected,
    check_split_counts,
)
from tests.conftest import synthetic_corpus_spec

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from rankstack.config import Settings
    from tests.conftest import SyntheticCorpus

pytestmark = pytest.mark.e2e


@pytest.fixture
def built(
    synthetic_corpus: SyntheticCorpus, test_settings: Settings
) -> Generator[tuple[Path, duckdb.DuckDBPyConnection], None, None]:
    spec = synthetic_corpus_spec(synthetic_corpus)
    build_dataset(spec, settings=test_settings, skip_download=True)
    features_path = test_settings.processed(spec.name) / AS_OF_FEATURES_PATH
    with duckdb.connect(database=":memory:") as connection:
        interactions = test_settings.processed(spec.name) / "interactions"
        names = ("train", "valid", "test")
        paths = [(interactions / f"{name}.parquet").as_posix() for name in names]
        literal = ", ".join(f"'{path}'" for path in paths)
        connection.execute(
            f"CREATE TABLE clean_interactions AS SELECT * FROM read_parquet([{literal}])"
        )
        yield features_path, connection
    connection.close()


def test_clean_pipeline_passes_every_leakage_check(built, synthetic_corpus) -> None:  # type: ignore[no-untyped-def]
    features_path, connection = built
    report = LeakageReport()
    check_partition_order(
        report,
        {
            "train": (TRAIN_END_MS - 86_400_000, TRAIN_END_MS - 1),
            "valid": (TRAIN_END_MS + 1, TEST_START_MS - 1),
            "test": (TEST_START_MS + 1, TEST_START_MS + 10 * 86_400_000),
        },
        synthetic_corpus_spec(synthetic_corpus),
    )
    check_split_counts(
        report, {"train": 100, "valid": 60, "test": 61}, synthetic_corpus_spec(synthetic_corpus)
    )
    check_feature_values(report, connection, features_path, "clean_interactions")
    check_future_only_items(report, connection, features_path, TRAIN_END_MS, "clean_interactions")
    assert report.clean, report.as_dict()
    report.raise_on_leak()


def test_negative_control_detects_a_shuffled_timeline(built) -> None:  # type: ignore[no-untyped-def]
    features_path, connection = built
    report = check_shuffled_timeline_is_detected(
        connection=connection,
        interactions_relation="clean_interactions",
        features_path=features_path,
    )
    assert not report.clean, "a shuffled timeline must trip the leakage suite"


def test_a_new_item_has_no_snapshot_at_any_boundary(
    built: tuple[Path, duckdb.DuckDBPyConnection], synthetic_corpus: SyntheticCorpus
) -> None:
    """The cold-start boundary, asserted from both sides.

    An item whose first interaction is inside the test window has no snapshot at t1, because it did
    not exist when training data was cut. It also has none at t2, because its first interaction is
    after t2. Collaborative features cannot exist for it at all; only the content path can score it,
    which is what Phase 4 builds.
    """
    features_path, connection = built
    for as_of in (TRAIN_END_MS, TEST_START_MS):
        rows = read_as_of(
            connection,
            features_path,
            "item",
            [synthetic_corpus.new_item_asin],
            as_of_ms=as_of,
        )
        assert rows == [], f"a new item must have no features at {as_of}"

    established = read_as_of(
        connection, features_path, "item", ["B000000001"], as_of_ms=TRAIN_END_MS
    )
    assert established, "an item with train-window history must have features at t1"


def test_feature_values_match_a_hand_computation(built) -> None:  # type: ignore[no-untyped-def]
    features_path, connection = built
    asin = "B000000001"
    expected = connection.execute(
        "SELECT count(*) FROM clean_interactions WHERE parent_asin = ? AND timestamp < ?",
        [asin, TRAIN_END_MS],
    ).fetchone()[0]
    rows = {
        feature: value
        for _, feature, _, value in read_as_of(
            connection, features_path, "item", [asin], as_of_ms=TRAIN_END_MS
        )
    }
    assert rows["item_interactions"] == float(expected)


def test_partition_order_violation_is_reported(synthetic_corpus) -> None:  # type: ignore[no-untyped-def]
    report = LeakageReport()
    check_partition_order(
        report,
        {"train": (0, TEST_START_MS + 1), "valid": (0, 0), "test": (0, 0)},
        synthetic_corpus_spec(synthetic_corpus),
    )
    assert not report.clean
    with pytest.raises(Exception, match="leakage violation"):
        report.raise_on_leak()
