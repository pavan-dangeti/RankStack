"""Property-based tests for the two invariants the whole evaluation rests on."""

from __future__ import annotations

import duckdb
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rankstack.data.asof import as_of_join_sql, item_as_of_sql
from rankstack.data.download import DownloadError, _open
from tests.helpers import row, scalar

TRAIN_END = 1_600_000_000_000
TEST_START = 1_700_000_000_000

timestamps = st.integers(min_value=1_500_000_000_000, max_value=1_900_000_000_000)
asins = st.sampled_from(["B000000001", "B000000002", "B000000003"])


def _load(rows: list[tuple[str, str, int]], relation: str = "events") -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(database=":memory:")
    connection.execute(
        f"CREATE TABLE {relation} (user_id VARCHAR, parent_asin VARCHAR, timestamp BIGINT)"
    )
    if rows:
        connection.executemany(
            f"INSERT INTO {relation} VALUES (?, ?, ?)",
            [(user, asin, ts) for user, asin, ts in rows],
        )
    return connection


@given(rows=st.lists(st.tuples(st.just("U1"), asins, timestamps), min_size=1, max_size=40))
@settings(max_examples=50, deadline=None)
def test_partitions_are_disjoint_and_cover_every_row(rows: list[tuple[str, str, int]]) -> None:
    connection = _load(rows)
    total = scalar(connection, "SELECT count(*) FROM events")
    counts = row(
        connection,
        f"""
        SELECT
            count(*) FILTER (WHERE timestamp < {TRAIN_END}),
            count(*) FILTER (WHERE timestamp >= {TRAIN_END} AND timestamp < {TEST_START}),
            count(*) FILTER (WHERE timestamp >= {TEST_START})
        FROM events
        """,
    )
    connection.close()
    assert sum(float(value) for value in counts) == total


@given(
    rows=st.lists(st.tuples(st.just("U1"), asins, timestamps), min_size=1, max_size=40),
    as_of=st.integers(min_value=1_500_000_000_000, max_value=1_900_000_000_000),
)
@settings(max_examples=50, deadline=None)
def test_as_of_features_only_use_interactions_before_the_boundary(
    rows: list[tuple[str, str, int]], as_of: int
) -> None:
    connection = _load(rows)
    connection.execute("CREATE OR REPLACE TABLE clean_interactions AS SELECT * FROM events")
    connection.execute(f"CREATE TABLE snapshots AS {item_as_of_sql('clean_interactions', as_of)}")
    leaked = scalar(
        connection,
        """
        SELECT count(*) FROM snapshots AS s
        WHERE s.feature_name = 'item_interactions'
          AND s.value_double <> (
            SELECT count(*)::DOUBLE FROM clean_interactions
            WHERE parent_asin = s.entity_id AND timestamp < s.as_of_ms
          )
        """,
    )
    future = scalar(connection, "SELECT count(*) FROM snapshots WHERE as_of_ms > ?", [as_of])
    connection.close()
    assert leaked == 0
    assert future == 0


@given(
    rows=st.lists(st.tuples(st.just("U1"), asins, timestamps), min_size=1, max_size=30),
    as_of=st.integers(min_value=1_500_000_000_000, max_value=1_900_000_000_000),
)
@settings(max_examples=30, deadline=None)
def test_as_of_join_never_returns_a_future_snapshot(
    rows: list[tuple[str, str, int]], as_of: int
) -> None:
    connection = _load(rows)
    connection.execute("CREATE OR REPLACE TABLE clean_interactions AS SELECT * FROM events")
    snapshots = item_as_of_sql("clean_interactions", as_of)
    connection.execute(f"CREATE TABLE features AS {snapshots}")
    entity = sorted({row[1] for row in rows})[0]
    result = connection.execute(
        as_of_join_sql(
            "features",
            f"(SELECT 'item' AS entity_type, '{entity}' AS entity_id, {as_of} AS as_of_ms)",
        )
    ).fetchall()
    connection.close()
    assert all(row[4] is None or row[4] <= as_of for row in result)


def test_unknown_scheme_is_refused() -> None:
    with pytest.raises(DownloadError, match="not allowed"):
        _open("file+weird://example/x")
