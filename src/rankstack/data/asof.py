"""Point-in-time feature reads.

The whole point of a feature store is that a feature value for a training example is computed only
from data that existed at that example's prediction moment. Here that is one primitive: a table of
feature snapshots carrying an `as_of_ms`, and a join that only ever returns snapshots with
`as_of_ms <= query_as_of_ms`.

Phase 1 materialises item-level snapshots. User-level and interaction-level snapshots arrive with the
baselines that need them; the join below is the same for all three.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import duckdb

from rankstack.logging import get_logger

_logger = get_logger(__name__)

AS_OF_FEATURES_PATH: Final[str] = "features/item_as_of_features.parquet"

#: Item features that are cheap to recompute at any boundary and that make the cold-start signal
#: visible: an item with no row at t1 did not exist yet, which is exactly what a recommender at
#: launch time is allowed to know.
ITEM_AS_OF_FEATURES: Final[tuple[str, ...]] = (
    "item_interactions",
    "item_days_since_first_interaction",
    "item_days_since_last_interaction",
)


@dataclass(frozen=True, slots=True)
class AsOfReadStats:
    """Counts describing a point-in-time read, used in manifests and by the leakage checks."""

    features: int
    rows: int
    entities: int
    future_rows: int

    def as_dict(self) -> dict[str, object]:
        return {
            "features": self.features,
            "rows": self.rows,
            "entities": self.entities,
            "future_rows": self.future_rows,
        }


def _quote(path: Path) -> str:
    return str(path).replace("'", "''")


def item_as_of_sql(source_relation: str, as_of_ms: int) -> str:
    """SQL producing long-format item feature snapshots as of `as_of_ms`.

    Every aggregate filters with a strict `< as_of_ms`, so an interaction at exactly the boundary is
    not counted. The filter is inside each aggregate; filtering after aggregating would leak. Each
    branch is self-contained (no CTE) so several boundaries can be UNIONed into one query.
    """
    day_ms = 86_400_000
    aggregated = (
        f"SELECT parent_asin AS entity_id, count(*)::DOUBLE AS interactions,"
        f" min(timestamp) AS first_ts, max(timestamp) AS last_ts"
        f" FROM {source_relation} WHERE timestamp < {as_of_ms} GROUP BY parent_asin"
    )
    template = (
        "SELECT 'item' AS entity_type, entity_id, '{name}' AS feature_name,"
        f" {as_of_ms} AS as_of_ms, {{value}} AS value_double FROM ({aggregated})"
    )
    return "\nUNION ALL\n".join(
        [
            template.format(name="item_interactions", value="interactions"),
            template.format(
                name="item_days_since_first_interaction",
                value=f"({as_of_ms} - first_ts)::DOUBLE / {day_ms}",
            ),
            template.format(
                name="item_days_since_last_interaction",
                value=f"({as_of_ms} - last_ts)::DOUBLE / {day_ms}",
            ),
        ]
    )


def write_item_as_of_features(
    connection: duckdb.DuckDBPyConnection,
    out_dir: Path,
    boundaries: dict[str, int],
) -> dict[str, object]:
    """Write item feature snapshots at each boundary and report what was written."""
    target = out_dir / "features" / Path(AS_OF_FEATURES_PATH).name
    target.parent.mkdir(parents=True, exist_ok=True)
    statements = [item_as_of_sql("clean_interactions", as_of) for as_of in boundaries.values()]
    union = "\nUNION ALL\n".join(statements)
    # Sorted output keeps the Parquet bytes identical across runs, so the manifest hash is a
    # meaningful build fingerprint rather than a record of DuckDB's thread scheduling.
    ordered = (
        f"SELECT * FROM ({union}) ORDER BY entity_type, entity_id, feature_name, as_of_ms"
    )
    connection.execute(f"COPY ({ordered}) TO '{_quote(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    row = connection.execute(
        f"SELECT count(*), count(DISTINCT entity_id) FROM read_parquet('{_quote(target)}')"
    ).fetchone()
    assert row is not None  # noqa: S101 - aggregate always returns one row
    stats = {"path": str(Path(AS_OF_FEATURES_PATH)), "rows": int(row[0]), "entities": int(row[1])}
    _logger.info("data.as_of_features_written", extra=stats)
    return stats


def as_of_join_sql(features_relation: str, query_relation: str) -> str:
    """SQL that attaches each query row to the newest feature snapshot not from the future.

    `query_relation` must expose `entity_type`, `entity_id` and `as_of_ms`.
    """
    return f"""
        SELECT
            query.entity_type,
            query.entity_id,
            query.as_of_ms AS query_as_of_ms,
            feature.feature_name,
            feature.as_of_ms AS feature_as_of_ms,
            feature.value_double
        FROM {query_relation} AS query
        LEFT JOIN LATERAL (
            SELECT feature_name, as_of_ms, value_double
            FROM {features_relation} AS candidate
            WHERE candidate.entity_type = query.entity_type
              AND candidate.entity_id = query.entity_id
              AND candidate.as_of_ms <= query.as_of_ms
            ORDER BY candidate.as_of_ms DESC, candidate.feature_name
            LIMIT 256
        ) AS feature ON TRUE
        WHERE feature.feature_name IS NOT NULL
        ORDER BY query.entity_type, query.entity_id, query.as_of_ms, feature.feature_name
    """


AsOfRow = tuple[str, str, int | None, float | None]
"""`(entity_id, feature_name, feature_as_of_ms, value)` for one feature of one entity."""


def read_as_of(
    connection: duckdb.DuckDBPyConnection,
    features_path: Path,
    entity_type: str,
    entity_ids: list[str],
    as_of_ms: int,
) -> list[AsOfRow]:
    """Read features for entities at one prediction moment.

    `feature_as_of_ms` is None where the entity had no snapshot in time, which is the cold-start
    case rather than an error.
    """
    if not features_path.exists():
        msg = f"no as-of feature table at {features_path}"
        raise FileNotFoundError(msg)
    id_list = ", ".join(f"'{value.replace(chr(39), chr(39) * 2)}'" for value in entity_ids)
    features = f"read_parquet('{_quote(features_path)}')"
    query = f"""
        (SELECT '{entity_type}' AS entity_type, unnest([{id_list}]) AS entity_id, {as_of_ms} AS as_of_ms)
    """
    sql = as_of_join_sql(features, query)
    rows = connection.execute(sql).fetchall()
    return [
        (
            str(row[1]),
            str(row[3]),
            int(row[4]) if row[4] is not None else None,
            float(row[5]) if row[5] is not None else None,
        )
        for row in rows
    ]


def count_feature_leaks(
    connection: duckdb.DuckDBPyConnection,
    features_path: Path,
    interactions_relation: str,
) -> int:
    """Recompute every stored snapshot from raw interactions and count the mismatches.

    `interactions_relation` is any SQL relation with `parent_asin` and `timestamp` columns, so the
    same check runs during a build against the in-memory table and in a test against a fixture.
    Zero mismatches is the only acceptable answer: a non-zero count means some snapshot used data
    that did not exist at its own `as_of_ms`.
    """
    if not features_path.exists():
        msg = f"no as-of feature table at {features_path}"
        raise FileNotFoundError(msg)
    day_ms = 86_400_000
    rows = connection.execute(
        f"""
        WITH stored AS (
            SELECT entity_id, feature_name, as_of_ms, value_double
            FROM read_parquet('{_quote(features_path)}')
        ),
        bounds AS (SELECT DISTINCT as_of_ms FROM stored),
        truth AS (
            SELECT
                bounds.as_of_ms,
                events.parent_asin AS entity_id,
                count(*)::DOUBLE AS interactions,
                min(events.timestamp) AS first_ts,
                max(events.timestamp) AS last_ts
            FROM bounds
            JOIN {interactions_relation} AS events ON events.timestamp < bounds.as_of_ms
            GROUP BY bounds.as_of_ms, events.parent_asin
        )
        SELECT count(*)
        FROM stored
        JOIN truth ON truth.entity_id = stored.entity_id AND truth.as_of_ms = stored.as_of_ms
        WHERE CASE stored.feature_name
            WHEN 'item_interactions' THEN truth.interactions
            WHEN 'item_days_since_first_interaction'
                THEN (stored.as_of_ms - truth.first_ts)::DOUBLE / {day_ms}
            WHEN 'item_days_since_last_interaction'
                THEN (stored.as_of_ms - truth.last_ts)::DOUBLE / {day_ms}
            ELSE stored.value_double
        END IS DISTINCT FROM stored.value_double
        """
    ).fetchone()
    assert rows is not None  # noqa: S101 - aggregate always returns one row
    return int(rows[0] or 0)
