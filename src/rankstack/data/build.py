"""DuckDB pipeline: raw publisher files to validated, time-split Parquet.

One engine, one statement per concern, and a manifest that records enough for the build to be
repeated and audited: source hashes, row counts per partition, validation outcomes, output hashes,
seed, tool versions and the platform it ran on.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb

from rankstack.config import Settings
from rankstack.config import settings as default_settings
from rankstack.data.corpus import DEFAULT_CORPUS, Corpus, get_corpus, raw_dir_for, utc_iso
from rankstack.data.download import download_corpus_files, sha256_file
from rankstack.data.schema import (
    DESCRIPTION_MAX_CHARS,
    INTERACTION_SOURCE_COLUMNS,
    Check,
    CheckKind,
    CheckResult,
    ValidationReport,
    duplicate_check,
    interaction_checks,
    item_checks,
    report_checks,
)
from rankstack.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

_logger = get_logger(__name__)

PARTITIONS: tuple[str, ...] = ("train", "valid", "test")

SOURCE_COLUMNS_SQL = ", ".join(
    f"'{name}': '{sql_type}'" for name, sql_type in INTERACTION_SOURCE_COLUMNS.items()
)


@dataclass(frozen=True, slots=True)
class PartitionStats:
    """Row counts and observed timestamp range for one split partition."""

    partition: str
    rows: int
    users: int
    items: int
    min_timestamp_ms: int | None
    max_timestamp_ms: int | None

    def as_dict(self) -> dict[str, object]:
        """Return the counters as a JSON-serialisable mapping."""
        return asdict(self)


@dataclass(slots=True)
class BuildManifest:
    """Everything needed to repeat and audit a build."""

    corpus: str
    category: str
    core: int
    generated_at: str
    seed: int
    boundaries: dict[str, int]
    sources: dict[str, dict[str, object]] = field(default_factory=dict)
    partitions: list[PartitionStats] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    items: dict[str, object] = field(default_factory=dict)
    features: dict[str, object] = field(default_factory=dict)
    outputs: list[dict[str, object]] = field(default_factory=list)
    validation: dict[str, object] = field(default_factory=dict)
    environment: dict[str, object] = field(default_factory=dict)
    report: ValidationReport | None = field(default=None, repr=False, compare=False)
    """Typed copy of `validation`, for summaries and tests. Not serialised."""

    def as_dict(self) -> dict[str, object]:
        """Return the manifest as a JSON-serialisable mapping."""
        return {
            "corpus": self.corpus,
            "category": self.category,
            "core": self.core,
            "generated_at": self.generated_at,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "seed": self.seed,
            "boundaries": self.boundaries,
            "sources": self.sources,
            "partitions": [p.as_dict() for p in self.partitions],
            "items": self.items,
            "features": self.features,
            "outputs": self.outputs,
            "validation": self.validation,
            "environment": self.environment,
        }


def _connection() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(database=":memory:")
    connection.execute("SET threads TO 4")
    return connection


def _quote(path: Path) -> str:
    return str(path).replace("'", "''")


def _stage_interactions(connection: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    """Expose the publisher's gzip CSV as a typed view. A malformed row fails here, loudly."""
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW staged AS
        SELECT
            user_id,
            parent_asin,
            rating,
            timestamp
        FROM read_csv(
            '{_quote(csv_path)}',
            header = true,
            columns = {{{SOURCE_COLUMNS_SQL}}},
            timestampformat = '%Y-%m-%d %H:%M:%S'
        )
        """
    )


def _stage_items(connection: duckdb.DuckDBPyConnection, jsonl_path: Path) -> None:
    """Expose item metadata as a typed view with title, category path, price and description."""
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW staged_items AS
        SELECT
            parent_asin,
            coalesce(
                nullif(trim(title), ''), nullif(trim(main_category), ''), parent_asin
            ) AS title,
            main_category,
            coalesce(array_to_string(categories, ' > '), '') AS category_path,
            -- `price` is heterogeneous upstream: 526,449 numbers, 1,083,247 nulls and 316 strings
            -- ('from 9.95', and 266 rows whose price is an em dash). Take the first number in the
            -- string, or null when there is none. Never coerce a missing price to zero.
            try_cast(nullif(regexp_extract(source.price, '[0-9]+(\\.[0-9]+)?', 1), '') AS DOUBLE)
                AS price,
            store,
            left(coalesce(array_to_string(description, ' '), ''), {DESCRIPTION_MAX_CHARS})
                AS description
        FROM read_json(
            '{_quote(jsonl_path)}',
            format = 'newline_delimited',
            maximum_object_size = 33554432,
            columns = {{
                'parent_asin': 'VARCHAR',
                'title': 'VARCHAR',
                'main_category': 'VARCHAR',
                'categories': 'VARCHAR[]',
                'price': 'VARCHAR',
                'store': 'VARCHAR',
                'description': 'VARCHAR[]'
            }}
        ) AS source
        """
    )


def _run_checks(
    connection: duckdb.DuckDBPyConnection, checks: list[Check], report: ValidationReport
) -> dict[str, float]:
    observed: dict[str, float] = {}
    for check in checks:
        value = connection.execute(check.sql).fetchone()
        measurement = float(value[0]) if value is not None else 0.0
        evaluation = report.add(check, measurement)
        observed[check.name] = measurement
        _logger.info(
            "data.check",
            extra={
                "check": check.name,
                "kind": str(check.kind),
                "observed": measurement,
                "expected": check.expected,
                "passed": evaluation.result.passed,
            },
        )
    return observed


def _clean_interactions(connection: duckdb.DuckDBPyConnection) -> None:
    """Drop malformed and duplicate rows, keeping one event per (user, item, millisecond)."""
    connection.execute(
        """
        CREATE OR REPLACE TABLE clean_interactions AS
        SELECT user_id, parent_asin, rating, timestamp
        FROM staged
        WHERE rating IS NOT NULL AND rating >= 1.0 AND rating <= 5.0
        QUALIFY row_number() OVER (
            PARTITION BY user_id, parent_asin, timestamp ORDER BY rating DESC
        ) = 1
        """
    )
    connection.execute("CREATE OR REPLACE VIEW staged AS SELECT * FROM clean_interactions")


def _write_partitions(
    connection: duckdb.DuckDBPyConnection,
    out_dir: Path,
    train_end_ms: int,
    test_start_ms: int,
) -> list[PartitionStats]:
    stats: list[PartitionStats] = []
    for partition in PARTITIONS:
        predicate = {
            "train": f"timestamp < {train_end_ms}",
            "valid": f"timestamp >= {train_end_ms} AND timestamp < {test_start_ms}",
            "test": f"timestamp >= {test_start_ms}",
        }[partition]
        target = out_dir / "interactions" / f"{partition}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        connection.execute(
            f"""
            COPY (
                SELECT
                    user_id,
                    parent_asin,
                    rating::FLOAT AS rating,
                    timestamp,
                    to_timestamp(timestamp / 1000.0) AS event_time
                FROM clean_interactions
                WHERE {predicate}
                ORDER BY timestamp, user_id, parent_asin
            ) TO '{_quote(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        row = connection.execute(
            f"""
            SELECT
                count(*),
                count(DISTINCT user_id),
                count(DISTINCT parent_asin),
                min(timestamp),
                max(timestamp)
            FROM clean_interactions
            WHERE {predicate}
            """
        ).fetchone()
        assert row is not None  # noqa: S101 - aggregate always returns one row
        stats.append(
            PartitionStats(
                partition=partition,
                rows=int(row[0]),
                users=int(row[1]),
                items=int(row[2]),
                min_timestamp_ms=int(row[3]) if row[3] is not None else None,
                max_timestamp_ms=int(row[4]) if row[4] is not None else None,
            )
        )
        _logger.info(
            "data.partition_written", extra={"partition": partition, "rows": stats[-1].rows}
        )
    return stats


def _corpus_totals(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Distinct users and items across the whole corpus, not per partition.

    The per-partition counts in `PartitionStats` cannot be summed: a user active in both the train
    and test windows appears in both.
    """
    row = connection.execute(
        "SELECT count(DISTINCT user_id), count(DISTINCT parent_asin) FROM clean_interactions"
    ).fetchone()
    assert row is not None  # noqa: S101 - aggregate always returns one row
    return {"distinct_users": int(row[0]), "distinct_interacted_items": int(row[1])}


def _write_items(connection: duckdb.DuckDBPyConnection, out_dir: Path) -> dict[str, object]:
    target = out_dir / "items" / "items.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"""
        COPY (
            SELECT parent_asin, title, main_category, category_path, price, store, description
            FROM clean_items
            ORDER BY parent_asin
        ) TO '{_quote(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    row = connection.execute(
        """
        SELECT
            count(*),
            count(*) FILTER (WHERE price IS NOT NULL),
            count(*) FILTER (WHERE description <> ''),
            count(*) FILTER (WHERE category_path = ''),
            count(*) FILTER (WHERE main_category IS NULL)
        FROM clean_items
        """
    ).fetchone()
    assert row is not None  # noqa: S101
    return {
        "rows": int(row[0]),
        "rows_with_price": int(row[1]),
        "rows_with_description": int(row[2]),
        "rows_without_category_path": int(row[3]),
        "rows_without_main_category": int(row[4]),
    }


def build_dataset(
    corpus: str | Corpus | None = None,
    settings: Settings = default_settings,
    *,
    skip_download: bool = False,
) -> BuildManifest:
    """Run the full data build for one corpus and write its manifest.

    `corpus` is a registry name or a `Corpus` value, so a test can build a synthetic corpus that
    satisfies the same contract without the published row counts.
    """
    corpus_value = corpus if isinstance(corpus, Corpus) else get_corpus(corpus or DEFAULT_CORPUS)
    return _build(corpus_value, settings=settings, skip_download=skip_download)


def _build(corpus_value: Corpus, settings: Settings, *, skip_download: bool) -> BuildManifest:
    started = time.perf_counter()
    settings.ensure_directories()
    out_dir = settings.processed(corpus_value.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources: dict[str, dict[str, object]] = {}
    if not skip_download:
        for label, result in download_corpus_files(corpus_value.name, settings=settings).items():
            sources[label] = result.as_dict()

    raw_dir = raw_dir_for(settings.data_root, corpus_value)
    csv_path = raw_dir / f"{corpus_value.category}.csv.gz"
    if not csv_path.exists():
        msg = f"missing {csv_path}; run the download step first"
        raise FileNotFoundError(msg)

    connection = _connection()
    report = ValidationReport()
    try:
        _stage_interactions(connection, csv_path)
        _run_checks(
            connection,
            interaction_checks(corpus_value.name, corpus_value.expected_interactions),
            report,
        )
        report.raise_on_failure()

        _run_checks(connection, [duplicate_check()], report)
        _clean_interactions(connection)

        if corpus_value.metadata_url and corpus_value.metadata_bytes:
            _stage_items(connection, raw_dir / f"meta_{corpus_value.category}.jsonl.gz")
            _run_checks(connection, item_checks(0.9), report)
            report.raise_on_failure()
            _run_checks(connection, report_checks(), report)
            connection.execute(
                """
                CREATE OR REPLACE TABLE clean_items AS
                SELECT * FROM staged_items QUALIFY row_number() OVER (
                    PARTITION BY parent_asin ORDER BY parent_asin
                ) = 1
                """
            )
            item_stats = _write_items(connection, out_dir)
            metadata_path = raw_dir / f"meta_{corpus_value.category}.jsonl.gz"
            sources.setdefault("metadata_sha256", {"sha256": sha256_file(metadata_path)})
        else:
            item_stats = {"rows": 0, "note": "corpus has no item metadata file"}

        sources.setdefault("interactions_sha256", {"sha256": sha256_file(csv_path)})

        partition_stats = _write_partitions(
            connection, out_dir, corpus_value.train_end_ms, corpus_value.test_start_ms
        )
        totals = _corpus_totals(connection)

        from rankstack.data.asof import write_item_as_of_features  # noqa: PLC0415 - import cycle

        feature_stats = write_item_as_of_features(
            connection=connection,
            out_dir=out_dir,
            boundaries={"t1": corpus_value.train_end_ms, "t2": corpus_value.test_start_ms},
        )
    finally:
        connection.close()

    manifest = BuildManifest(
        corpus=corpus_value.name,
        category=corpus_value.category,
        core=corpus_value.core,
        generated_at=datetime.now(tz=UTC).isoformat(),
        seed=settings.seed,
        boundaries={
            "train_end_ms": corpus_value.train_end_ms,
            "test_start_ms": corpus_value.test_start_ms,
        },
        sources=sources,
        partitions=partition_stats,
        items=item_stats,
        features=feature_stats,
        validation=report.as_dict(),
        environment={
            **_environment(),
            **totals,
            "distinct_items": item_stats.get("rows", 0),
        },
        report=report,
    )
    manifest.elapsed_seconds = time.perf_counter() - started
    manifest.outputs = _hash_outputs(out_dir)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n")
    _logger.info(
        "data.build_complete",
        extra={"corpus": corpus_value.name, "manifest": str(manifest_path)},
    )
    return manifest


def _as_int(value: object, default: int = 0) -> int:
    """Read an integer out of a `dict[str, object]` without asserting its type."""
    return value if isinstance(value, int) else default


def result_summary(manifest: BuildManifest, command: str) -> dict[str, object]:
    """Build the committed result file that backs every published figure.

    Same numbers as the manifest, keyed so a document can point at a single value
    (`results/dataset_<corpus>.json#/partitions/train/rows`).
    """
    partitions = {p.partition: p.as_dict() for p in manifest.partitions}
    users = _as_int(manifest.environment.get("distinct_users"))
    interacted_items = _as_int(manifest.environment.get("distinct_interacted_items"))
    totals = {
        "interactions": sum(p.rows for p in manifest.partitions),
        "users": users,
        "interacted_items": interacted_items,
        "items_with_metadata": _as_int(manifest.environment.get("distinct_items")),
    }
    train = next(p for p in manifest.partitions if p.partition == "train")
    return {
        "corpus": manifest.corpus,
        "category": manifest.category,
        "core": manifest.core,
        "command": command,
        "generated_at": manifest.generated_at,
        "elapsed_seconds": manifest.elapsed_seconds,
        "seed": manifest.seed,
        "boundaries": manifest.boundaries,
        "boundary_iso": {
            "train_end": utc_iso(manifest.boundaries["train_end_ms"]),
            "test_start": utc_iso(manifest.boundaries["test_start_ms"]),
        },
        "totals": totals,
        "partitions": partitions,
        "cold_start": {
            # Items the model has never seen at training time: present in the corpus but absent from
            # the train partition. Only the content path can reach these.
            "new_items": interacted_items - train.items,
            "new_users": users - train.users,
        },
        "items": manifest.items,
        "features": manifest.features,
        "sources": manifest.sources,
        "outputs": manifest.outputs,
        "validation": manifest.validation,
        "environment": manifest.environment,
    }


def write_result_file(manifest: BuildManifest, command: str, results_root: Path) -> Path:
    """Write `results/dataset_<corpus>.json`. Documentation figures point at this file."""
    results_root.mkdir(parents=True, exist_ok=True)
    path = results_root / f"dataset_{manifest.corpus}.json"
    payload = result_summary(manifest, command)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _logger.info("data.result_written", extra={"path": str(path)})
    return path


def _hash_outputs(out_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(out_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(out_dir.rglob("*.parquet"))
    ]


def _environment() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "duckdb": duckdb.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def summarise(manifest: BuildManifest) -> str:
    """Human-readable build summary, used by the script entry point and the PR body."""
    lines = [
        f"corpus: {manifest.corpus} (category={manifest.category}, k-core={manifest.core})",
        f"boundaries: train < {manifest.boundaries['train_end_ms']}, test >= "
        f"{manifest.boundaries['test_start_ms']}",
    ]
    for partition in manifest.partitions:
        window = (
            f"{partition.min_timestamp_ms}..{partition.max_timestamp_ms}"
            if partition.min_timestamp_ms is not None
            else "empty"
        )
        lines.append(
            f"  {partition.partition:5s} rows={partition.rows:>12,} users={partition.users:>10,} "
            f"items={partition.items:>10,} window={window}"
        )
    items = manifest.items
    rows = items.get("rows", 0)
    priced = items.get("rows_with_price", 0)
    lines.append(f"  items rows={rows:>12,} with_price={priced:,}")
    report = manifest.report
    checks = report.evaluations if report else []
    failures = report.failures if report else []
    lines.append(f"  validation checks={len(checks)} failing={len(failures)}")
    lines.append(f"  outputs={len(manifest.outputs)} files")
    return "\n".join(lines)


def load_manifest(corpus_name: str, settings: Settings = default_settings) -> dict[str, object]:
    """Read a previously written manifest. Results are always read back from disk, never cached."""
    path = settings.processed(corpus_name) / "manifest.json"
    if not path.exists():
        msg = f"no manifest at {path}; run the build first"
        raise FileNotFoundError(msg)
    payload: dict[str, object] = json.loads(path.read_text())
    return payload


__all__ = [
    "BuildManifest",
    "CheckKind",
    "CheckResult",
    "PartitionStats",
    "build_dataset",
    "load_manifest",
    "result_summary",
    "summarise",
    "write_result_file",
]
