"""Integration tests against a real DuckDB database and the publisher's on-disk file formats."""

from __future__ import annotations

import gzip
from typing import TYPE_CHECKING, cast

import duckdb
import pytest

from rankstack.config import Settings
from rankstack.data.build import _stage_interactions, _stage_items, build_dataset
from rankstack.data.corpus import Corpus, get_corpus
from rankstack.data.download import DownloadError, download_file, sha256_file
from tests.conftest import SyntheticCorpus, make_synthetic_corpus, raw_dir_for
from tests.helpers import row, scalar

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


def synthetic_corpus_spec(corpus: SyntheticCorpus, name: str = "synthetic") -> Corpus:
    """Return a `Corpus` whose published expectations match the generated fixture."""
    return Corpus(
        name=name,
        category="Electronics",
        core=0,
        expected_users=corpus.users,
        expected_items=corpus.items + 1,
        expected_interactions=corpus.interactions,
        interactions_bytes=corpus.csv_path.stat().st_size,
        metadata_bytes=corpus.metadata_path.stat().st_size if corpus.metadata_path else None,
        is_primary=True,
    )


def test_reads_the_publisher_csv_format(
    connection: duckdb.DuckDBPyConnection, synthetic_corpus: SyntheticCorpus
) -> None:
    _stage_interactions(connection, synthetic_corpus.csv_path)
    first = row(connection, "SELECT user_id, parent_asin, rating, timestamp FROM staged LIMIT 1")
    assert float(first[2]) > 0.0
    assert 820_454_400_000 <= first[3] <= 1_698_796_800_000


def test_a_malformed_row_fails_the_reader(
    connection: duckdb.DuckDBPyConnection,
    tmp_path: Path,
) -> None:
    """Staging is lazy, so the reader rejects a bad value when the column is first read."""
    bad = tmp_path / "bad.csv.gz"
    bad.write_bytes(
        gzip.compress(
            b"user_id,parent_asin,rating,timestamp\nU1,B000000001,not_a_number,1650000000000\n"
        )
    )
    _stage_interactions(connection, bad)
    with pytest.raises(duckdb.Error, match="Conversion Error"):
        connection.execute("SELECT rating FROM staged").fetchone()


def test_reads_the_publisher_metadata_format(
    connection: duckdb.DuckDBPyConnection, synthetic_corpus: SyntheticCorpus
) -> None:
    assert synthetic_corpus.metadata_path is not None
    _stage_items(connection, synthetic_corpus.metadata_path)
    first = row(
        connection, "SELECT title, category_path, price, description FROM staged_items LIMIT 1"
    )
    assert first[1] == "Electronics > Accessories"
    assert float(first[2]) > 0.0


def test_title_falls_back_to_main_category(
    connection: duckdb.DuckDBPyConnection, synthetic_corpus: SyntheticCorpus
) -> None:
    assert synthetic_corpus.metadata_path is not None
    _stage_items(connection, synthetic_corpus.metadata_path)
    assert synthetic_corpus.metadata_path is not None
    _stage_items(connection, synthetic_corpus.metadata_path)
    found = row(
        connection,
        "SELECT title FROM staged_items WHERE parent_asin = ?",
        [synthetic_corpus.empty_title_asin],
    )
    assert found[0] == "Electronics"


def test_full_build_writes_split_parquet_and_a_manifest(
    synthetic_corpus: SyntheticCorpus, test_settings: Settings
) -> None:
    spec = synthetic_corpus_spec(synthetic_corpus)
    manifest = build_dataset(spec, settings=test_settings, skip_download=True)

    out_dir = test_settings.processed(spec.name)
    for partition in ("train", "valid", "test"):
        assert (out_dir / "interactions" / f"{partition}.parquet").exists()
    assert (out_dir / "items" / "items.parquet").exists()
    assert (out_dir / "manifest.json").exists()

    counts = {partition.rows for partition in manifest.partitions}
    assert sum(counts) == synthetic_corpus.interactions - 1  # the deliberate duplicate is dropped
    assert manifest.report is not None
    assert manifest.report.rows_dropped == 1  # the deliberate duplicate

    train_parquet = (out_dir / "interactions" / "train.parquet").as_posix()
    connection = duckdb.connect(database=":memory:")
    written = scalar(connection, f"SELECT count(*) FROM read_parquet('{train_parquet}')")
    connection.close()
    train_rows = next(p.rows for p in manifest.partitions if p.partition == "train")
    assert written == train_rows


def test_build_is_deterministic(synthetic_corpus: SyntheticCorpus, test_settings: Settings) -> None:
    spec = synthetic_corpus_spec(synthetic_corpus)
    first = build_dataset(spec, settings=test_settings, skip_download=True)
    hashes_first = {entry["path"]: entry["sha256"] for entry in first.outputs}

    second_settings = Settings(
        data_root=test_settings.data_root.with_name("data2"),
        artifact_root=test_settings.artifact_root,
        results_root=test_settings.results_root,
    )
    make_synthetic_corpus(raw_dir_for(second_settings))
    second = build_dataset(spec, settings=second_settings, skip_download=True)
    hashes_second = {entry["path"]: entry["sha256"] for entry in second.outputs}

    assert hashes_first == hashes_second


def test_manifest_records_enough_to_reproduce_a_build(synthetic_corpus, test_settings) -> None:  # type: ignore[no-untyped-def]
    spec = synthetic_corpus_spec(synthetic_corpus)
    manifest = build_dataset(spec, settings=test_settings, skip_download=True).as_dict()
    for key in ("corpus", "seed", "boundaries", "sources", "environment", "outputs", "validation"):
        assert key in manifest, key
    environment = cast("dict[str, object]", manifest["environment"])
    boundaries = cast("dict[str, int]", manifest["boundaries"])
    outputs = cast("list[dict[str, str]]", manifest["outputs"])
    assert environment["duckdb"]
    assert boundaries["train_end_ms"] < boundaries["test_start_ms"]
    assert all(entry["sha256"] for entry in outputs)


def test_download_verifies_size_and_reuses_an_existing_file(tmp_path: Path) -> None:
    source = tmp_path / "source.csv.gz"
    source.write_bytes(b"user_id,parent_asin,rating,timestamp\nU1,B1,5.0,1650000000000\n")
    destination = tmp_path / "nested" / "copy.csv.gz"

    result = download_file(
        url=source.as_uri(), destination=destination, expected_bytes=source.stat().st_size
    )
    assert result.bytes_written == source.stat().st_size
    assert result.sha256 == sha256_file(source)
    assert not result.already_present

    again = download_file(
        url=source.as_uri(), destination=destination, expected_bytes=source.stat().st_size
    )
    assert again.already_present


def test_download_rejects_a_size_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.csv.gz"
    source.write_bytes(b"payload")
    with pytest.raises(DownloadError, match="published size"):
        download_file(
            url=source.as_uri(),
            destination=tmp_path / "copy.gz",
            expected_bytes=source.stat().st_size + 100,
        )
    assert not (tmp_path / "copy.gz.partial").exists()


def test_download_rejects_a_non_http_scheme(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="not allowed"):
        download_file(url="ftp://example.com/x.gz", destination=tmp_path / "x.gz", expected_bytes=1)


def test_corpora_registry_urls_are_the_documented_ones() -> None:
    corpus = get_corpus("electronics_0core")
    assert corpus.interactions_bytes == 969_766_940
    assert corpus.metadata_bytes == 1_312_900_427
    assert get_corpus("electronics_5core").interactions_bytes == 257_671_078
    assert get_corpus("musical_instruments_5core").interactions_bytes == 8_437_657
