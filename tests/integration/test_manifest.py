"""Build outputs that humans and CI read back: the summary and the manifest."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rankstack.data.asof import read_as_of
from rankstack.data.build import build_dataset, load_manifest, summarise
from rankstack.data.corpus import Corpus
from rankstack.data.download import download_corpus_files_from, sha256_file
from tests.integration.test_pipeline import synthetic_corpus_spec

if TYPE_CHECKING:
    import duckdb

    from rankstack.config import Settings
    from tests.conftest import SyntheticCorpus


def test_summary_reports_every_partition_and_the_validation_outcome(
    synthetic_corpus: SyntheticCorpus, test_settings: Settings
) -> None:
    spec = synthetic_corpus_spec(synthetic_corpus)
    manifest = build_dataset(spec, settings=test_settings, skip_download=True)
    summary = summarise(manifest)
    assert spec.name in summary
    for partition in ("train", "valid", "test"):
        assert partition in summary
    assert "validation checks=" in summary
    assert "outputs=" in summary


def test_manifest_round_trips_through_disk(
    synthetic_corpus: SyntheticCorpus, test_settings: Settings
) -> None:
    spec = synthetic_corpus_spec(synthetic_corpus)
    build_dataset(spec, settings=test_settings, skip_download=True)
    payload = load_manifest(spec.name, settings=test_settings)
    assert payload["corpus"] == spec.name
    partitions = payload["partitions"]
    assert isinstance(partitions, list)
    assert {entry["partition"] for entry in partitions} == {"train", "valid", "test"}


def test_loading_a_manifest_that_was_never_written_fails_clearly(test_settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="run the build first"):
        load_manifest("electronics_0core", settings=test_settings)


def test_a_corpus_without_metadata_downloads_only_interactions(test_settings: Settings) -> None:
    """The transfer-check corpus has no metadata file, so only the interactions file is fetched."""
    category = "Musical_Instruments"
    source = test_settings.data_root / "amazon_2023" / f"{category.lower()}_5core"
    source.mkdir(parents=True)
    payload = b"user_id,parent_asin,rating,timestamp\nU1,B000000001,5.0,1650000000000\n"
    path = source / f"{category}.csv.gz"
    path.write_bytes(payload)

    transfer = Corpus(
        name="transfer_check",
        category=category,
        core=5,
        expected_users=1,
        expected_items=1,
        expected_interactions=1,
        interactions_bytes=len(payload),
        metadata_bytes=None,
    )
    assert transfer.metadata_url is None
    results = download_corpus_files_from(transfer, settings=test_settings)
    assert set(results) == {"interactions"}
    assert results["interactions"].path == path
    assert results["interactions"].sha256 == sha256_file(path)


def test_missing_raw_file_is_reported_before_any_work(
    synthetic_corpus: SyntheticCorpus, test_settings: Settings
) -> None:
    spec = synthetic_corpus_spec(synthetic_corpus)
    synthetic_corpus.csv_path.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        build_dataset(spec, settings=test_settings, skip_download=True)


def test_as_of_read_of_a_missing_table_fails_clearly(
    test_settings: Settings, connection: duckdb.DuckDBPyConnection
) -> None:
    with pytest.raises(FileNotFoundError, match="as-of feature table"):
        read_as_of(
            connection, test_settings.processed("missing") / "features.parquet", "item", ["B1"], 1
        )
