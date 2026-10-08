"""The corpus registry must describe files that actually exist."""

from __future__ import annotations

import pytest

from rankstack.data.corpus import (
    CORPORA,
    DEFAULT_CORPUS,
    MAX_TIMESTAMP_MS,
    MIN_TIMESTAMP_MS,
    TEST_START_MS,
    TRAIN_END_MS,
    Corpus,
    get_corpus,
    publisher_issued_range,
    utc_iso,
)


def test_default_corpus_is_registered_and_primary() -> None:
    corpus = get_corpus(DEFAULT_CORPUS)
    assert corpus.is_primary
    assert corpus.core == 0
    assert corpus.category == "Electronics"


def test_unknown_corpus_names_fail_loudly() -> None:
    with pytest.raises(KeyError, match="known corpora"):
        get_corpus("books_0core")


def test_every_corpus_has_coherent_expectations() -> None:
    for name, corpus in CORPORA.items():
        assert corpus.interactions_url.startswith("https://"), name
        assert corpus.expected_interactions >= corpus.expected_users, name
        assert corpus.expected_users > 0, name
        assert corpus.expected_items > 0, name
        assert corpus.interactions_bytes > 0, name
        assert corpus.train_end_ms < corpus.test_start_ms, name


def test_metadata_is_declared_exactly_when_a_file_exists() -> None:
    for corpus in CORPORA.values():
        assert (corpus.metadata_url is None) == (corpus.metadata_bytes is None), corpus.name
        if corpus.metadata_url:
            assert corpus.metadata_url.endswith("jsonl.gz"), corpus.name


def test_only_the_primary_corpus_publishes_split_expectations() -> None:
    with_split = [c.name for c in CORPORA.values() if c.split is not None]
    assert with_split == ["electronics_0core"]


def test_boundaries_are_the_publishers_published_values() -> None:
    assert TRAIN_END_MS == 1628643414042
    assert TEST_START_MS == 1658002729837
    assert utc_iso(TRAIN_END_MS) == "2021-08-11T00:56:54.042000+00:00"
    assert utc_iso(TEST_START_MS) == "2022-07-16T20:18:49.837000+00:00"


def test_corpus_urls_point_at_the_publisher_not_a_mirror() -> None:
    corpus = get_corpus("electronics_0core")
    assert "benchmark/0core/rating_only/Electronics.csv.gz" in corpus.interactions_url
    assert "meta_categories/meta_Electronics.jsonl.gz" in (corpus.metadata_url or "")
    assert "timestamp_w_his/Electronics.train.csv.gz" in corpus.publisher_split_url("train")


def test_publisher_issued_range_covers_1996_to_2023() -> None:
    assert publisher_issued_range().startswith("1996-01-01")
    assert publisher_issued_range().endswith("2023-11-01T00:00:00+00:00")
    assert MIN_TIMESTAMP_MS < TRAIN_END_MS < TEST_START_MS < MAX_TIMESTAMP_MS


def test_corpus_is_hashable_and_comparable() -> None:
    a = get_corpus("electronics_5core")
    b = get_corpus("electronics_5core")
    assert a == b
    assert {a: "x"}[b] == "x"
    assert isinstance(a, Corpus)
