"""Baselines on a synthetic corpus: encoding, training, evaluation and the cold-start boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pytest

from rankstack.data.build import BuildManifest, build_dataset
from rankstack.data.encode import build_cohort, build_encoding, load_encoding, write_encoding
from rankstack.eval.harness import CohortHistory, EvalConfig, compare_sources, evaluate_source
from rankstack.retrieval.als import AlsConfig, train_als
from rankstack.retrieval.base import exact_rank
from rankstack.retrieval.cooccurrence import build_history, train_cooccurrence
from rankstack.retrieval.popularity import train_popularity
from tests.integration.test_pipeline import synthetic_corpus_spec

if TYPE_CHECKING:
    from pathlib import Path

    from rankstack.config import Settings
    from rankstack.data.corpus import Corpus
    from rankstack.data.encode import Encoding
    from rankstack.retrieval.als import AlsModel
    from rankstack.retrieval.cooccurrence import CooccurrenceModel
    from rankstack.retrieval.popularity import PopularityModel
    from tests.conftest import SyntheticCorpus

pytestmark = pytest.mark.integration


def boundaries_of(manifest: BuildManifest) -> dict[str, int]:
    """Read the split boundaries a build recorded, as plain integers."""
    boundaries = manifest.boundaries
    return {
        "train_end_ms": int(boundaries["train_end_ms"]),
        "test_start_ms": int(boundaries["test_start_ms"]),
    }


@dataclass(frozen=True, slots=True)
class Trained:
    """Every artefact produced by one baseline training run on the synthetic corpus."""

    spec: Corpus
    settings: Settings
    synthetic: SyntheticCorpus
    processed: Path
    encoding: Encoding
    boundaries: dict[str, int]
    popularity: PopularityModel
    cooccurrence: CooccurrenceModel
    als: AlsModel


@pytest.fixture
def trained(synthetic_corpus: SyntheticCorpus, test_settings: Settings) -> Trained:
    """Build the synthetic corpus, then fit every baseline on its train partition."""
    spec = synthetic_corpus_spec(synthetic_corpus)
    manifest = build_dataset(spec, settings=test_settings, skip_download=True)
    processed = test_settings.processed(spec.name)
    train = processed / "interactions" / "train.parquet"

    boundaries = {
        "train_end_ms": int(boundaries_of(manifest)["train_end_ms"]),
        "test_start_ms": int(boundaries_of(manifest)["test_start_ms"]),
    }
    encoding = build_encoding(train, boundaries, seed=test_settings.seed)
    write_encoding(encoding, test_settings.artifact_root / spec.name)

    popularity = train_popularity(train, encoding.item_to_code, boundaries["train_end_ms"])
    cooccurrence = train_cooccurrence(train, encoding.user_to_code, encoding.item_to_code, top_k=10)
    als = train_als(
        cooccurrence.history_indptr,
        cooccurrence.history_codes,
        encoding.n_users,
        encoding.n_items,
        AlsConfig(factors=8, iterations=3, min_user_history=2, seed=test_settings.seed),
    )
    return Trained(
        spec=spec,
        settings=test_settings,
        synthetic=synthetic_corpus,
        processed=processed,
        encoding=encoding,
        boundaries=boundaries,
        popularity=popularity,
        cooccurrence=cooccurrence,
        als=als,
    )


def history_for(trained: Trained, counts: np.ndarray) -> CohortHistory:
    """Build the evaluation history from the co-occurrence model's capped train history."""
    return CohortHistory(
        indptr=trained.cooccurrence.history_indptr,
        codes=trained.cooccurrence.history_codes,
        n_items=trained.encoding.n_items,
        popularity=counts,
    )


def popularity_counts(trained: Trained) -> np.ndarray:
    """Train-window interaction count per item, used for novelty and target-popularity segments."""
    codes = trained.cooccurrence.history_codes
    return np.bincount(codes.astype(np.int64), minlength=trained.encoding.n_items).astype(
        np.float32
    )


def test_encoding_round_trips_through_disk(trained: Trained) -> None:
    written = trained.encoding
    reloaded = load_encoding(trained.settings.artifact_root, trained.spec.name)
    assert reloaded.user_to_code == written.user_to_code
    assert reloaded.item_to_code == written.item_to_code
    assert reloaded.n_users == written.n_users
    assert reloaded.n_items == written.n_items


def test_encoding_excludes_an_item_that_appears_only_after_the_boundary(trained: Trained) -> None:
    """The cold-start boundary, enforced by the code space itself."""
    encoding = trained.encoding
    new_item = trained.synthetic.new_item_asin
    assert new_item not in encoding.item_to_code
    assert trained.cooccurrence.n_items() == encoding.n_items


def test_popularity_serves_every_user_including_cold_start_ones(trained: Trained) -> None:
    popularity = trained.popularity
    seen = np.zeros(popularity.n_items(), dtype=bool)
    scores = popularity.score(0, seen)
    assert np.isfinite(scores).sum() == popularity.n_items()
    assert popularity.can_serve(999_999) is True


def test_a_user_with_no_train_history_is_unservable(trained: Trained) -> None:
    """A user first seen in the test window has no train history at all."""
    cold_user = trained.synthetic.test_only_user
    assert cold_user not in trained.encoding.user_to_code


def test_als_cannot_serve_a_user_with_one_interaction(trained: Trained) -> None:
    """One interaction gives a user factor no information, so ALS excludes them by design."""
    encoding = trained.encoding
    code = encoding.user_to_code[trained.synthetic.single_history_user]
    assert trained.cooccurrence.history_length(code) == 1
    assert trained.als.can_serve(code) is False
    assert not trained.als.user_factors[code].any()

    seen = np.zeros(encoding.n_items, dtype=bool)
    scores = trained.als.score(code, seen)
    assert not np.isfinite(scores).any()
    # Co-occurrence still has a neighbour list for that single item.
    assert trained.cooccurrence.can_serve(code) is True


def test_seen_items_are_never_recommended(trained: Trained) -> None:
    cooccurrence = trained.cooccurrence
    user = next(
        code for code in range(trained.encoding.n_users) if cooccurrence.history_length(code) >= 2
    )
    seen = np.zeros(trained.encoding.n_items, dtype=bool)
    seen[cooccurrence.history(user)] = True
    scores = cooccurrence.score(user, seen)
    assert not np.isfinite(scores[seen]).any()


def test_cooccurrence_is_cosine_normalised(trained: Trained) -> None:
    similarity = trained.cooccurrence.similarity
    assert similarity.shape[0] == similarity.shape[1] == trained.encoding.n_items
    assert similarity.nnz > 0
    # Cosine values cannot exceed 1.
    assert float(similarity.data.max()) <= 1.0 + 1e-6


def test_history_is_capped_and_ordered_by_recency(trained: Trained) -> None:
    cooccurrence = trained.cooccurrence
    assert cooccurrence.history_indptr.shape[0] == trained.encoding.n_users + 1
    assert cooccurrence.max_history == 50
    for code in range(min(trained.encoding.n_users, 20)):
        assert cooccurrence.history_length(code) <= cooccurrence.max_history


def test_build_history_returns_popularity_counts(trained: Trained) -> None:
    encoding = trained.encoding
    train = trained.processed / "interactions" / "train.parquet"
    indptr, codes, popularity = build_history(
        train, encoding.user_to_code, encoding.item_to_code, 50
    )
    assert indptr.shape[0] == encoding.n_users + 1
    assert codes.size == int(indptr[-1])
    assert popularity.sum() > 0
    assert popularity.shape[0] == encoding.n_items


def test_exact_rank_ranks_by_score_then_by_item_code() -> None:
    ascending = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert exact_rank(ascending, 2) == 1  # highest score ranks first
    assert exact_rank(ascending, 0) == 3

    # Four identical scores: ascending item code decides, so the last code comes fourth.
    tied = np.zeros(4, dtype=np.float32)
    assert exact_rank(tied, 3) == 4
    assert exact_rank(tied, 0) == 1


def test_exact_rank_ignores_masked_items_entirely() -> None:
    """Masked items are `-inf`, so they must not shift the rank of an unmasked target."""
    scores = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 9.0], dtype=np.float32)
    seen = np.zeros(6, dtype=bool)
    seen[[5]] = True
    masked = np.where(seen, -np.inf, scores)
    assert exact_rank(masked, 5) == -1  # a masked item has no finite rank
    assert exact_rank(masked, 2) == 2  # only item 3 (3.0) outranks it; item 5 is masked away
    assert exact_rank(masked, 0) == 4  # codes 1-3 outrank it; code 4 ties but ranks after


def test_evaluation_reports_segments_and_a_usable_cohort(trained: Trained) -> None:
    encoding = trained.encoding
    connection = duckdb.connect(database=":memory:")
    cohort = build_cohort(
        connection,
        "all_test_users",
        encoding,
        trained.processed,
        size=100,
        seed=7,
    )
    connection.close()
    assert len(cohort) > 0
    assert cohort.population_size >= len(cohort)

    users = np.asarray(cohort.user_codes, dtype=np.int64)
    targets = np.asarray(cohort.target_codes, dtype=np.int64)
    history = history_for(trained, popularity_counts(trained))
    result = evaluate_source(
        trained.cooccurrence,
        users,
        targets,
        history,
        EvalConfig(name="cooccurrence", cohort_name="all_test_users"),
    )
    payload = result.as_dict()
    assert "ranking" in payload
    assert "segments" in payload
    assert "segment_sizes" in result.segments
    assert result.ranking.users == len(cohort)
    assert 0.0 <= result.ranking.mean("hr")["hr@20"] <= 1.0
    assert result.ranking.mean("hr")["hr@10"] <= result.ranking.mean("hr")["hr@20"]
    assert result.ranking.mean("ndcg")["ndcg@10"] <= result.ranking.mean("ndcg")["ndcg@50"]
    assert "coverage_at_50" in result.beyond_list


def test_cohort_is_deterministic_for_a_given_seed(trained: Trained) -> None:
    encoding = trained.encoding
    connection = duckdb.connect(database=":memory:")
    first = build_cohort(connection, "warm_users", encoding, trained.processed, size=50, seed=3)
    second = build_cohort(connection, "warm_users", encoding, trained.processed, size=50, seed=3)
    other = build_cohort(connection, "warm_users", encoding, trained.processed, size=50, seed=4)
    connection.close()
    assert first.user_codes == second.user_codes
    assert first.target_codes == second.target_codes
    assert first.user_codes != other.user_codes or first.user_codes == []


def test_warm_users_cohort_excludes_users_without_enough_history(trained: Trained) -> None:
    encoding = trained.encoding
    connection = duckdb.connect(database=":memory:")
    cohort = build_cohort(connection, "warm_users", encoding, trained.processed, size=200, seed=5)
    connection.close()
    depths = [trained.cooccurrence.history_length(code) for code in cohort.user_codes]
    assert depths, "cohort should not be empty on the synthetic corpus"
    assert min(depths) >= 2


def test_paired_comparison_reports_an_interval_and_a_probability(trained: Trained) -> None:
    encoding = trained.encoding
    connection = duckdb.connect(database=":memory:")
    cohort = build_cohort(
        connection, "all_test_users", encoding, trained.processed, size=100, seed=1
    )
    connection.close()
    users = np.asarray(cohort.user_codes, dtype=np.int64)
    targets = np.asarray(cohort.target_codes, dtype=np.int64)
    history = history_for(trained, popularity_counts(trained))
    config = EvalConfig(name="x", cohort_name="all_test_users", ks=(10, 20))
    popularity = evaluate_source(trained.popularity, users, targets, history, config)
    cooccurrence = evaluate_source(trained.cooccurrence, users, targets, history, config)
    comparison = compare_sources(cooccurrence, popularity, resamples=100, seed=2)
    assert set(comparison) == {"recall", "hr", "ndcg", "mrr", "served_fraction"}
    entry = comparison["hr"]["hr@20"]
    assert isinstance(entry, dict)
    assert entry["ci_low"] <= entry["difference"] <= entry["ci_high"]
    probability = entry["bootstrap_probability_alternative_wins"]
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0
    assert entry["n"] == len(cohort)
