"""Two-tower on the synthetic corpus: trained, served and scored by the shared harness.

The point of these tests is not that the model learns (five users cannot), but that it plugs into
the same `CandidateSource` contract as the baselines, is served on the same cohort, and never
recommends an item the user has already seen.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pytest

from rankstack.data.encode import build_cohort
from rankstack.eval.harness import EvalConfig, compare_sources, evaluate_source
from rankstack.retrieval.base import exact_rank, top_k
from rankstack.retrieval.two_tower import (
    TwoTowerConfig,
    item_log_popularity,
    pad_histories,
    train_two_tower,
    write_model,
)
from tests.conftest import Trained, history_for, popularity_counts

pytestmark = pytest.mark.integration


@pytest.fixture
def with_two_tower(trained: Trained) -> Trained:
    """Fit a small two-tower on the same history the co-occurrence model used."""
    log_q = item_log_popularity(trained.cooccurrence.history_codes, trained.encoding.n_items)
    model, _ = train_two_tower(
        trained.cooccurrence.history_indptr,
        trained.cooccurrence.history_codes,
        trained.encoding.n_users,
        trained.encoding.n_items,
        log_q,
        TwoTowerConfig(
            factors=8,
            batch_size=32,
            epochs=10,
            learning_rate=0.05,
            min_user_history=2,
            seed=trained.settings.seed,
        ),
    )
    write_model(model, trained.settings.artifact_root / trained.spec.name / "two_tower.npz")
    return Trained(
        spec=trained.spec,
        settings=trained.settings,
        synthetic=trained.synthetic,
        processed=trained.processed,
        encoding=trained.encoding,
        boundaries=trained.boundaries,
        popularity=trained.popularity,
        cooccurrence=trained.cooccurrence,
        als=trained.als,
        two_tower=model,
    )


def test_two_tower_scores_every_catalogue_item(with_two_tower: Trained) -> None:
    model = with_two_tower.two_tower
    assert model is not None
    seen = np.zeros(model.n_items(), dtype=bool)
    scores = model.score(0, seen)
    assert scores.shape == (model.n_items(),)
    assert np.isfinite(scores).any()


def test_two_tower_masks_items_the_user_already_saw(with_two_tower: Trained) -> None:
    model = with_two_tower.two_tower
    assert model is not None
    user = next(
        code for code in range(with_two_tower.encoding.n_users) if model.history_length(code) >= 2
    )
    seen = np.zeros(model.n_items(), dtype=bool)
    seen[model.history(user)] = True
    scores = model.score(user, seen)
    assert not np.isfinite(scores[seen]).any()


def test_two_tower_rank_agrees_with_its_own_top_k(with_two_tower: Trained) -> None:
    """A hit rate that contradicts the model's own ranking would make every metric suspect."""
    model = with_two_tower.two_tower
    assert model is not None
    seen = np.zeros(model.n_items(), dtype=bool)
    seen[model.history(0)] = True
    scores = model.score(0, seen)
    for target in range(model.n_items()):
        if seen[target]:
            assert exact_rank(scores, target) == -1
            continue
        rank = exact_rank(scores, target)
        codes, _ = top_k(scores, 3)
        assert (target in set(codes.tolist())) == (0 < rank <= 3)


def test_two_tower_evaluates_through_the_shared_harness(with_two_tower: Trained) -> None:
    """Same cohort, same metrics, same harness as every baseline."""
    model = with_two_tower.two_tower
    assert model is not None
    connection = duckdb.connect(database=":memory:")
    cohort = build_cohort(
        connection,
        "all_test_users",
        with_two_tower.encoding,
        with_two_tower.processed,
        size=100,
        seed=7,
    )
    connection.close()

    users = np.asarray(cohort.user_codes, dtype=np.int64)
    targets = np.asarray(cohort.target_codes, dtype=np.int64)
    history = history_for(with_two_tower, popularity_counts(with_two_tower))
    config = EvalConfig(name="two_tower", cohort_name="all_test_users")

    result = evaluate_source(model, users, targets, history, config)
    assert result.source == "two_tower"
    assert result.ranking.users == len(cohort)
    assert 0.0 <= result.ranking.mean("hr")["hr@20"] <= 1.0
    assert "segment_sizes" in result.segments

    popularity = evaluate_source(with_two_tower.popularity, users, targets, history, config)
    comparison = compare_sources(result, popularity, resamples=100, seed=1)
    entry = comparison["hr"]["hr@20"]
    assert isinstance(entry, dict)
    assert entry["n"] == len(cohort)


def test_two_tower_cannot_serve_a_user_with_no_history(with_two_tower: Trained) -> None:
    """The cold-start boundary still holds: no history means no collaborative answer."""
    model = with_two_tower.two_tower
    assert model is not None
    seen = np.zeros(model.n_items(), dtype=bool)
    scores = model.score(with_two_tower.encoding.n_users - 1, seen)
    history_length = model.history_length(with_two_tower.encoding.n_users - 1)
    if history_length == 0:
        assert not np.isfinite(scores).any()
        assert model.can_serve(with_two_tower.encoding.n_users - 1) is False
    else:
        assert np.isfinite(scores).any()


def test_training_holdout_is_excluded_from_its_own_input(with_two_tower: Trained) -> None:
    """Leave-one-out: the target is the most recent train item, and the input starts after it.

    This is the guarantee that matters. A cohort target may legitimately also appear in the user's
    train history (Amazon reviews repeat products), so the eval cohort is not where self-leakage
    could occur; the training target is.
    """
    model = with_two_tower.two_tower
    assert model is not None
    indptr = model.history_indptr
    codes = model.history_codes
    depths = np.diff(indptr)
    eligible = np.flatnonzero(depths >= 2)
    padded, row_lengths = pad_histories(
        codes, indptr[eligible] + 1, depths[eligible] - 1, width_limit=model.max_history
    )
    for row, user_code in enumerate(eligible):
        target = codes[indptr[user_code]]
        # Only real positions count: the padding value is 0, which is also a valid item code.
        real = padded[row, : row_lengths[row]]
        assert target not in set(real.tolist())


def test_two_tower_artefact_is_written_under_the_corpus_directory(
    with_two_tower: Trained,
) -> None:
    path = Path(with_two_tower.settings.artifact_root) / with_two_tower.spec.name / "two_tower.npz"
    assert path.exists()
    assert path.with_suffix(".json").exists()


def test_two_tower_and_baselines_share_one_code_space(with_two_tower: Trained) -> None:
    """Different models on different code spaces would make the paired comparison meaningless."""
    model = with_two_tower.two_tower
    assert model is not None
    assert model.n_items() == with_two_tower.encoding.n_items
    assert model.history_codes.size == with_two_tower.cooccurrence.history_codes.size
    assert np.array_equal(model.history_indptr, with_two_tower.cooccurrence.history_indptr)


def test_two_tower_is_not_evaluated_against_a_test_only_item(with_two_tower: Trained) -> None:
    """A test-only item has no code, so it can never be a cohort target."""
    assert with_two_tower.synthetic.new_item_asin not in with_two_tower.encoding.item_to_code
    connection = duckdb.connect(database=":memory:")
    cohort = build_cohort(
        connection,
        "all_test_users",
        with_two_tower.encoding,
        with_two_tower.processed,
        size=100,
        seed=11,
    )
    connection.close()
    assert max(cohort.target_codes, default=-1) < with_two_tower.encoding.n_items
