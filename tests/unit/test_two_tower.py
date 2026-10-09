"""Two-tower model invariants: pooling, normalisation, correction, artefacts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rankstack.retrieval.two_tower import (
    TwoTower,
    TwoTowerConfig,
    TwoTowerModel,
    item_log_popularity,
    load_model,
    recency_weights,
    train_two_tower,
    write_model,
)


def histories(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Build a CSR-style `(indptr, codes)` pair from a list of per-user item lists."""
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    for index, row in enumerate(rows):
        indptr[index + 1] = indptr[index] + len(row)
    codes = np.asarray([item for row in rows for item in row], dtype=np.int32)
    return indptr, codes


def test_recency_weights_decay_geometrically() -> None:
    weights = recency_weights(4, 0.5, torch.device("cpu")).numpy()
    assert weights == pytest.approx([1.0, 0.5, 0.25, 0.125], rel=1e-6)


def test_a_decay_of_one_gives_every_position_equal_weight() -> None:
    weights = recency_weights(3, 1.0, torch.device("cpu")).numpy()
    assert weights == pytest.approx([1.0, 1.0, 1.0])


def test_item_vectors_are_unit_length() -> None:
    module = TwoTower(n_items=5, config=TwoTowerConfig(factors=8))
    vectors = module.item_vectors(torch.tensor([0, 1, 2]))
    assert vectors.shape == (3, 8)
    assert torch.allclose(vectors.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_user_vector_is_unit_length_and_ignores_padding() -> None:
    module = TwoTower(n_items=10, config=TwoTowerConfig(factors=4, recency_decay=0.5))
    history = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
    lengths = torch.tensor([2, 1])
    vector = module.user_vector(history, lengths)
    assert vector.shape == (2, 4)
    assert torch.allclose(vector.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_padding_cannot_change_a_user_embedding() -> None:
    """A wide padded row and a narrow one must produce the same vector for the same real history."""
    module = TwoTower(n_items=10, config=TwoTowerConfig(factors=4, recency_decay=0.7))
    narrow = module.user_vector(torch.tensor([[5, 6]]), torch.tensor([2]))
    wide = module.user_vector(torch.tensor([[5, 6, 0, 0, 0]]), torch.tensor([2]))
    assert torch.allclose(narrow, wide, atol=1e-6)


def test_log_popularity_is_finite_for_items_with_no_interactions() -> None:
    log_q = item_log_popularity(np.array([0, 0, 1], dtype=np.int32), 5)
    assert log_q.shape == (5,)
    assert np.isfinite(log_q).all()


def test_log_popularity_is_lower_for_rarer_items() -> None:
    """`log q` is low for rare items, which is what shifts their logit up during training."""
    log_q = item_log_popularity(np.array([0] * 9 + [1], dtype=np.int32), 2)
    assert log_q[1] < log_q[0]
    assert log_q[1] == pytest.approx(np.log(0.1), rel=1e-6)


def test_training_reduces_the_loss_and_returns_a_usable_model() -> None:
    indptr, codes = histories([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 0, 1]])
    config = TwoTowerConfig(factors=8, batch_size=5, epochs=3, temperature=0.5, seed=1)
    model, report = train_two_tower(
        indptr,
        codes,
        n_users=5,
        n_items=5,
        item_log_popularity=item_log_popularity(codes, 5),
        config=config,
    )
    assert report.steps == 3  # three epochs over one batch each
    assert report.final_loss < 4.0  # ln(batch) is the loss of an untrained model
    assert model.n_items() == 5
    norms = np.linalg.norm(model.item_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_log_q_correction_changes_what_the_model_learns() -> None:
    """The ablation must actually differ, otherwise it is not measuring anything.

    Two properties of the fixture matter. Item 0 is in every user's history but in nobody's
    *holdout* position, which is what makes `log q` large for it; and each user's most recent item
    differs, so the in-batch diagonal is a learnable label rather than eight copies of one item.
    """
    rows = [[1, 0], [0, 2], [3, 0], [4, 0], [1, 2, 0], [3, 4, 0], [2, 3, 0], [1, 4, 0]]
    indptr, codes = histories(rows)
    log_q = item_log_popularity(codes, 5)
    targets = codes[indptr[:-1]]
    assert float(log_q[targets].max() - log_q[targets].min()) > 0.0, (
        "targets must differ in popularity for the per-column correction to do anything"
    )
    corrected, corrected_report = train_two_tower(
        indptr,
        codes,
        8,
        5,
        log_q,
        TwoTowerConfig(
            factors=8,
            batch_size=8,
            epochs=80,
            learning_rate=0.05,
            temperature=0.5,
            log_q_correction=True,
            seed=1,
        ),
    )
    plain, plain_report = train_two_tower(
        indptr,
        codes,
        8,
        5,
        log_q,
        TwoTowerConfig(
            factors=8,
            batch_size=8,
            epochs=80,
            learning_rate=0.05,
            temperature=0.5,
            log_q_correction=False,
            seed=1,
        ),
    )
    assert not np.allclose(corrected.item_embeddings, plain.item_embeddings, atol=1e-4)
    assert corrected_report.final_loss < 2.2, "the model should actually learn the holdout"
    assert abs(corrected_report.final_loss - plain_report.final_loss) > 1e-4


def test_training_refuses_when_nobody_has_enough_history() -> None:
    indptr, codes = histories([[0], [1], [2]])
    with pytest.raises(ValueError, match="no user has at least"):
        train_two_tower(
            indptr, codes, 3, 3, item_log_popularity(codes, 3), TwoTowerConfig(min_user_history=2)
        )


def test_numpy_user_vectors_match_the_torch_tower() -> None:
    """Serving must reproduce training's pooling exactly, or the scores are subtly wrong."""
    indptr, codes = histories([[0, 1, 2], [3, 4]])
    config = TwoTowerConfig(factors=6, recency_decay=0.8)
    torch.manual_seed(3)
    module = TwoTower(n_items=5, config=config)
    embeddings = module.item_embedding.weight.detach().numpy()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    model = TwoTowerModel(
        item_embeddings=embeddings / norms,
        history_indptr=indptr,
        history_codes=codes,
        recency_decay=config.recency_decay,
        max_history=config.max_history,
        config=config,
    )
    served = model.user_vectors(np.array([0, 1]))
    with torch.no_grad():
        expected = module.user_vector(
            torch.tensor([[0, 1, 2], [3, 4, 0]]), torch.tensor([3, 2])
        ).numpy()
    assert np.allclose(served, expected, atol=1e-5)


def test_seen_items_are_masked_out_of_the_score() -> None:
    indptr, codes = histories([[0, 1, 2]])
    embeddings = np.eye(3, dtype=np.float32)
    model = TwoTowerModel(
        item_embeddings=embeddings,
        history_indptr=indptr,
        history_codes=codes,
        recency_decay=0.9,
        max_history=50,
        config=TwoTowerConfig(factors=3),
    )
    seen = np.zeros(3, dtype=bool)
    seen[0] = True
    scores = model.score(0, seen)
    assert not np.isfinite(scores[0])
    assert np.isfinite(scores[1:]).all()


def test_untrained_items_are_never_recommended() -> None:
    """A zero embedding scores 0.0, which outranks almost every real item. It must be excluded."""
    embeddings = np.eye(4, dtype=np.float32)
    embeddings[3] = 0.0  # an item the model never trained on keeps its initial zero vector
    model = TwoTowerModel(
        item_embeddings=embeddings,
        history_indptr=np.array([0, 2], dtype=np.int64),
        history_codes=np.array([0, 1], dtype=np.int32),
        recency_decay=0.9,
        max_history=50,
        config=TwoTowerConfig(factors=4),
        trained_items=np.array([True, True, True, False]),
    )
    seen = np.zeros(4, dtype=bool)
    seen[[0, 1]] = True
    scores = model.score(0, seen)
    assert not np.isfinite(scores[3])
    assert np.isfinite(scores[2])


def test_dead_items_outlive_live_ones_without_the_guard() -> None:
    """Documents the failure the guard exists to prevent, so the reason for it stays visible."""
    # Item 1 sits opposite the user vector, so its cosine is negative and the untrained item 2 wins.
    embeddings = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    unguarded = TwoTowerModel(
        item_embeddings=embeddings,
        history_indptr=np.array([0, 1], dtype=np.int64),
        history_codes=np.array([0], dtype=np.int32),
        recency_decay=0.9,
        max_history=50,
        config=TwoTowerConfig(factors=2),
        trained_items=None,
    )
    seen = np.array([True, False, False])
    top = int(np.argmax(np.where(seen, -np.inf, unguarded.score(0, seen))))
    assert top == 2, "the dead item wins, which is why TwoTowerModel.score excludes it"


def test_model_round_trips_through_disk(tmp_path) -> None:  # type: ignore[no-untyped-def]
    indptr, codes = histories([[0, 1, 2], [1, 2]])
    config = TwoTowerConfig(factors=4, recency_decay=0.75, temperature=0.07)
    model = TwoTowerModel(
        item_embeddings=np.eye(3, dtype=np.float32),
        history_indptr=indptr,
        history_codes=codes,
        recency_decay=0.75,
        max_history=50,
        config=config,
    )
    path = write_model(model, tmp_path / "two_tower.npz")
    reloaded = load_model(path)
    assert reloaded.config.temperature == 0.07
    assert reloaded.trained_items is not None
    assert reloaded.trained_items.all()
    assert reloaded.recency_decay == 0.75
    assert np.array_equal(reloaded.history_codes, codes)
    assert np.allclose(reloaded.item_embeddings, model.item_embeddings)
