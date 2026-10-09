"""Post-ranking rules, checked against hand-computed expectations.

These are pure functions, so they are tested in isolation rather than only end to end. Each test
states the expected ordering explicitly: a rule that silently changes behaviour should fail here,
not six layers up in an evaluation result.
"""

from __future__ import annotations

import numpy as np
import pytest

from rankstack.rerank.candidates import merge_sources, recall_ceiling
from rankstack.rerank.features import CodeLookup, margin_to_best, within_user_rank
from rankstack.rerank.model import apply_mmr, cap_categories, freshness_bonus


def test_mmr_with_lambda_one_is_plain_relevance_order() -> None:
    items = np.array([0, 1, 2])
    scores = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    similarity = np.eye(3, dtype=np.float32)
    assert apply_mmr(items, scores, similarity, 3, mmr_lambda=1.0).tolist() == [1, 2, 0]


def test_mmr_prefers_a_diverse_second_pick() -> None:
    """Two near-identical items cannot both win: the second slot goes to a different one."""
    items = np.array([0, 1, 2])
    scores = np.array([0.9, 0.85, 0.2], dtype=np.float32)
    similarity = np.array([[1.0, 0.95, 0.0], [0.95, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    picked = apply_mmr(items, scores, similarity, 2, mmr_lambda=0.5)
    assert picked[0] == 0
    assert picked[1] == 2, "item 1 is nearly identical to the already-selected item 0"


def test_mmr_returns_nothing_for_an_empty_list() -> None:
    empty = np.empty(0, dtype=np.int64)
    assert apply_mmr(empty, np.empty(0, np.float32), np.empty((0, 0), np.float32), 5).size == 0


def test_category_cap_promotes_the_next_best_from_another_category() -> None:
    items = np.arange(6)
    categories = np.array([1, 1, 1, 2, 2, 3])
    kept = cap_categories(items, categories, 4, cap=2)
    assert kept.tolist() == [0, 1, 3, 4], "third item of category 1 is skipped for category 2's"


def test_category_cap_does_not_count_unknown_categories() -> None:
    """An unknown category is not evidence of a category, so it must not consume the cap."""
    items = np.arange(5)
    categories = np.array([-1, -1, -1, -1, -1])
    assert cap_categories(items, categories, 3, cap=1).tolist() == [0, 1, 2]


def test_a_zero_cap_disables_the_rule() -> None:
    items = np.arange(4)
    assert cap_categories(items, np.array([1, 1, 1, 1]), 3, cap=0).tolist() == [0, 1, 2]


def test_freshness_bonus_decays_with_age_and_rewards_the_unknown() -> None:
    days = np.array([0.0, 30.0, np.nan])
    bonus = freshness_bonus(days, weight=1.0, halflife_days=30.0)
    assert bonus[0] == pytest.approx(1.0)
    assert bonus[1] == pytest.approx(0.5, abs=1e-6)
    assert bonus[2] == pytest.approx(1.0), "an unseen item is treated as brand new, not as old"


def test_merge_sources_unions_in_order_and_counts_agreement() -> None:
    seen = np.zeros(10, dtype=bool)
    seen[9] = True
    items, counts = merge_sources(
        {"popularity_recent": np.array([0, 1, 2, 9]), "cooccurrence": np.array([1, 3])}, seen
    )
    assert items.tolist() == [0, 1, 2, 3], "seen items dropped, duplicates kept once, order kept"
    assert counts.tolist() == [1.0, 2.0, 1.0, 1.0], "item 1 was proposed by both sources"


def test_merge_sources_of_nothing_is_empty_not_an_error() -> None:
    items, counts = merge_sources({"a": np.empty(0, dtype=np.int64)}, np.zeros(3, dtype=bool))
    assert items.size == 0
    assert counts.size == 0


def test_recall_ceiling_reports_the_fraction_of_targets_returned() -> None:
    lists = [np.arange(100), np.arange(100), np.arange(100)]
    found = recall_ceiling(lists, np.array([2, 999, 7]))
    assert found["recall_full_list"] == pytest.approx(2 / 3)
    assert found["mean_candidates"] == 100.0
    assert found["recall_at_20"] == pytest.approx(2 / 3), "both hits are inside the top 20"


def test_recall_ceiling_ignores_lengths_the_lists_never_reach() -> None:
    """Reporting recall@500 for 50-item lists would overstate the ceiling."""
    found = recall_ceiling([np.arange(50)], np.array([10]))
    assert "recall_at_100" not in found
    assert "recall_at_20" in found


def test_code_lookup_resolves_known_ids_and_rejects_unknown_ones() -> None:
    lookup = CodeLookup({"a": 0, "b": 1, "c": 2})
    assert lookup.codes_for(np.array(["a", "c", "zz"])).tolist() == [0, 2, -1]


def test_within_user_rank_resets_per_user_and_ignores_missing_scores() -> None:
    scores = np.array([0.1, 0.9, np.nan, 0.5], dtype=np.float32)
    users = np.array([1, 1, 1, 2])
    assert np.array_equal(
        within_user_rank(scores, users), np.array([1.0, 0.0, np.nan, 0.0]), equal_nan=True
    )


def test_margin_to_best_is_zero_for_the_best_and_negative_below_it() -> None:
    scores = np.array([1.0, 0.0, -1.0, 5.0], dtype=np.float32)
    users = np.array([1, 1, 1, 2])
    margin = margin_to_best(scores, users)
    assert margin[0] == 0.0
    assert margin[1] == pytest.approx(-1.0)
    assert margin[3] == 0.0, "each user is compared against their own best"
