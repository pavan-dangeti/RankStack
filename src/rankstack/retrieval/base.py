"""Candidate sources.

Every source scores the **whole** train catalogue for one user. That is what makes the evaluation
honest: Recall@K and NDCG@K are then exact full-catalogue numbers rather than the inflated
sampled-metric figures that dominate published recommender results, and no source can hide behind a
100-negative candidate sample.

A source returns raw scores with already-seen items set to `-inf`, so ranking never has to re-apply
the filter and never leaks an item the user has already reviewed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

NEG_INF = -np.inf


@runtime_checkable
class CandidateSource(Protocol):
    """Scores every catalogue item for one user."""

    @property
    def name(self) -> str:
        """Name reported in results files."""
        ...

    def n_items(self) -> int:
        """Return the number of items this source can score."""
        ...

    def score(self, user_code: int, seen: np.ndarray) -> np.ndarray:
        """Return a float32 score per item code, with `seen` codes set to `-inf`.

        `seen` is a boolean mask over item codes, so a source can honour it in whatever way is
        cheapest without a second pass.
        """
        ...

    def can_serve(self, user_code: int) -> bool:
        """Return True when the source has any information about this user."""
        ...


def mask_seen(scores: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Set already-seen items to `-inf`. Returns the same array, modified in place."""
    scores[seen] = NEG_INF
    return scores


def exact_rank(scores: np.ndarray, target_code: int) -> int:
    """1-based rank of one item among the finite scores, with ties broken by item code.

    Ties are broken deterministically rather than arbitrarily, so two sources that produce equal
    scores are ranked identically across runs and machines.
    """
    target = scores[target_code]
    if not np.isfinite(target):
        return -1
    # Compare against the whole vector: `-inf` never satisfies `> target` or `== target`, so masked
    # (already-seen) items drop out on their own. Counting over a filtered copy instead would
    # misalign values with their codes and corrupt the tie-break.
    higher = int(np.count_nonzero(scores > target))
    codes = np.arange(scores.size)
    equal_lower_code = int(np.count_nonzero((scores == target) & (codes < target_code)))
    return 1 + higher + equal_lower_code


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return `(codes, scores)` for the `k` best finite scores, best first.

    The order is *exactly* the order `exact_rank` implies: score descending, then ascending item
    code. The two must match, because a source whose top-K disagreed with its own rank of the same
    item would report a hit rate contradicting its own NDCG.

    Cost is O(n) in the catalogue size: a partial sort picks the boundary value, everything strictly
    above it is taken, and the tie group at the boundary is filled from its lowest codes.
    """
    finite = np.flatnonzero(np.isfinite(scores))
    if finite.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    values = scores[finite]
    size = min(k, finite.size)
    if size < finite.size:
        boundary = np.partition(values, finite.size - size)[finite.size - size]
        above = np.flatnonzero(values > boundary)
        tied = np.flatnonzero(values == boundary)
        needed = size - above.size
        chosen = finite[np.concatenate([above, tied[:needed]])]
    else:
        chosen = finite
    order = np.lexsort((chosen, -scores[chosen]))
    codes = chosen[order].astype(np.int64)
    return codes, scores[codes]
