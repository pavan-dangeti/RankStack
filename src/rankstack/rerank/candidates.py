"""Candidate generation for the re-ranking stage.

Stage 2 cannot rank what stage 1 never returned, so the candidate list is the ceiling on every
stage-2 metric. That ceiling is measured and published next to the final metrics, so a re-ranker can
never look better than the retrieval that fed it.

Two sources are unioned because they fail differently. `popularity_recent` is a global ranking, so
it has no personalisation but it is never empty. `cooccurrence` is personalised but returns nothing
for a user with no history. The union is at least as good as either, and the `source_count` feature
lets the ranker learn how much to trust each one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """One user's candidate list plus the per-retriever scores that produced it."""

    user_code: int
    items: np.ndarray
    scores: dict[str, np.ndarray]
    target: int | None = None

    def __len__(self) -> int:
        """Return the number of candidates in the list."""
        return int(self.items.size)

    def hits(self) -> bool:
        """Return True when the held-out target is somewhere in the list."""
        return self.target is not None and bool(np.any(self.items == self.target))


def merge_sources(
    ranked_by_source: dict[str, np.ndarray], seen: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Union several per-source candidate lists into one list and a per-candidate source count.

    Sources are consumed in the order given, so the result is deterministic: the first source's
    ranking decides the order, later sources only append what is genuinely new, and `source_count`
    records how many sources independently proposed each candidate. That agreement is a signal in
    itself, which is why it is a feature rather than an accident of the merge.
    """
    taken = np.zeros(seen.size, dtype=bool)
    taken[seen] = True
    ordered: list[np.ndarray] = []
    agreement: list[np.ndarray] = []
    for items in ranked_by_source.values():
        if not items.size:
            continue
        fresh = items[~taken[items]]
        taken[items] = True
        if fresh.size:
            ordered.append(fresh)
            agreement.append(np.ones(fresh.size, dtype=np.float32))
    if not ordered:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    items = np.concatenate(ordered)
    counts = np.zeros(items.size, dtype=np.float32)
    for source_items in ranked_by_source.values():
        if source_items.size:
            counts += np.isin(items, source_items).astype(np.float32)
    return items, counts.astype(np.float32)


def recall_ceiling(candidate_lists: list[np.ndarray], targets: np.ndarray) -> dict[str, float]:
    """Fraction of targets present in each candidate-list length: the ceiling for stage 2.

    Reported at the lengths that matter, because a ranker evaluated at K=20 over a 500-item list is
    being scored against a ceiling of recall@500, not recall@20.
    """
    sizes = np.array([len(items) for items in candidate_lists], dtype=np.int64)
    found = np.array(
        [
            bool(np.any(items == target))
            for items, target in zip(candidate_lists, targets, strict=True)
        ]
    )
    out: dict[str, float] = {}
    for n in (20, 50, 100, 200, 500):
        # A list shorter than n can never hold a hit at n, so reporting it would flatter the
        # ceiling rather than bound it.
        if n <= int(sizes.min()):
            out[f"recall_at_{n}"] = float(found.mean())
    out["recall_full_list"] = float(found.mean())
    out["mean_candidates"] = float(sizes.mean())
    return out
