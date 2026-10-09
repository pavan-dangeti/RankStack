"""Item-to-item co-occurrence.

The classic neighbourhood baseline: users who bought X also bought Y. Two engineering decisions are
worth stating because they change what the numbers mean.

**Capped history.** Co-occurrence is quadratic in history length, and 0.03% of train users have more
than 100 items. History is capped at `max_history` items (most recent kept) so one power user cannot
dominate the neighbourhood of everything they touched.

**Users with one interaction are skipped when counting.** A single-item history produces no pairs at
all, so excluding those 9.05M users from the product loses nothing and removes 59% of the work. They
keep their history for scoring and are still evaluated.

**Pruned neighbours.** The full co-occurrence matrix does not fit: 1.33M items squared is not a
sparse-friendly shape. Counts are accumulated in user blocks and reduced to the top `top_k`
neighbours per item **after every block**, so the running matrix never exceeds
`n_items * top_k` non-zeros. Measured on Electronics 0-core: one 200k-user block produces 131M
non-zeros before pruning and about 48M after, and pruning the *accumulated* matrix instead of each
block would reach ~2.2B non-zeros, which does not fit in 16 GB.

Pruning after each block makes the final neighbour ranking approximate: a neighbour dropped early
could have earned a place in the final top-k. `top_k` is therefore a tuned parameter reported in the
result file, not a silent constant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime by the writers

import duckdb
import numpy as np
import scipy.sparse as sp

from rankstack.logging import get_logger

_logger = get_logger(__name__)

#: A history of this size or longer contains at least one item pair, so it can contribute.
MIN_PAIRABLE_HISTORY = 2
DEFAULT_MAX_HISTORY = 50
DEFAULT_TOP_K = 100
DEFAULT_USER_BLOCK = 200_000


@dataclass(frozen=True, slots=True)
class CooccurrenceModel:
    """Cosine-normalised item-item neighbours plus the capped history they are built from."""

    similarity: sp.csr_matrix
    item_popularity: np.ndarray
    history_indptr: np.ndarray
    history_codes: np.ndarray
    max_history: int
    top_k: int
    user_block: int

    @property
    def name(self) -> str:
        """Name reported in results files."""
        return "cooccurrence"

    def n_items(self) -> int:
        """Return the number of scorable items."""
        return int(self.similarity.shape[0])

    def can_serve(self, user_code: int) -> bool:
        """Return True when the user has at least one train-window interaction."""
        return self.history_length(user_code) >= 1

    def history(self, user_code: int) -> np.ndarray:
        """Item codes the user interacted with in the train window, most recent first."""
        start, end = self.history_indptr[user_code], self.history_indptr[user_code + 1]
        return self.history_codes[start:end]

    def history_length(self, user_code: int) -> int:
        """Return the size of this user's capped history."""
        start, end = self.history_indptr[user_code], self.history_indptr[user_code + 1]
        return int(end - start)

    def score(self, user_code: int, seen: np.ndarray) -> np.ndarray:
        """Score every item as the similarity-weighted sum over the user's capped history."""
        return self.score_with_history(self.history(user_code), seen)

    def score_with_history(self, history: np.ndarray, seen: np.ndarray) -> np.ndarray:
        """Score every item from an explicit history, for leave-one-out training.

        A held-out item is still inside the model's stored history, so scoring a user with that
        stored history would let the target boost itself through its own row. Passing the history in
        is what makes leave-one-out honest.
        """
        scores = np.zeros(self.n_items(), dtype=np.float32)
        if history.size:
            accumulated = np.asarray(self.similarity[history].sum(axis=0)).ravel()
            scores[: accumulated.size] = accumulated.astype(np.float32)
        scores[seen] = -np.inf
        return scores


def build_history(
    train_parquet: Path,
    user_to_code: dict[str, int],
    item_to_code: dict[str, int],
    max_history: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(user_indptr, item_items, item_popularity)` for the capped train history.

    Fully vectorised: pulling 35M rows into Python lists and looping over 15M users takes minutes,
    while the Arrow -> numpy path takes seconds. Items within a user are ordered most-recent-first,
    so truncation keeps the recent half of a long history.
    """
    connection = duckdb.connect(database=":memory:")
    table = connection.execute(
        """
        SELECT user_id, parent_asin, max(timestamp) AS last_seen
        FROM read_parquet(?)
        GROUP BY user_id, parent_asin
        """,
        [str(train_parquet)],
    ).arrow()
    connection.close()
    table = table.read_all()

    n_users = len(user_to_code)
    n_items = len(item_to_code)
    user_ids = table.column("user_id").to_numpy(zero_copy_only=False)
    item_ids = table.column("parent_asin").to_numpy(zero_copy_only=False)
    timestamps = table.column("last_seen").to_numpy(zero_copy_only=False).astype(np.int64)

    # Map string ids to codes in one vectorised pass via pandas-free lookup on the unique values.
    unique_users, user_index = np.unique(user_ids, return_inverse=True)
    user_codes = np.array(
        [user_to_code.get(str(value), -1) for value in unique_users], dtype=np.int64
    )[user_index]
    unique_items, item_index = np.unique(item_ids, return_inverse=True)
    item_codes = np.array(
        [item_to_code.get(str(value), -1) for value in unique_items], dtype=np.int64
    )[item_index]

    known = (user_codes >= 0) & (item_codes >= 0)
    user_codes, item_codes, timestamps = user_codes[known], item_codes[known], timestamps[known]

    # Sort by user ascending, then most recent first, so the cap keeps recent items.
    order = np.lexsort((-timestamps, user_codes))
    user_codes, item_codes = user_codes[order], item_codes[order]

    raw_counts = np.bincount(user_codes, minlength=n_users)
    capped = np.minimum(raw_counts, max_history)
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(capped, out=indptr[1:])

    starts = np.concatenate([[0], np.cumsum(raw_counts)[:-1]])
    positions = np.arange(user_codes.size) - np.repeat(starts, raw_counts)
    keep = positions < np.repeat(capped, raw_counts)
    codes = item_codes[keep].astype(np.int32)
    popularity = np.bincount(item_codes, minlength=n_items).astype(np.float32)
    return indptr, codes, popularity


def train_cooccurrence(
    train_parquet: Path,
    user_to_code: dict[str, int],
    item_to_code: dict[str, int],
    *,
    max_history: int = DEFAULT_MAX_HISTORY,
    top_k: int = DEFAULT_TOP_K,
    user_block: int = DEFAULT_USER_BLOCK,
) -> CooccurrenceModel:
    """Build cosine-normalised item-item neighbours from the capped train history."""
    indptr, indices, popularity = build_history(
        train_parquet, user_to_code, item_to_code, max_history
    )
    n_users = len(user_to_code)
    n_items = len(item_to_code)

    rows = np.repeat(np.arange(n_users, dtype=np.int32), np.diff(indptr))
    user_item = sp.csr_matrix(
        (np.ones(indices.size, dtype=np.float32), (rows, indices)), shape=(n_users, n_items)
    )
    user_item.eliminate_zeros()

    depths = np.diff(indptr)
    pairable = depths >= MIN_PAIRABLE_HISTORY
    skipped = int(n_users - pairable.sum())
    user_item = user_item[pairable]
    pairable_users = int(pairable.sum())
    _logger.info(
        "cooccurrence.pairable_users",
        extra={"pairable": pairable_users, "skipped_single_interaction": skipped},
    )

    counts = sp.csr_matrix((n_items, n_items), dtype=np.float32)
    blocks_processed = 0
    for start in range(0, pairable_users, user_block):
        block = user_item[start : start + user_block]
        if block.nnz == 0:
            continue
        merged = (counts + (block.T @ block)).tocsr()
        counts = _prune_rows(merged, top_k)
        blocks_processed += 1
    _logger.info("cooccurrence.blocks", extra={"blocks": blocks_processed, "nnz": int(counts.nnz)})

    # cosine(i, j) = count(i, j) / sqrt(popularity_i * popularity_j)
    norm = np.sqrt(popularity)
    inverse = np.where(norm > 0, 1.0 / np.where(norm > 0, norm, 1.0), 0.0).astype(np.float32)
    cosine = (sp.diags(inverse) @ counts @ sp.diags(inverse)).tocsr()
    # An item is its own most frequent co-occurrer, so the diagonal is a self-count artefact rather
    # than a neighbour signal. Leaving it in would make every item in a user's history rank first.
    cosine.setdiag(0.0)
    cosine.eliminate_zeros()

    model = CooccurrenceModel(
        similarity=cosine,
        item_popularity=popularity,
        history_indptr=indptr,
        history_codes=indices,
        max_history=max_history,
        top_k=top_k,
        user_block=user_block,
    )
    _logger.info(
        "cooccurrence.trained",
        extra={
            "nnz": int(cosine.nnz),
            "density": round(cosine.nnz / max(n_items * n_items, 1), 9),
            "max_history": max_history,
            "top_k": top_k,
        },
    )
    return model


def _prune_rows(matrix: sp.csr_matrix, top_k: int) -> sp.csr_matrix:
    """Keep only the `top_k` largest values in each row of a CSR matrix."""
    matrix = matrix.tocsr()
    indptr = matrix.indptr
    data = matrix.data
    indices = matrix.indices
    keep_data: list[np.ndarray] = []
    keep_indices: list[np.ndarray] = []
    new_indptr = np.zeros(matrix.shape[0] + 1, dtype=indices.dtype)
    for row in range(matrix.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        row_data = data[start:end]
        row_indices = indices[start:end]
        if row_data.size > top_k:
            head = np.argpartition(-row_data, top_k - 1)[:top_k]
            head = head[np.argsort(-row_data[head], kind="stable")]
            row_data = row_data[head]
            row_indices = row_indices[head]
        keep_data.append(row_data)
        keep_indices.append(row_indices)
        new_indptr[row + 1] = new_indptr[row] + row_data.size
    return sp.csr_matrix(
        (
            np.concatenate(keep_data) if keep_data else np.empty(0, dtype=data.dtype),
            np.concatenate(keep_indices) if keep_indices else np.empty(0, dtype=indices.dtype),
            new_indptr,
        ),
        shape=matrix.shape,
    )


def write_model(model: CooccurrenceModel, path: Path) -> Path:
    """Persist the similarity matrix, history and parameters to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, model.similarity)
    np.savez(
        path.with_suffix(".history.npz"),
        history_indptr=model.history_indptr,
        history_codes=model.history_codes,
        item_popularity=model.item_popularity,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "max_history": model.max_history,
                "top_k": model.top_k,
                "user_block": model.user_block,
                "n_items": model.n_items(),
                "nnz": int(model.similarity.nnz),
            },
            sort_keys=True,
        )
    )
    return path
