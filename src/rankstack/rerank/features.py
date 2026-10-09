"""Features for the re-ranking stage.

Stage 2 reorders a candidate list, so it can only ever be as good as the list it is given. Every
feature here is computed **as of the training boundary** and read from the point-in-time feature
store Phase 1 built and its leakage tests cover, so nothing can carry information from after the
moment it is used at.

Three groups:

- **Retrieval signals** — the score each retriever gave a candidate, plus the rank that score
  implies within the user. A candidate's score is recomputed from stored factors rather than by
  re-scoring the catalogue, which is what makes stage-2 feature extraction affordable.
- **Item signals** — popularity, age, price, read from the as-of snapshot.
- **Cross signals** — how the candidate relates to *this* user's history: category affinity and
  price affinity. These are the only features that can separate two candidates a global retriever
  ranked identically, which is the entire reason a second stage exists.

A candidate a retriever never scored gets `NaN`, never zero. Zero means "scored and scored nothing";
`NaN` means "this retriever did not consider it", and LightGBM can split on that difference.
Collapsing the two would tell the model an unscored candidate is a bad one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime by the writer

import duckdb
import numpy as np

from rankstack.logging import get_logger

_logger = get_logger(__name__)

#: Retrievers whose score becomes a feature.
SCORED_RETRIEVERS: tuple[str, ...] = (
    "popularity",
    "popularity_recent",
    "cooccurrence",
    "als",
    "two_tower",
)

#: Column order. LightGBM indexes features positionally, so this tuple is part of the artefact
#: contract and is written next to the saved model.
FEATURE_NAMES: tuple[str, ...] = (
    "score_popularity",
    "score_popularity_recent",
    "score_cooccurrence",
    "score_als",
    "score_two_tower",
    "rank_popularity_recent",
    "log_item_interactions",
    "item_days_since_first",
    "item_days_since_last",
    "log_price",
    "price_missing",
    "price_percentile",
    "log_user_depth",
    "category_affinity",
    "price_affinity",
    "score_margin_to_best",
    "candidate_position",
    "source_count",
)

N_FEATURES = len(FEATURE_NAMES)
_ITEM_COLUMNS = slice(6, 12)


class CodeLookup:
    """Vectorised string-id to integer-code lookup.

    Built once from the encoding dict and queried with `searchsorted`, because a Python loop over
    1.3M items costs minutes and shows up as minutes.
    """

    def __init__(self, id_to_code: dict[str, int]) -> None:
        """Precompute the sorted key table used to resolve ids to codes."""
        ids = np.asarray(list(id_to_code), dtype=str)
        keys, inverse = np.unique(ids, return_inverse=True)
        if not np.array_equal(keys[inverse], ids):
            msg = "encoding ids are not unique"
            raise ValueError(msg)
        self._keys = keys
        self._inverse = inverse

    def codes_for(self, values: np.ndarray) -> np.ndarray:
        """Return codes for an array of ids, with -1 where the id is unknown."""
        values = np.asarray(values, dtype=str)
        clipped = np.clip(np.searchsorted(self._keys, values), 0, self._keys.size - 1)
        known = self._keys[clipped] == values
        return np.where(known, self._inverse[clipped], -1).astype(np.int64)


@dataclass(frozen=True, slots=True)
class ItemFeatures:
    """Per-item features as of one boundary, indexed by item code."""

    log_interactions: np.ndarray
    days_since_first: np.ndarray
    days_since_last: np.ndarray
    log_price: np.ndarray
    price_missing: np.ndarray
    price_percentile: np.ndarray
    category: np.ndarray
    n_categories: int

    def n_items(self) -> int:
        """Return the number of items these features cover."""
        return int(self.log_interactions.size)

    def gather(self, codes: np.ndarray) -> np.ndarray:
        """Return the `(rows, 6)` item-level feature block for a flat code array."""
        codes = codes.astype(np.int64)
        return np.column_stack(
            (
                self.log_interactions[codes],
                self.days_since_first[codes],
                self.days_since_last[codes],
                self.log_price[codes],
                self.price_missing[codes],
                self.price_percentile[codes],
            )
        ).astype(np.float32)


@dataclass(frozen=True, slots=True)
class UserAggregates:
    """Per-user aggregates over the capped histories of one batch, keyed by user code."""

    depth: dict[int, int]
    mean_log_price: dict[int, float]
    category_share: dict[tuple[int, int], float]

    def category_affinity(self, user_codes: np.ndarray, item_categories: np.ndarray) -> np.ndarray:
        """Share of each user's history spent in the candidate's category; 0 when unknown."""
        return np.array(
            [
                self.category_share.get((int(user), int(category)), 0.0)
                for user, category in zip(user_codes, item_categories, strict=True)
            ],
            dtype=np.float32,
        )

    def depths(self, user_codes: np.ndarray) -> np.ndarray:
        """Return each user's train-window interaction count."""
        return np.array([self.depth.get(int(user), 0) for user in user_codes], dtype=np.float32)

    def prices(self, user_codes: np.ndarray) -> np.ndarray:
        """Return each user's mean log price, `NaN` when they bought nothing with a price."""
        return np.array(
            [self.mean_log_price.get(int(user), np.nan) for user in user_codes], dtype=np.float32
        )


def user_aggregates(
    user_codes: np.ndarray,
    history_ptr: np.ndarray,
    history_codes: np.ndarray,
    items: ItemFeatures,
) -> UserAggregates:
    """Compute depth, mean log price and per-category share for a batch of users.

    Only the batch's users are processed. Building these for all 15.4M users to then use 100k of
    them would be wasted minutes.
    """
    depth: dict[int, int] = {}
    mean_price: dict[int, float] = {}
    shares: dict[tuple[int, int], float] = {}
    for raw_user in user_codes:
        user = int(raw_user)
        start, end = int(history_ptr[user]), int(history_ptr[user + 1])
        if end <= start:
            depth[user] = 0
            continue
        history = history_codes[start:end].astype(np.int64)
        depth[user] = history.size
        prices = items.log_price[history]
        finite = prices[np.isfinite(prices)]
        if finite.size:
            mean_price[user] = float(finite.mean())
        categories = items.category[history]
        valid = categories[categories >= 0]
        if valid.size:
            values, counts = np.unique(valid, return_counts=True)
            for value, count in zip(values, counts, strict=True):
                shares[(user, int(value))] = float(count) / history.size
    return UserAggregates(depth=depth, mean_log_price=mean_price, category_share=shares)


def within_user_rank(scores: np.ndarray, user_codes: np.ndarray) -> np.ndarray:
    """Rank of each row within its user by descending score; `NaN` where the score is not finite."""
    ranks = np.full(scores.size, np.nan, dtype=np.float32)
    # lexsort's last key is the primary one: group by user first, then order by score inside it.
    order = np.lexsort(
        (np.arange(scores.size), -np.where(np.isfinite(scores), scores, np.inf), user_codes)
    )
    boundaries = np.flatnonzero(np.diff(user_codes[order])) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [scores.size]])
    for start, end in zip(starts, ends, strict=True):
        rows = order[start:end]
        finite = np.isfinite(scores[rows])
        if finite.any():
            ranks[rows[finite]] = np.arange(int(finite.sum()), dtype=np.float32)
    return ranks


def margin_to_best(scores: np.ndarray, user_codes: np.ndarray) -> np.ndarray:
    """Each row's distance below the best score in its own user."""
    filled = np.where(np.isfinite(scores), scores, -np.inf)
    best = np.full(scores.size, -np.inf)
    for user in np.unique(user_codes):
        rows = user_codes == user
        best[rows] = filled[rows].max()
    return np.where(np.isfinite(filled), filled - best, 0.0).astype(np.float32)


def assemble(
    user_codes: np.ndarray,
    item_codes: np.ndarray,
    scores: dict[str, np.ndarray],
    items: ItemFeatures,
    aggregates: UserAggregates,
    *,
    source_count: np.ndarray | None = None,
) -> np.ndarray:
    """Build the `(rows, N_FEATURES)` matrix for one batch of user-candidate pairs.

    `scores` maps a retriever name to a per-row score array where `NaN` means the retriever never
    ranked that candidate.
    """
    rows = user_codes.size
    columns: dict[str, np.ndarray] = {}

    for name in SCORED_RETRIEVERS:
        values = scores.get(name)
        columns[f"score_{name}"] = (
            np.full(rows, np.nan, dtype=np.float32) if values is None else values.astype(np.float32)
        )
    recent = scores.get("popularity_recent")
    columns["rank_popularity_recent"] = (
        np.full(rows, np.nan, dtype=np.float32)
        if recent is None
        else within_user_rank(recent, user_codes)
    )

    item_block = items.gather(item_codes)
    for offset, name in enumerate(FEATURE_NAMES[_ITEM_COLUMNS]):
        columns[name] = item_block[:, offset]

    depths = aggregates.depths(user_codes)
    columns["log_user_depth"] = np.log1p(depths)
    columns["category_affinity"] = aggregates.category_affinity(
        user_codes, items.category[item_codes.astype(np.int64)]
    )
    columns["price_affinity"] = np.abs(aggregates.prices(user_codes) - item_block[:, 3])

    finite = np.vstack([columns[f"score_{name}"] for name in SCORED_RETRIEVERS])
    columns["score_margin_to_best"] = margin_to_best(np.nanmax(finite, axis=0), user_codes)
    columns["candidate_position"] = np.arange(rows, dtype=np.float32)
    columns["source_count"] = (
        np.isfinite(finite).sum(axis=0).astype(np.float32)
        if source_count is None
        else source_count.astype(np.float32)
    )
    return np.column_stack([columns[name] for name in FEATURE_NAMES])


def load_item_features(
    processed_dir: Path,
    as_of_ms: int,
    item_to_code: dict[str, int],
    category_column: str = "main_category",
) -> ItemFeatures:
    """Read the point-in-time item features and item metadata as of `as_of_ms`.

    `as_of_ms` must be one of the publisher boundaries Phase 1 chose, because the snapshots only
    exist there. Using a boundary the leakage tests already cover is cheaper and safer than
    recomputing an arbitrary as-of.
    """
    features_path = processed_dir / "features" / "item_as_of_features.parquet"
    items_path = processed_dir / "items" / "items.parquet"
    if not features_path.exists():
        msg = f"no point-in-time feature snapshot at {features_path}; run the data build first"
        raise FileNotFoundError(msg)

    lookup = CodeLookup(item_to_code)
    n_items = len(item_to_code)
    connection = duckdb.connect(database=":memory:")
    snapshot = (
        connection.execute(
            """
            SELECT entity_id, feature_name, value_double
            FROM read_parquet(?)
            WHERE as_of_ms = ?
            """,
            [features_path.as_posix(), as_of_ms],
        )
        .arrow()
        .read_all()
    )
    # Both category columns are read and one is selected in numpy. DuckDB cannot bind a column
    # identifier as a query parameter, and interpolating one into the SQL would be the injection
    # vector the project's own lint rule exists to prevent.
    metadata = (
        connection.execute(
            """
            SELECT parent_asin, main_category, category_path, price FROM read_parquet(?)
            """,
            [items_path.as_posix()],
        )
        .arrow()
        .read_all()
    )
    connection.close()

    interactions = np.zeros(n_items, dtype=np.float32)
    days_first = np.full(n_items, np.nan, dtype=np.float32)
    days_last = np.full(n_items, np.nan, dtype=np.float32)
    names = snapshot.column("feature_name").to_numpy(zero_copy_only=False).astype(str)
    values = snapshot.column("value_double").to_numpy(zero_copy_only=False).astype(np.float32)
    codes = lookup.codes_for(snapshot.column("entity_id").to_numpy(zero_copy_only=False))
    known = codes >= 0
    for mask, target in (
        (names == "item_interactions", interactions),
        (names == "item_days_since_first_interaction", days_first),
        (names == "item_days_since_last_interaction", days_last),
    ):
        selected = known & mask
        target[codes[selected]] = values[selected]

    meta_codes = lookup.codes_for(metadata.column("parent_asin").to_numpy(zero_copy_only=False))
    meta_known = meta_codes >= 0
    raw_price = metadata.column("price").to_numpy(zero_copy_only=False)
    prices = np.array(
        [np.nan if value is None else float(value) for value in raw_price], dtype=np.float32
    )
    price_block = np.full(n_items, np.nan, dtype=np.float32)
    # Scatter through the codes: the metadata file has more rows than there are codes, and some of
    # its items were never in the train window.
    price_block[meta_codes[meta_known]] = prices[meta_known]

    category = np.full(n_items, -1, dtype=np.int64)
    category_ids: dict[str, int] = {}
    labels = metadata.column(category_column).to_numpy(zero_copy_only=False)
    for position, label in enumerate(labels):
        if label is None or not meta_known[position]:
            continue
        category[meta_codes[position]] = category_ids.setdefault(str(label), len(category_ids))

    priced = price_block > 0
    percentile = np.full(n_items, np.nan, dtype=np.float32)
    if priced.any():
        ordered = np.sort(price_block[priced])
        percentile[priced] = np.searchsorted(ordered, price_block[priced], side="left") / max(
            int(priced.sum()) - 1, 1
        )
    log_price = np.full(n_items, np.nan, dtype=np.float32)
    log_price[priced] = np.log1p(price_block[priced])

    features = ItemFeatures(
        log_interactions=np.log1p(interactions),
        days_since_first=days_first,
        days_since_last=days_last,
        log_price=log_price,
        price_missing=(~priced).astype(np.float32),
        price_percentile=percentile,
        category=category,
        n_categories=max(len(category_ids), 1),
    )
    _logger.info(
        "rerank.item_features",
        extra={
            "as_of_ms": as_of_ms,
            "items": features.n_items(),
            "with_price": int(priced.sum()),
            "categories": features.n_categories,
        },
    )
    return features


def write_feature_schema(path: Path) -> Path:
    """Persist the column order, which the saved model depends on positionally."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"features": list(FEATURE_NAMES), "scored_retrievers": list(SCORED_RETRIEVERS)},
            indent=2,
        )
        + "\n"
    )
    return path
