"""Shared fixtures.

No dataset file is ever committed, so every test that needs data builds a synthetic corpus with a
fixed seed. The generator writes the publisher's on-disk formats (`rating_only` gzip CSV and
`meta_*` JSONL) so that the tests exercise the real readers, not a stand-in.
"""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import pytest

from rankstack.config import Settings
from rankstack.data.build import build_dataset
from rankstack.data.corpus import (
    DEFAULT_CORPUS,
    TEST_START_MS,
    TRAIN_END_MS,
    Corpus,
    get_corpus,
)
from rankstack.data.corpus import raw_dir_for as package_raw_dir_for
from rankstack.data.encode import build_encoding, write_encoding
from rankstack.eval.harness import CohortHistory
from rankstack.retrieval.als import AlsConfig, train_als
from rankstack.retrieval.cooccurrence import train_cooccurrence
from rankstack.retrieval.popularity import train_popularity

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from rankstack.data.build import BuildManifest
    from rankstack.data.encode import Encoding
    from rankstack.retrieval.als import AlsModel
    from rankstack.retrieval.cooccurrence import CooccurrenceModel
    from rankstack.retrieval.popularity import PopularityModel
    from rankstack.retrieval.two_tower import TwoTowerModel

SEED = 20261008
ITEMS = 40
USERS = 60
DAY_MS = 86_400_000
EPOCH_MS = 820_454_400_000  # 1996-01-01T00:00:00Z


@dataclass(frozen=True, slots=True)
class SyntheticCorpus:
    """A generated corpus on disk plus the numbers a test can assert against."""

    csv_path: Path
    metadata_path: Path | None
    interactions: int
    items: int
    users: int
    new_item_asin: str
    """An item whose first and only interaction is inside the test window."""
    single_history_user: str
    """A user with exactly one train-window interaction."""
    test_only_user: str
    """A user whose first interaction is inside the test window."""
    empty_title_asin: str
    """An item with no title, to exercise the title fallback."""


def _ts(day_offset: int) -> int:
    """Timestamp `day_offset` days after the start of 2021, away from both split boundaries."""
    base = 1625270400000  # 2021-07-01T00:00:00Z
    return base + day_offset * DAY_MS


def write_interactions_csv(path: Path, rows: list[tuple[str, str, float, int]]) -> None:
    """Write rows in the publisher's `rating_only` format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("user_id,parent_asin,rating,timestamp\n")
        for user_id, asin, rating, timestamp in rows:
            handle.write(f"{user_id},{asin},{rating},{timestamp}\n")


def write_metadata_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_synthetic_corpus(
    root: Path, *, seed: int = SEED, new_item_asin: str = "B0NEWITEM001"
) -> SyntheticCorpus:
    """Generate a small corpus that satisfies the same validation contract as the real one."""
    rng = random.Random(seed)  # noqa: S311 - fixture data
    catalogue = [f"B{index:09d}" for index in range(ITEMS)]
    new_item_asin = new_item_asin[:10]

    rows: list[tuple[str, str, float, int]] = []
    # Three users per item across the train window, so every catalogue item has history before t1.
    for index, asin in enumerate(catalogue):
        for offset in range(3):
            user_id = f"U{(index * 3 + offset) % USERS:04d}"
            rows.append((user_id, asin, float(rng.randint(1, 5)), _ts(offset)))

    # Validation window: one interaction per user, so every user is evaluable.
    for user_index in range(USERS):
        asin = catalogue[user_index % ITEMS]
        rows.append((f"U{user_index:04d}", asin, float(rng.randint(1, 5)), TRAIN_END_MS + DAY_MS))

    # Test window: the held-out target for every user, plus a brand-new item nobody knew before t1.
    for user_index in range(USERS):
        asin = catalogue[(user_index + 7) % ITEMS]
        rows.append(
            (
                f"U{user_index:04d}",
                asin,
                float(rng.randint(1, 5)),
                TEST_START_MS + DAY_MS * (user_index % 30 + 1),
            )
        )
    rows.append(("U0000", new_item_asin, 5.0, TEST_START_MS + DAY_MS * 3))

    # An item whose metadata has no title, to exercise the title fallback.
    empty_title_asin = "B0NOTITLE1"
    rows.append(("U0001", empty_title_asin, 4.0, _ts(1)))

    # A user with exactly one train-window interaction: collaborative filtering has no signal beyond
    # that item, so ALS cannot serve them while co-occurrence still can.
    single_history_user = "U9998"
    rows.append((single_history_user, catalogue[3], 5.0, _ts(0)))

    # A user who appears only in the test window: the cold-start case at miniature scale.
    test_only_user = "U9999"
    rows.append((test_only_user, catalogue[5], 4.0, TEST_START_MS + DAY_MS * 5))

    # A duplicate event and an item with no metadata, so the drop and coverage paths run.
    duplicate = rows[0]
    rows.append(duplicate)

    csv_path = root / "Electronics.csv.gz"
    write_interactions_csv(csv_path, rows)

    meta_rows: list[dict[str, Any]] = [
        {
            "parent_asin": asin,
            "title": None if asin == empty_title_asin else f"Synthetic product {asin}",
            "main_category": "Electronics",
            "categories": ["Electronics", "Accessories"],
            "price": round(rng.uniform(5, 500), 2),
            "store": "Synthetic Store",
            "description": ["A synthetic description used only in tests."],
        }
        for asin in [*catalogue, new_item_asin, empty_title_asin]
    ]
    # One item in the interactions with no metadata row at all.
    meta_rows = [row for row in meta_rows if row["parent_asin"] != catalogue[-1]]
    metadata_path = root / "meta_Electronics.jsonl.gz"
    write_metadata_jsonl(metadata_path, meta_rows)

    return SyntheticCorpus(
        csv_path=csv_path,
        single_history_user=single_history_user,
        test_only_user=test_only_user,
        metadata_path=metadata_path,
        interactions=len(rows),
        items=len({row[1] for row in rows}),
        users=len({row[0] for row in rows}),
        new_item_asin=new_item_asin,
        empty_title_asin=empty_title_asin,
    )


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Settings rooted entirely inside the test's temporary directory."""
    return Settings(
        data_root=tmp_path / "data",
        artifact_root=tmp_path / "artifacts",
        results_root=tmp_path / "results",
    )


@pytest.fixture
def synthetic_corpus(test_settings: Settings) -> SyntheticCorpus:
    """Return a corpus placed exactly where the build pipeline looks for raw files."""
    return make_synthetic_corpus(raw_dir_for(test_settings))


def raw_dir_for(settings: Settings) -> Path:
    """Return the raw-file directory `build_dataset` expects for the Electronics corpus."""
    return package_raw_dir_for(settings.data_root, get_corpus(DEFAULT_CORPUS))


@pytest.fixture
def connection() -> Iterator[object]:
    conn = duckdb.connect(database=":memory:")
    yield conn
    conn.close()


def iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()


def synthetic_corpus_spec(corpus: SyntheticCorpus, name: str = "synthetic") -> Corpus:
    """Return a `Corpus` whose published expectations match the generated fixture."""
    return Corpus(
        name=name,
        category="Electronics",
        core=0,
        expected_users=corpus.users,
        expected_items=corpus.items + 1,
        expected_interactions=corpus.interactions,
        interactions_bytes=corpus.csv_path.stat().st_size,
        metadata_bytes=corpus.metadata_path.stat().st_size if corpus.metadata_path else None,
        is_primary=True,
    )


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
    two_tower: TwoTowerModel | None = None


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
