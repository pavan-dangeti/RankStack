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
import pytest

from rankstack.config import Settings
from rankstack.data.corpus import DEFAULT_CORPUS, TEST_START_MS, TRAIN_END_MS, get_corpus
from rankstack.data.corpus import raw_dir_for as package_raw_dir_for

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

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
