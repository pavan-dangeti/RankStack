"""Corpus registry: the datasets RankStack is allowed to train on.

Every number here was read from the publisher's documentation on 2026-10-08 and re-verified against
the published files (HTTP status and `content-length`) rather than recalled. Row counts are
re-derived by the pipeline and compared against these expectations, so a silently changed or
truncated download fails the build instead of quietly producing different results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
_INTERACTIONS = _BASE + "/benchmark/{core}core/rating_only/{category}.csv.gz"
_METADATA = _BASE + "/raw/meta_categories/meta_{category}.jsonl.gz"
_PUBLISHER_SPLIT = _BASE + "/benchmark/0core/timestamp_w_his/{category}.{split}.csv.gz"

#: Publisher's global-timeline boundaries for the 0-core corpus, in epoch milliseconds.
#: t1 = 2021-08-11T00:56:54.042Z, t2 = 2022-07-16T20:18:49.837Z.
TRAIN_END_MS = 1628643414042
TEST_START_MS = 1658002729837

#: Corpus-wide timestamp sanity bounds (see schema validation).
MIN_TIMESTAMP_MS = 820454400000  # 1996-01-01T00:00:00Z
MAX_TIMESTAMP_MS = 1698796800000  # 2023-11-01T00:00:00Z; corpus ends Sep 2023

#: Published statistics are rounded to three significant figures, so comparisons need a tolerance.
STAT_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class SplitExpectation:
    """Published row counts for the global-timeline split."""

    train: int
    valid: int
    test: int


@dataclass(frozen=True, slots=True)
class Corpus:
    """One dataset selection: a category, a k-core level, and what is expected of it."""

    name: str
    category: str
    core: int
    expected_users: int
    expected_items: int
    expected_interactions: int
    interactions_bytes: int
    metadata_bytes: int | None
    split: SplitExpectation | None = None
    is_primary: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def interactions_url(self) -> str:
        """Publisher URL of the `rating_only` interactions file."""
        return _INTERACTIONS.format(core=self.core, category=self.category)

    @property
    def metadata_url(self) -> str | None:
        """Publisher URL of the item metadata file, or None when the corpus has none."""
        return _METADATA.format(category=self.category) if self.metadata_bytes else None

    def publisher_split_url(self, split: str) -> str:
        """Return the publisher URL of a pre-split file, used only to cross-check our split."""
        return _PUBLISHER_SPLIT.format(category=self.category, split=split)

    @property
    def train_end_ms(self) -> int:
        """Exclusive upper bound of the train partition, in epoch milliseconds."""
        return TRAIN_END_MS

    @property
    def test_start_ms(self) -> int:
        """Inclusive lower bound of the test partition, in epoch milliseconds."""
        return TEST_START_MS


ELECTRONICS_0CORE = Corpus(
    name="electronics_0core",
    category="Electronics",
    core=0,
    expected_users=18_300_000,
    expected_items=1_600_000,
    expected_interactions=43_400_000,
    interactions_bytes=969_766_940,
    metadata_bytes=1_312_900_427,
    split=SplitExpectation(train=35_300_000, valid=4_000_000, test=4_100_000),
    is_primary=True,
    tags=("primary", "cold-start"),
)

ELECTRONICS_5CORE = Corpus(
    name="electronics_5core",
    category="Electronics",
    core=5,
    expected_users=1_600_000,
    expected_items=368_200,
    expected_interactions=15_500_000,
    interactions_bytes=257_671_078,
    metadata_bytes=1_312_900_427,
    tags=("comparability",),
)

MUSICAL_INSTRUMENTS_5CORE = Corpus(
    name="musical_instruments_5core",
    category="Musical_Instruments",
    core=5,
    expected_users=57_400,
    expected_items=24_600,
    expected_interactions=511_800,
    interactions_bytes=8_437_657,
    metadata_bytes=None,
    tags=("transfer-check",),
)

CORPORA: dict[str, Corpus] = {
    corpus.name: corpus
    for corpus in (ELECTRONICS_0CORE, ELECTRONICS_5CORE, MUSICAL_INSTRUMENTS_5CORE)
}

DEFAULT_CORPUS = ELECTRONICS_0CORE.name


def get_corpus(name: str) -> Corpus:
    """Look up a corpus by registry name."""
    try:
        return CORPORA[name]
    except KeyError as exc:
        known = ", ".join(sorted(CORPORA))
        msg = f"unknown corpus {name!r}; known corpora: {known}"
        raise KeyError(msg) from exc


def raw_dir_for(data_root: Path, corpus: Corpus) -> Path:
    """Directory holding a corpus's raw publisher files."""
    return data_root / "amazon_2023" / f"{corpus.category.lower()}_{corpus.core}core"


def utc_iso(timestamp_ms: int) -> str:
    """Format epoch milliseconds as an ISO-8601 UTC timestamp."""
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()


def publisher_issued_range() -> str:
    """Human-readable timestamp range the publisher says the 2023 corpus covers."""
    return f"{utc_iso(MIN_TIMESTAMP_MS)} to {utc_iso(MAX_TIMESTAMP_MS)}"
