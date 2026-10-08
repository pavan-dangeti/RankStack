"""Popularity and recency baselines.

These are the floor every learned model has to clear. They also serve the 57% of test users who have
no train history at all, where nothing collaborative can help.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime by the writers

import duckdb
import numpy as np

from rankstack.logging import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PopularityModel:
    """Item scores from the train window, with optional recency weighting.

    `decay_days` is the half-life-free linear decay applied over the trailing window: an item's
    score is its interaction count multiplied by `exp(-age_days / decay_days)`. Decay is applied
    against `as_of_ms`, the prediction moment, never against wall-clock now.
    """

    scores: np.ndarray
    train_end_ms: int
    decayed: bool

    @property
    def name(self) -> str:
        """Name reported in results files."""
        return "popularity_recent" if self.decayed else "popularity"

    def n_items(self) -> int:
        """Return the number of scorable items."""
        return int(self.scores.size)

    def can_serve(self, user_code: int) -> bool:
        """Return True: a global ranking needs nothing about the user."""
        del user_code  # a global ranking needs nothing about the user
        return True

    def score(self, user_code: int, seen: np.ndarray) -> np.ndarray:
        """Return the same global score vector for every user, with seen items masked out."""
        del user_code
        return np.where(seen, -np.inf, self.scores).astype(np.float32)

    def write(self, path: Path) -> Path:
        """Persist the score vector and its parameters."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.scores)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "train_end_ms": self.train_end_ms,
                    "decayed": self.decayed,
                    "n_items": self.n_items(),
                },
                sort_keys=True,
            )
        )
        return path

    @classmethod
    def load(cls, path: Path) -> PopularityModel:
        """Load a persisted model."""
        metadata = json.loads(path.with_suffix(".json").read_text())
        return cls(
            scores=np.load(path),
            train_end_ms=int(metadata["train_end_ms"]),
            decayed=bool(metadata["decayed"]),
        )


def train_popularity(
    train_parquet: Path,
    item_to_code: dict[str, int],
    train_end_ms: int,
    *,
    decay_days: float | None = None,
) -> PopularityModel:
    """Count train interactions per item, optionally decayed by age at the prediction moment.

    `decay_days=None` gives plain popularity. A decay makes the baseline recency-aware, which is the
    honest comparison for a ranking model that also sees a freshness feature.
    """
    connection = duckdb.connect(database=":memory:")
    if decay_days is None:
        rows = connection.execute(
            """
            SELECT parent_asin, count(*) AS interactions
            FROM read_parquet(?)
            GROUP BY parent_asin
            """,
            [str(train_parquet)],
        ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT parent_asin,
                   sum(exp(-(? - timestamp) / (? * 86400000.0))) AS interactions
            FROM read_parquet(?)
            GROUP BY parent_asin
            """,
            [train_end_ms, decay_days, str(train_parquet)],
        ).fetchall()
    connection.close()

    scores = np.zeros(len(item_to_code), dtype=np.float32)
    for asin, value in rows:
        code = item_to_code.get(asin)
        if code is not None:
            scores[code] = float(value)
    model = PopularityModel(
        scores=scores, train_end_ms=train_end_ms, decayed=decay_days is not None
    )
    _logger.info(
        "popularity.trained",
        extra={
            "n_items": int((scores > 0).sum()),
            "decayed": model.decayed,
            "decay_days": decay_days,
        },
    )
    return model
