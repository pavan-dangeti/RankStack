"""The trained two-tower as numpy arrays: configuration, user pooling, scoring and persistence.

Kept apart from the PyTorch training code so that anything which only *uses* a trained model
(evaluation, the ANN benchmark, serving) does not import torch. On macOS that is a correctness
issue, not just a size one: torch and faiss each bundle their own `libomp`, and a process that runs
faiss with torch loaded aborts at OpenMP initialisation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime by the writers

import numpy as np

DEFAULT_FACTORS = 128
DEFAULT_BATCH_SIZE = 4_096
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-6
DEFAULT_EPOCHS = 1
DEFAULT_MIN_USER_HISTORY = 2
DEFAULT_MAX_HISTORY = 50
DEFAULT_RECENCY_DECAY = 0.9
DEFAULT_TEMPERATURE = 0.05
DEFAULT_SEED = 20261008


@dataclass(frozen=True, slots=True)
class TwoTowerConfig:
    """Hyperparameters for the two-tower fit, reported in the result file.

    `temperature` is a hyperparameter rather than a constant because the sampled-softmax gradient
    scale depends on it, and it is the first thing to tune when the loss does not move.
    """

    factors: int = DEFAULT_FACTORS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    epochs: int = DEFAULT_EPOCHS
    max_history: int = DEFAULT_MAX_HISTORY
    recency_decay: float = DEFAULT_RECENCY_DECAY
    temperature: float = DEFAULT_TEMPERATURE
    min_user_history: int = DEFAULT_MIN_USER_HISTORY
    log_q_correction: bool = True
    seed: int = DEFAULT_SEED

    def as_dict(self) -> dict[str, object]:
        """Return the hyperparameters as a JSON-serialisable mapping."""
        return {
            "factors": self.factors,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "max_history": self.max_history,
            "recency_decay": self.recency_decay,
            "temperature": self.temperature,
            "min_user_history": self.min_user_history,
            "log_q_correction": self.log_q_correction,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class TwoTowerModel:
    """A trained two-tower plus the score-side view the harness needs.

    `item_vectors` is stored transposed and contiguous so scoring a user against the whole catalogue
    is one matrix-vector product rather than a gather per item.
    """

    item_embeddings: np.ndarray
    history_indptr: np.ndarray
    history_codes: np.ndarray
    recency_decay: float
    max_history: int
    config: TwoTowerConfig
    trained_items: np.ndarray | None = None

    @property
    def name(self) -> str:
        """Name reported in results files."""
        return "two_tower"

    def n_items(self) -> int:
        """Return the number of scorable items."""
        return int(self.item_embeddings.shape[0])

    def can_serve(self, user_code: int) -> bool:
        """Return True when the user has at least one train-window item."""
        return self.history_length(user_code) >= 1

    def history(self, user_code: int) -> np.ndarray:
        """Return the item codes this user interacted with, most recent first."""
        start, end = self.history_indptr[user_code], self.history_indptr[user_code + 1]
        return self.history_codes[start:end]

    def history_length(self, user_code: int) -> int:
        """Return the size of this user's capped history."""
        return int(self.history_indptr[user_code + 1] - self.history_indptr[user_code])

    def user_vectors(self, user_codes: np.ndarray) -> np.ndarray:
        """Embed a batch of users with the same recency-weighted mean pool used in training.

        Implemented in numpy rather than by rebuilding a `TwoTower`: scoring runs once per
        evaluation user, and constructing a module per user would dominate the loop.
        """
        lengths = np.diff(self.history_indptr)[user_codes]
        width = int(min(self.max_history, max(int(lengths.max(initial=0)), 1)))
        padded = np.zeros((user_codes.size, width), dtype=np.int64)
        mask = np.zeros((user_codes.size, width), dtype=np.float32)
        for row, code in enumerate(user_codes):
            items = self.history(int(code))[:width]
            padded[row, : items.size] = items
            mask[row, : items.size] = 1.0

        weights = mask * np.power(self.recency_decay, np.arange(width, dtype=np.float32)[None, :])
        embeddings = self.item_embeddings[padded]
        pooled = (embeddings * weights[:, :, None]).sum(axis=1) / np.clip(
            weights.sum(axis=1, keepdims=True), 1e-12, None
        )
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return np.asarray(pooled / np.clip(norms, 1e-12, None), dtype=np.float32)

    def score(self, user_code: int, seen: np.ndarray) -> np.ndarray:
        """Score every catalogue item as the dot product of the two towers' embeddings.

        Items the model never trained on are returned as `-inf` rather than at their raw score. A
        zero embedding scores exactly 0.0, and because a recency-weighted mean of unit vectors is
        usually non-positive, that 0.0 outranks almost every real item: on Electronics 0-core the
        117,011 untrained items filled the top of the ranking and froze the same order into every
        top-K, which is why every metric came out identical across training runs. A collaborative
        model has nothing to say about an item it has never seen, so it must not offer one.
        """
        user_vector = self.user_vectors(np.array([user_code]))[0]
        scores = self.item_embeddings @ user_vector
        if self.trained_items is not None:
            scores[~self.trained_items] = -np.inf
        scores[seen] = -np.inf
        return np.asarray(scores, dtype=np.float32)


def write_model(model: TwoTowerModel, path: Path) -> Path:
    """Persist embeddings, history and parameters to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    trained = model.trained_items
    np.savez(
        path,
        item_embeddings=model.item_embeddings,
        history_indptr=model.history_indptr,
        history_codes=model.history_codes,
        trained_items=(np.ones(model.n_items(), dtype=bool) if trained is None else trained),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "recency_decay": model.recency_decay,
                "max_history": model.max_history,
                "n_items": model.n_items(),
                "config": model.config.as_dict(),
            },
            sort_keys=True,
        )
    )
    return path


def load_model(path: Path) -> TwoTowerModel:
    """Load a trained model from its artefact directory."""
    payload = np.load(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    return TwoTowerModel(
        item_embeddings=payload["item_embeddings"],
        history_indptr=payload["history_indptr"],
        history_codes=payload["history_codes"],
        recency_decay=float(metadata["recency_decay"]),
        max_history=int(metadata["max_history"]),
        config=TwoTowerConfig(**metadata["config"]),
        trained_items=payload["trained_items"],
    )
