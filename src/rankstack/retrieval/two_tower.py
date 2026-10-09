"""Two-tower neural retrieval.

ALS asks the model to reconstruct the interaction matrix. This asks a different question: given a
user's history, which item should come next? That is the objective serving actually cares about, and
it is why a two-tower usually beats matrix factorisation on ranking metrics even when both see the
same interactions.

**Leave-one-out inside the train window.** For each eligible user the most recent train item is the
target and the remainder is the input, so the model never sees its own answer. This mirrors the
evaluation protocol, and it means no test-window data touches training at all.

**In-batch softmax with log-frequency correction.** Each batch of users supplies one positive item
per user, and those positives are each other's negatives. The loss is cross-entropy over the batch.
Because a popular item is drawn as a positive far more often than a rare one, the raw softmax is
biased towards popular items; subtracting `log q_i` from each logit removes that bias
(log-uniform sampling correction). `log_q_correction=False` turns the correction off, which is the
ablation that shows how much of the gain is the objective rather than the architecture.

**Mean pooling with recency decay.** The user tower is a weighted mean of the item tower's
embeddings, weighted by position with a geometric decay. No recurrence: a user history has no
fixed length, and on this corpus the recency signal is worth more than the ordering machinery a
GRU would add for 3M parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime by the writers

import numpy as np
import torch
from torch import nn

from rankstack.logging import get_logger

_logger = get_logger(__name__)

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
class TrainingReport:
    """What one training run measured, written into the results file verbatim."""

    steps: int
    final_loss: float
    device: str
    eligible_users: int

    def as_dict(self) -> dict[str, object]:
        """Return the report as a JSON-serialisable mapping."""
        return {
            "steps": self.steps,
            "final_loss": round(self.final_loss, 6),
            "device": self.device,
            "eligible_users": self.eligible_users,
        }


def recency_weights(length: int, decay: float, device: torch.device) -> torch.Tensor:
    """Position weights for a history, most recent first.

    `decay=1.0` gives every position equal weight, which is the ablation that isolates how much of
    the model's accuracy comes from recency ordering rather than from the collaborative signal.
    """
    positions = torch.arange(length, dtype=torch.float32, device=device)
    return torch.pow(torch.tensor(decay, dtype=torch.float32, device=device), positions)


def pad_histories(
    codes: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    *,
    width_limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a padded `(batch, width)` history matrix and its per-row lengths.

    Vectorised on purpose: slicing one history per row in Python costs more than the forward pass
    at batch 4096. Width is the batch's longest history capped at `width_limit`, so a batch of
    one-history users does not pay for the 50-column worst case.
    """
    width = int(min(width_limit, max(int(lengths.max(initial=1)), 1)))
    kept = np.minimum(lengths, width)
    offsets = np.arange(width, dtype=np.int64)[None, :]
    # Positions past a user's own history are masked out below, but the gather still evaluates them,
    # so the indices are clipped to stay inside `codes`.
    positions = np.minimum(starts[:, None] + offsets, codes.size - 1)
    return np.where(offsets < kept[:, None], codes[positions], 0), kept.astype(np.int64)


class TwoTower(nn.Module):
    """Item tower (an embedding table) and a mean-pooled user tower over it."""

    def __init__(self, n_items: int, config: TwoTowerConfig) -> None:
        """Build the item tower for `n_items` and remember the pooling configuration."""
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(n_items, config.factors)

    def item_vectors(self, codes: torch.Tensor) -> torch.Tensor:
        """L2-normalised item embeddings for `codes`, shape `(..., factors)`."""
        return torch.nn.functional.normalize(self.item_embedding(codes), dim=-1)

    def user_vector(self, history: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """User embedding from a padded history, shape `(batch, factors)`.

        `history` is `(batch, max_history)` of item codes, most recent first, and `lengths` says how
        many of those positions are real. Padded positions are excluded from both the numerator and
        the denominator, so a short history is not diluted by its own padding.
        """
        device = history.device
        # Pool the *normalised* item vectors, not the raw ones: scoring dot-products against
        # normalised items, so pooling raw vectors would train a different model than is served.
        embeddings = self.item_vectors(history)
        positions = torch.arange(history.shape[1], device=device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        weights = recency_weights(history.shape[1], self.config.recency_decay, device)
        weights = weights.unsqueeze(0) * valid.float()
        weighted = embeddings * weights.unsqueeze(-1)
        pooled = weighted.sum(dim=1) / weights.sum(dim=1).clamp(min=1e-12).unsqueeze(-1)
        return torch.nn.functional.normalize(pooled, dim=-1)


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


def train_two_tower(
    history_indptr: np.ndarray,
    history_codes: np.ndarray,
    n_users: int,
    n_items: int,
    item_log_popularity: np.ndarray,
    config: TwoTowerConfig | None = None,
    *,
    device: str | None = None,
) -> tuple[TwoTowerModel, TrainingReport]:
    """Fit the two-tower on the train-window histories and report the training curve.

    `item_log_popularity` is `log(count / total)` per item from the same history: the `log q`
    term of the sampling correction. Items with no interactions get a finite floor, so a
    cold-start item's logit is shifted rather than dropped.
    """
    settings = config or TwoTowerConfig()
    torch.manual_seed(settings.seed)
    target_device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))

    depths = np.diff(history_indptr)
    eligible = np.flatnonzero(depths >= settings.min_user_history)
    if eligible.size == 0:
        msg = f"no user has at least {settings.min_user_history} train interactions"
        raise ValueError(msg)
    _logger.info(
        "two_tower.eligible_users",
        extra={
            "eligible": int(eligible.size),
            "total": int(n_users),
            "fraction": round(float(eligible.size) / max(n_users, 1), 4),
        },
    )

    module = TwoTower(n_items, settings).to(target_device)
    optimiser = torch.optim.AdamW(
        module.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    # Most recent train item is the target; the rest is the input. Never the other way round.
    targets = history_codes[history_indptr[eligible]].astype(np.int64)
    starts = history_indptr[eligible] + 1
    lengths = depths[eligible] - 1

    log_q = torch.from_numpy(item_log_popularity.astype(np.float32)).to(target_device)
    rng = np.random.default_rng(settings.seed)
    steps_per_epoch = int(np.ceil(eligible.size / settings.batch_size))
    steps_done = 0

    module.train()
    for _epoch in range(settings.epochs):
        order = rng.permutation(eligible.size)
        running_loss = torch.zeros((), device=target_device)
        for step in range(steps_per_epoch):
            batch = order[step * settings.batch_size : (step + 1) * settings.batch_size]
            if batch.size == 0:
                break
            padded, row_lengths = pad_histories(
                history_codes, starts[batch], lengths[batch], width_limit=settings.max_history
            )

            batch_targets = torch.from_numpy(targets[batch]).to(target_device)
            users = module.user_vector(
                torch.from_numpy(padded).to(target_device),
                torch.from_numpy(row_lengths).to(target_device),
            )
            positives = module.item_vectors(batch_targets)
            logits = users @ positives.T / settings.temperature
            if settings.log_q_correction:
                # Column j holds the logit for the j-th user's positive, so subtracting that
                # item's log q cancels the popularity prior in that column.
                logits = logits - log_q[batch_targets].unsqueeze(0)
            labels = torch.arange(batch.size, device=target_device)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            optimiser.zero_grad(set_to_none=True)
            # torch annotates Tensor.backward as untyped in its own stubs.
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
            # Keep the loss as a tensor: reading it every step would force a device sync per batch.
            running_loss += loss.detach()
            steps_done += 1

    module.eval()
    embeddings = module.item_embedding.weight.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    model = TwoTowerModel(
        item_embeddings=embeddings / np.clip(norms, 1e-12, None),
        history_indptr=history_indptr,
        history_codes=history_codes,
        recency_decay=settings.recency_decay,
        max_history=settings.max_history,
        config=settings,
        trained_items=np.bincount(history_codes.astype(np.int64), minlength=n_items) > 0,
    )
    report = TrainingReport(
        steps=steps_done,
        final_loss=float(running_loss) / max(steps_done, 1),
        device=target_device.type,
        eligible_users=int(eligible.size),
    )
    _logger.info("two_tower.trained", extra=report.as_dict())
    return model, report


def item_log_popularity(history_codes: np.ndarray, n_items: int) -> np.ndarray:
    """`log(count / total)` per item, the sampling-bias term of the softmax correction.

    Items with no interactions get the smallest finite log value rather than `-inf`, so a cold-start
    item's logit is shifted instead of becoming NaN.
    """
    counts = np.bincount(history_codes.astype(np.int64), minlength=n_items).astype(np.float64)
    total = max(counts.sum(), 1.0)
    with np.errstate(divide="ignore"):
        values = np.log(np.clip(counts, 1.0, None) / total)
    return values.astype(np.float32)


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
