"""Implicit ALS.

Weighted matrix factorisation on the binary train matrix. Two choices are recorded because they
shape the result:

- **Users with a single train interaction are excluded.** Their user factor carries no information
  beyond that one item, so training on them adds cost without signal. They are still *evaluated* —
  they simply cannot be served by this source, which the harness reports as a served fraction.
- **Factors are capped at 64.** 15.4M train users at 128 factors is 7.9 GB of user factors alone,
  which does not fit the development machine. 64 factors is the largest power-of-two block that
  fits alongside the matrix, and the resulting model is compared against `implicit`'s own
  recommendations on the validation window before it is accepted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime by the writers

import numpy as np
import scipy.sparse as sp

from rankstack.logging import get_logger

_logger = get_logger(__name__)

DEFAULT_FACTORS = 64
DEFAULT_ITERATIONS = 15
DEFAULT_REGULARISATION = 0.01
DEFAULT_ALPHA = 40.0
DEFAULT_MIN_USER_HISTORY = 2


@dataclass(frozen=True, slots=True)
class AlsConfig:
    """Hyperparameters for the ALS fit, reported in the result file."""

    factors: int = DEFAULT_FACTORS
    iterations: int = DEFAULT_ITERATIONS
    regularisation: float = DEFAULT_REGULARISATION
    alpha: float = DEFAULT_ALPHA
    min_user_history: int = DEFAULT_MIN_USER_HISTORY
    seed: int = 20261008

    def as_dict(self) -> dict[str, object]:
        """Return the hyperparameters as a JSON-serialisable mapping."""
        return {
            "factors": self.factors,
            "iterations": self.iterations,
            "regularisation": self.regularisation,
            "alpha": self.alpha,
            "min_user_history": self.min_user_history,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class AlsModel:
    """User and item factors plus the history each user was trained on."""

    user_factors: np.ndarray
    item_factors: np.ndarray
    history_indptr: np.ndarray
    history_codes: np.ndarray
    min_user_history: int
    factors: int
    iterations: int

    @property
    def name(self) -> str:
        """Name reported in results files."""
        return "als"

    def n_items(self) -> int:
        """Return the number of scorable items."""
        return int(self.item_factors.shape[0])

    def can_serve(self, user_code: int) -> bool:
        """Return True when the user was eligible for training and has a non-zero factor."""
        return self.user_factors.shape[0] > user_code >= 0 and bool(
            np.any(self.user_factors[user_code])
        )

    def history(self, user_code: int) -> np.ndarray:
        """Return the item codes this user interacted with in the train window."""
        start, end = self.history_indptr[user_code], self.history_indptr[user_code + 1]
        return self.history_codes[start:end]

    def history_length(self, user_code: int) -> int:
        """Return the size of this user's train-window history."""
        start, end = self.history_indptr[user_code], self.history_indptr[user_code + 1]
        return int(end - start)

    def score(self, user_code: int, seen: np.ndarray) -> np.ndarray:
        """Score every item as the dot product of the user's and the item's factor vector."""
        if not self.can_serve(user_code):
            scores = np.full(self.n_items(), -np.inf, dtype=np.float32)
            scores[seen] = -np.inf
            return scores
        scores = self.user_factors[user_code] @ self.item_factors.T
        scores = np.asarray(scores, dtype=np.float32)
        scores[seen] = -np.inf
        return scores


def _csr_from_history(
    history_indptr: np.ndarray, history_codes: np.ndarray, n_users: int, n_items: int
) -> sp.csr_matrix:
    rows = np.repeat(np.arange(n_users, dtype=np.int32), np.diff(history_indptr))
    data = np.ones(history_codes.size, dtype=np.float32)
    return sp.csr_matrix((data, (rows, history_codes)), shape=(n_users, n_items))


def train_als(
    history_indptr: np.ndarray,
    history_codes: np.ndarray,
    n_users: int,
    n_items: int,
    config: AlsConfig | None = None,
) -> AlsModel:
    """Fit implicit ALS on the binary history, keeping factors only for eligible users.

    The returned user factors are indexed by the *original* user code, with ineligible users left as
    zero rows, so downstream code can use one code space for every source.
    """
    from implicit.als import (  # noqa: PLC0415 - keeps the heavy import off module import
        AlternatingLeastSquares,
    )

    settings = config or AlsConfig()
    depths = np.diff(history_indptr)
    eligible = np.flatnonzero(depths >= settings.min_user_history)
    _logger.info(
        "als.eligible_users",
        extra={
            "eligible": int(eligible.size),
            "total": int(n_users),
            "fraction": round(float(eligible.size) / max(n_users, 1), 4),
        },
    )

    sub_indptr = np.zeros(eligible.size + 1, dtype=np.int64)
    sub_codes: list[np.ndarray] = []
    for position, user_code in enumerate(eligible):
        start, end = history_indptr[user_code], history_indptr[user_code + 1]
        sub_codes.append(history_codes[start:end])
        sub_indptr[position + 1] = sub_indptr[position] + (end - start)
    codes = np.concatenate(sub_codes) if sub_codes else np.empty(0, dtype=np.int32)

    # Items that only ever appear among single-interaction users have no eligible user at all, so
    # they are empty columns. ALS solves an unregularised system for those and returns NaN, which
    # `implicit` raises on. Fit over the live items and scatter the factors back, leaving dead items
    # with a zero vector: they carry no collaborative signal, which is the truth about them.
    live = np.zeros(n_items, dtype=bool)
    live[codes] = True
    live_codes = np.flatnonzero(live)
    column_of = np.full(n_items, -1, dtype=np.int64)
    column_of[live_codes] = np.arange(live_codes.size, dtype=np.int64)
    matrix = _csr_from_history(sub_indptr, column_of[codes], eligible.size, live_codes.size)
    matrix.eliminate_zeros()
    _logger.info(
        "als.live_items",
        extra={"live_items": int(live_codes.size), "dead_items": int(n_items - live_codes.size)},
    )

    model = AlternatingLeastSquares(
        factors=settings.factors,
        regularization=settings.regularisation,
        alpha=settings.alpha,
        iterations=settings.iterations,
        random_state=settings.seed,
    )
    model.fit(matrix, show_progress=False)

    user_factors = np.zeros((n_users, settings.factors), dtype=np.float32)
    user_factors[eligible] = model.user_factors.astype(np.float32)
    item_factors = np.zeros((n_items, settings.factors), dtype=np.float32)
    live_factors = np.asarray(model.item_factors, dtype=np.float32)
    # Defensive: a non-finite factor would poison every score it touches, so it is zeroed and
    # counted rather than propagated.
    non_finite = int((~np.isfinite(live_factors)).any(axis=1).sum())
    if non_finite:
        _logger.warning("als.non_finite_item_factors", extra={"rows_zeroed": non_finite})
        live_factors[~np.isfinite(live_factors).all(axis=1)] = 0.0
    item_factors[live_codes] = live_factors

    fitted = AlsModel(
        user_factors=user_factors,
        item_factors=item_factors,
        history_indptr=history_indptr,
        history_codes=history_codes,
        min_user_history=settings.min_user_history,
        factors=settings.factors,
        iterations=settings.iterations,
    )
    _logger.info(
        "als.trained",
        extra={
            "factors": settings.factors,
            "iterations": settings.iterations,
            "user_factor_mb": round(user_factors.nbytes / 1e6, 1),
            "item_factor_mb": round(item_factors.nbytes / 1e6, 1),
        },
    )
    return fitted


def write_model(model: AlsModel, path: Path) -> Path:
    """Load a fitted model from its artefact directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        user_factors=model.user_factors,
        item_factors=model.item_factors,
        history_indptr=model.history_indptr,
        history_codes=model.history_codes,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "factors": model.factors,
                "iterations": model.iterations,
                "min_user_history": model.min_user_history,
                "n_users": int(model.user_factors.shape[0]),
                "n_items": model.n_items(),
            },
            sort_keys=True,
        )
    )
    return path


def load_model(path: Path) -> AlsModel:
    """Load a fitted model from its artefact directory."""
    payload = np.load(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    return AlsModel(
        user_factors=payload["user_factors"],
        item_factors=payload["item_factors"],
        history_indptr=payload["history_indptr"],
        history_codes=payload["history_codes"],
        min_user_history=int(metadata["min_user_history"]),
        factors=int(metadata["factors"]),
        iterations=int(metadata["iterations"]),
    )
