"""User and item segments.

Averages hide failures, so every result table is broken out by segment with its `n`. Segments come
from the **train** window only, because that is all a model may know at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LIGHT_USER_MIN = 2
LIGHT_USER_MAX = 9
HEAVY_USER_MIN = 20
POPULAR_TARGET_MIN = 100
LONG_TAIL_ITEM_MAX = 5


@dataclass(frozen=True, slots=True)
class SegmentMasks:
    """Boolean masks over cohort users, all derived from train-window history only."""

    new_user: np.ndarray
    light_user: np.ndarray
    heavy_user: np.ndarray
    long_tail_target: np.ndarray
    popular_target: np.ndarray
    depths: np.ndarray

    def as_dict(self) -> dict[str, int]:
        """Return the size of each segment."""
        return {
            "new_user": int(self.new_user.sum()),
            "light_user": int(self.light_user.sum()),
            "heavy_user": int(self.heavy_user.sum()),
            "long_tail_target": int(self.long_tail_target.sum()),
            "popular_target": int(self.popular_target.sum()),
        }


def segment_masks(
    user_depths: np.ndarray,
    target_popularity: np.ndarray,
) -> SegmentMasks:
    """Build the segment masks for one cohort.

    `user_depths` counts the user's train-window interactions (0 means the `new_user` segment) and
    `target_popularity` is the train-window interaction count of the held-out item.
    """
    depths = np.asarray(user_depths, dtype=np.int64)
    popularity = np.asarray(target_popularity, dtype=np.float64)
    return SegmentMasks(
        new_user=depths == 0,
        light_user=(depths >= LIGHT_USER_MIN) & (depths <= LIGHT_USER_MAX),
        heavy_user=depths >= HEAVY_USER_MIN,
        long_tail_target=popularity <= LONG_TAIL_ITEM_MAX,
        popular_target=popularity > POPULAR_TARGET_MIN,
        depths=depths,
    )
