"""Process-wide settings and filesystem layout.

One place decides where data, artefacts and results live, so that a run is reproducible from a
single source of truth. Nothing else in the codebase builds a path by hand.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root, used to locate committed directories from an installed package.
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings. Every field can be overridden by an environment variable."""

    model_config = SettingsConfigDict(env_prefix="RANKSTACK_", extra="ignore")

    data_root: Path = Field(default=REPO_ROOT / "data")
    """Raw and processed dataset files. Never committed."""

    artifact_root: Path = Field(default=REPO_ROOT / "artifacts")
    """Model, index and embedding artefacts. Never committed."""

    results_root: Path = Field(default=REPO_ROOT / "results")
    """Committed result files. Every published number must come from here."""

    seed: int = 20261008
    """Default seed. Recorded in every manifest so a run can be repeated exactly."""

    download_chunk_bytes: int = 8 * 1024 * 1024
    """Streaming chunk size for dataset downloads."""

    fusion_cooccurrence_weight: float = Field(default=0.75, ge=0.0)
    """Weight of co-occurrence in the served fusion for known users; 0 serves popularity alone.

    Chosen by the pre-registered rule in docs/adr/0007-serving-fusion-weight.md, not by hand:
    `results/fusion_weight_sweep_electronics_0core.json#/selected_weight`.
    """

    @cached_property
    def processed_root(self) -> Path:
        """Where validated Parquet and manifests for every corpus are written."""
        return self.data_root / "processed"

    def processed(self, *parts: str) -> Path:
        """Path inside the processed-data tree."""
        return self.processed_root.joinpath(*parts)

    def raw(self, *parts: str) -> Path:
        """Path inside the raw-data tree."""
        return self.data_root.joinpath(*parts)

    def ensure_directories(self) -> None:
        """Create the writable roots. Safe to call repeatedly."""
        for path in (self.data_root, self.artifact_root, self.results_root, self.processed_root):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
