"""Serving artefacts, loaded once and verified before use.

The serving layer must not start on artefacts it cannot vouch for. Each file is hashed with SHA-256
and compared against `artifacts/<corpus>/serving_manifest.json`; a mismatch refuses to load rather
than silently serving stale or half-written models. This is the cheap half of the model-registry
contract in the design document: content addressing, checked at load time rather than trusted from a
comment.

Titles are optional and separately budgeted. 1.6M product strings cost roughly 150 MB, which fits
the 16 GB target host but not a 512 MB free tier, so `with_titles` is a deliberate switch rather
than a default the deployer has to discover the hard way.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime by the readers and writers
from typing import TYPE_CHECKING, cast

import duckdb
import numpy as np
import scipy.sparse as sp

from rankstack.config import Settings
from rankstack.config import settings as default_settings
from rankstack.data.encode import Encoding, load_encoding
from rankstack.logging import get_logger
from rankstack.rerank.features import CodeLookup
from rankstack.retrieval.cooccurrence import CooccurrenceModel
from rankstack.retrieval.fusion import DEFAULT_DEPTH, DEFAULT_RRF_K, FusedSource
from rankstack.retrieval.popularity import PopularityModel

if TYPE_CHECKING:
    from rankstack.retrieval.base import CandidateSource

_logger = get_logger(__name__)

MANIFEST_NAME = "serving_manifest.json"
_DIGEST_CHUNK = 1 << 20
_HASHED_ARTEFACTS = (
    "encoding.json",
    "popularity.npy",
    "popularity.json",
    "popularity_recent.npy",
    "popularity_recent.json",
    "cooccurrence.npz",
    "cooccurrence.json",
    "cooccurrence.history.npz",
)
#: A module constant, not a caller-supplied value: the only interpolation in the query below.
_CATALOGUE_COLUMNS = "parent_asin, main_category, price, title"


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks so a large artefact is not slurped."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_DIGEST_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ItemCatalogue:
    """Item display fields, indexed by item code.

    `titles` is `None` when the process was started without them, which is a supported
    configuration rather than an error: the API then returns `title: null`.
    """

    titles: np.ndarray | None
    categories: np.ndarray
    prices: np.ndarray

    def describe(self, code: int) -> tuple[str | None, str | None, float | None]:
        """Return `(title, category, price)` for one item code, normalising empties to `None`."""
        title = None if self.titles is None else str(self.titles[code])
        category = str(self.categories[code]) if self.categories[code] else None
        price = float(self.prices[code]) if self.prices[code] > 0 else None
        return title or None, category, price


@dataclass(frozen=True, slots=True)
class ServingArtifacts:
    """Everything the request path needs, loaded and checksum-verified up front."""

    encoding: Encoding
    popularity: PopularityModel
    popularity_recent: PopularityModel
    cooccurrence: CooccurrenceModel | None
    catalogue: ItemCatalogue
    history_indptr: np.ndarray
    history_codes: np.ndarray
    corpus: str
    versions: dict[str, str]

    @property
    def n_items(self) -> int:
        """Return the size of the catalogue being served."""
        return self.encoding.n_items

    def sources(self) -> dict[str, CandidateSource]:
        """Return the retrievers available to the request path, strongest first.

        Co-occurrence is optional: the popularity pair alone clears the latency SLO, so a
        memory-constrained deploy can leave the 45M-non-zero matrix out.
        """
        available: dict[str, CandidateSource] = {
            "popularity_recent": self.popularity_recent,
            "popularity": self.popularity,
        }
        if self.cooccurrence is not None:
            available["cooccurrence"] = self.cooccurrence
        return available

    def served_source(self) -> CandidateSource:
        """Return what the request path actually serves.

        Fusion, not the single best-scoring retriever. `popularity_recent` has the best hit rate
        (HR@20 0.0200 against fusion's 0.0150 on the warm cohort) but it is a *global* ranking: it
        returned only 3 distinct top items across 2,000 warm users. Fusion returns 18 and costs 25%
        of the hit rate. A recommender that shows everyone the same list is not a
        recommender, so the trade is taken deliberately and its cost is published rather than
        discovered.

        Falls back to the global ranking when co-occurrence was not loaded, which is the
        memory-constrained configuration.
        """
        if self.cooccurrence is None:
            return self.popularity_recent
        fused_sources = {
            "popularity_recent": self.popularity_recent,
            "cooccurrence": self.cooccurrence,
        }
        return FusedSource(sources=fused_sources, rrf_k=DEFAULT_RRF_K, depth=DEFAULT_DEPTH)

    def seen_mask(self, user_code: int) -> np.ndarray:
        """Return the boolean mask of items the user has already interacted with."""
        mask = np.zeros(self.n_items, dtype=bool)
        start, end = self.history_indptr[user_code], self.history_indptr[user_code + 1]
        if end > start:
            mask[self.history_codes[start:end]] = True
        return mask


def build_manifest(root: Path) -> Path:
    """Hash every serving artefact and write the manifest the loader verifies against."""
    entries: dict[str, str] = {}
    for name in _HASHED_ARTEFACTS:
        path = root / name
        if path.exists():
            entries[name] = sha256_file(path)
    path = root / MANIFEST_NAME
    path.write_text(json.dumps({"files": entries}, indent=2, sort_keys=True) + "\n")
    _logger.info("serving.manifest_written", extra={"files": len(entries)})
    return path


def verify(root: Path) -> dict[str, str]:
    """Return the manifest, raising when a listed artefact is missing or has changed."""
    path = root / MANIFEST_NAME
    if not path.exists():
        msg = f"no serving manifest at {path}; run scripts/build_serving_manifest.py first"
        raise FileNotFoundError(msg)
    manifest = json.loads(path.read_text())
    files = cast("dict[str, str]", manifest.get("files", {}))
    for name, expected in files.items():
        artefact = root / name
        if not artefact.exists():
            msg = f"serving manifest lists {name} but it is missing"
            raise FileNotFoundError(msg)
        if sha256_file(artefact) != expected:
            msg = f"{name} does not match the serving manifest"
            raise ValueError(msg)
    return files


def load_artifacts(
    artifact_root: Path,
    corpus: str,
    *,
    with_titles: bool = True,
    verify_checksums: bool = True,
    config: Settings | None = None,
) -> ServingArtifacts:
    """Load and verify every serving artefact for one corpus."""
    root = artifact_root / corpus
    versions = verify(root) if verify_checksums else {}
    if not verify_checksums and (root / MANIFEST_NAME).exists():
        versions = cast("dict[str, str]", json.loads((root / MANIFEST_NAME).read_text())["files"])

    encoding = load_encoding(artifact_root, corpus)
    history_path = root / "cooccurrence.history.npz"
    if not history_path.exists():
        msg = f"no serving history at {history_path}"
        raise FileNotFoundError(msg)
    history = np.load(history_path)
    history_indptr = np.asarray(history["history_indptr"])
    history_codes = np.asarray(history["history_codes"])

    cooccurrence = None
    if (root / "cooccurrence.npz").exists():
        metadata = json.loads((root / "cooccurrence.json").read_text())
        cooccurrence = CooccurrenceModel(
            similarity=sp.load_npz(root / "cooccurrence.npz"),
            item_popularity=history["item_popularity"],
            history_indptr=history_indptr,
            history_codes=history_codes,
            max_history=int(metadata["max_history"]),
            top_k=int(metadata["top_k"]),
            user_block=int(metadata["user_block"]),
        )

    catalogue = load_catalogue(corpus, encoding, with_titles=with_titles, config=config)
    artifacts = ServingArtifacts(
        encoding=encoding,
        popularity=PopularityModel.load(root / "popularity.npy"),
        popularity_recent=PopularityModel.load(root / "popularity_recent.npy"),
        cooccurrence=cooccurrence,
        catalogue=catalogue,
        history_indptr=history_indptr,
        history_codes=history_codes,
        corpus=corpus,
        versions=versions,
    )
    _logger.info(
        "serving.artifacts_loaded",
        extra={
            "corpus": corpus,
            "n_items": artifacts.n_items,
            "cooccurrence": cooccurrence is not None,
            "titles": catalogue.titles is not None,
        },
    )
    return artifacts


def load_catalogue(
    corpus: str,
    encoding: Encoding,
    *,
    with_titles: bool,
    config: Settings | None = None,
) -> ItemCatalogue:
    """Read item display fields as a code-indexed table."""
    resolved = config or default_settings
    items_path = resolved.processed(corpus) / "items" / "items.parquet"
    if not items_path.exists():
        msg = f"no item metadata at {items_path}; run the data build first"
        raise FileNotFoundError(msg)

    connection = duckdb.connect(database=":memory:")
    table = (
        connection.execute(
            f"SELECT {_CATALOGUE_COLUMNS} FROM read_parquet(?)",  # noqa: S608 - constant columns
            [items_path.as_posix()],
        )
        .arrow()
        .read_all()
    )
    connection.close()

    n_items = encoding.n_items
    titles = np.full(n_items, "", dtype=object) if with_titles else None
    categories = np.full(n_items, "", dtype=object)
    prices = np.zeros(n_items, dtype=np.float32)
    codes = CodeLookup(encoding.item_to_code).codes_for(
        table.column("parent_asin").to_numpy(zero_copy_only=False)
    )
    rows = np.flatnonzero(codes >= 0)
    targets = codes[rows].astype(np.int64)

    raw_categories = table.column("main_category").to_numpy(zero_copy_only=False)
    categories[targets] = ["" if v is None else str(v) for v in raw_categories[rows]]
    raw_prices = table.column("price").to_numpy(zero_copy_only=False)
    prices[targets] = [0.0 if v is None else float(v) for v in raw_prices[rows]]
    if titles is not None:
        raw_titles = table.column("title").to_numpy(zero_copy_only=False)
        titles[targets] = ["" if v is None else str(v) for v in raw_titles[rows]]
    return ItemCatalogue(titles=titles, categories=categories, prices=prices)
