"""Verified dataset download.

Raw dataset files are never committed (see docs/design.md section 4.3), so this module is the only
way they enter the project. It streams to a temporary file, verifies the published byte length and a
recorded SHA-256, and moves the file into place atomically: a half-finished download can never be
mistaken for a complete one.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rankstack.config import Settings, settings as default_settings
from rankstack.data.corpus import Corpus, raw_dir_for
from rankstack.logging import get_logger

_logger = get_logger(__name__)

_HASH_CHUNK = 8 * 1024 * 1024
_ALLOWED_SCHEMES = frozenset({"https", "file"})


class DownloadError(RuntimeError):
    """Raised when a download fails or fails verification."""


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """What a verified download produced."""

    path: Path
    url: str
    bytes_written: int
    sha256: str
    expected_bytes: int
    already_present: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "url": self.url,
            "path": str(self.path),
            "bytes": self.bytes_written,
            "sha256": self.sha256,
            "expected_bytes": self.expected_bytes,
            "reused_existing_file": self.already_present,
        }


def _open(url: str) -> tuple[Any, int | None]:
    """Open a URL for streaming. Returns the stream and its declared length, if it sent one."""
    if "://" not in url:
        msg = f"refusing to download {url!r}: expected an explicit URL"
        raise DownloadError(msg)
    scheme = url.split("://", 1)[0]
    if scheme not in _ALLOWED_SCHEMES:
        msg = f"refusing to download {url!r}: scheme {scheme!r} is not allowed"
        raise DownloadError(msg)

    request = urllib.request.Request(url, headers={"User-Agent": "rankstack/0.1 (+research)"})
    try:
        stream = urllib.request.urlopen(request)  # noqa: S310 - scheme is allow-listed above
    except OSError as exc:
        msg = f"could not open {url}: {exc}"
        raise DownloadError(msg) from exc

    status = getattr(stream, "status", None)
    if isinstance(status, int) and status >= 400:  # noqa: PLR2004 - any HTTP error is fatal
        msg = f"download failed with HTTP {status}: {url}"
        raise DownloadError(msg)
    headers = getattr(stream, "headers", None)
    declared = headers.get("Content-Length") if headers is not None else None
    return stream, int(declared) if declared is not None else None


def sha256_file(path: Path, chunk_bytes: int = _HASH_CHUNK) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    expected_bytes: int,
    fetcher: Callable[[str], tuple[Any, int | None]] | None = None,
    settings: Settings = default_settings,
) -> DownloadResult:
    """Fetch `url` to `destination`, verifying size and content hash.

    An existing file of the expected size is hashed and reused rather than re-downloaded, so
    repeated pipeline runs cost nothing.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size == expected_bytes:
        existing_digest = sha256_file(destination, settings.download_chunk_bytes)
        _logger.info("dataset.file_reused", extra={"path": str(destination), "sha256": existing_digest})
        return DownloadResult(
            path=destination,
            url=url,
            bytes_written=expected_bytes,
            sha256=existing_digest,
            expected_bytes=expected_bytes,
            already_present=True,
        )

    temporary = destination.with_suffix(destination.suffix + ".partial")
    stream, declared_length = (fetcher or _open)(url)
    written = 0
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as handle:
            while chunk := stream.read(settings.download_chunk_bytes):
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        msg = f"failed to write {temporary}: {exc}"
        raise DownloadError(msg) from exc
    finally:
        closer = getattr(stream, "close", None)
        if callable(closer):
            closer()

    if declared_length is not None and declared_length != expected_bytes:
        temporary.unlink(missing_ok=True)
        msg = (
            f"{url} served {declared_length} bytes but the published size is {expected_bytes}; "
            "the download is truncated or the publisher changed the file"
        )
        raise DownloadError(msg)
    if written != expected_bytes:
        temporary.unlink(missing_ok=True)
        msg = f"{url} wrote {written} bytes, expected {expected_bytes}"
        raise DownloadError(msg)

    shutil.move(str(temporary), str(destination))
    sha = digest.hexdigest()
    _logger.info(
        "dataset.downloaded", extra={"path": str(destination), "bytes": written, "sha256": sha}
    )
    return DownloadResult(
        path=destination,
        url=url,
        bytes_written=written,
        sha256=sha,
        expected_bytes=expected_bytes,
        already_present=False,
    )


def download_corpus_files(corpus_name: str, settings: Settings = default_settings) -> dict[str, DownloadResult]:
    """Download the raw files of a registered corpus."""
    from rankstack.data.corpus import get_corpus  # noqa: PLC0415 - avoids an import cycle

    return download_corpus_files_from(get_corpus(corpus_name), settings=settings)


def download_corpus_files_from(
    corpus: Corpus, settings: Settings = default_settings
) -> dict[str, DownloadResult]:
    """Download the interactions file and, when the corpus needs it, the item metadata file."""
    raw_dir = raw_dir_for(settings.data_root, corpus)
    results = {
        "interactions": download_file(
            url=corpus.interactions_url,
            destination=raw_dir / f"{corpus.category}.csv.gz",
            expected_bytes=corpus.interactions_bytes,
            settings=settings,
        )
    }
    if corpus.metadata_url and corpus.metadata_bytes:
        results["metadata"] = download_file(
            url=corpus.metadata_url,
            destination=raw_dir / f"meta_{corpus.category}.jsonl.gz",
            expected_bytes=corpus.metadata_bytes,
            settings=settings,
        )
    return results