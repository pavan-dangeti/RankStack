"""Hash the serving artefacts so the service can refuse to start on anything unverified.

Run after `scripts/train_baselines.py`. The manifest is what makes the serving layer's integrity
claim checkable: it records the SHA-256 of every artefact it loads, and `load_artifacts` refuses any
file whose hash has moved since.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rankstack.config import settings
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, get_corpus
from rankstack.logging import configure_logging, get_logger
from rankstack.serving.artifacts import MANIFEST_NAME, build_manifest, verify

_logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the serving manifest and immediately verify it reads back."""
    args = parse_args(argv)
    configure_logging()
    corpus = get_corpus(args.corpus)
    root = settings.artifact_root / corpus.name
    if not root.exists():
        print(f"error: no artefacts at {root}", file=sys.stderr)
        return 1

    path = build_manifest(root)
    manifest = json.loads(Path(path).read_text())
    _logger.info(
        "serving.manifest_built",
        extra={"path": str(path), "files": len(manifest["files"])},
    )
    print(f"wrote {path}")
    for name, digest in sorted(manifest["files"].items()):
        print(f"  {digest[:16]}  {name}")
    verify(root)
    print(f"\nverified {len(verify(root))} artefact(s) against {MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
