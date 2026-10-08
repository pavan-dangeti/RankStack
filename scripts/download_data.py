"""Download and verify raw dataset files.

Raw dataset files are never committed; this script is the only supported way to obtain them.
"""

from __future__ import annotations

import argparse
import sys

from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, publisher_issued_range
from rankstack.data.download import DownloadError, download_corpus_files
from rankstack.logging import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the script and return a process exit code."""
    args = parse_args(argv)
    configure_logging()
    print(f"publisher timestamp range: {publisher_issued_range()}")
    try:
        results = download_corpus_files(args.corpus)
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for label, result in results.items():
        state = "reused" if result.already_present else "downloaded"
        print(
            f"{label:12s} {state:10s} {result.bytes_written:>13,} bytes"
            f"  sha256={result.sha256[:16]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
