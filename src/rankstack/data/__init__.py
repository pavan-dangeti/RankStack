"""Data pipeline: download, validate, split, and materialise point-in-time features."""

from rankstack.data.build import build_dataset, load_manifest, summarise
from rankstack.data.corpus import CORPORA, DEFAULT_CORPUS, Corpus, get_corpus
from rankstack.data.download import DownloadResult, download_corpus_files
from rankstack.data.leakage import LeakageError, LeakageReport
from rankstack.data.schema import ValidationError

__all__ = [
    "CORPORA",
    "DEFAULT_CORPUS",
    "Corpus",
    "DownloadResult",
    "LeakageError",
    "LeakageReport",
    "ValidationError",
    "build_dataset",
    "download_corpus_files",
    "get_corpus",
    "load_manifest",
    "summarise",
]