"""Integer encodings and evaluation cohorts.

Model code wants dense integer ids; the pipeline stores strings. This module is the only place the
two meet, and it draws the code space from the **train partition only**, so a user or item first
seen in the test window has no code and no collaborative model can serve it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 - used at runtime by the writers

import duckdb
import numpy as np

from rankstack.config import Settings
from rankstack.config import settings as default_settings
from rankstack.logging import get_logger

_logger = get_logger(__name__)

CODEC_PATH = "encoding.json"
CODEC_SCHEMA_VERSION = 1
#: What serving loads instead of `CODEC_PATH`: the code space without its user map, and the user
#: map as a `CodeIndex`.
SERVING_CODEC_PATH = "encoding.serving.json"
USER_INDEX_PATH = "users.npz"


class CodeIndex(Mapping[str, int]):
    """An id-to-code map held as two arrays and searched by bisection.

    A dict of fifteen million ids holds a Python string and a Python int per entry, gigabytes for
    the user map alone. This holds each id once as fixed-width UTF-8 bytes in a sorted array, with
    the codes alongside, and finds an id by binary search. Iteration yields ids in code order, the
    order a dict built from codes 0..n-1 iterates in.
    """

    __slots__ = ("_by_code", "_codes", "_keys")

    def __init__(self, keys: np.ndarray, codes: np.ndarray) -> None:
        """Wrap ids already sorted bytewise (`np.bytes_` array) and their aligned codes."""
        if keys.shape != codes.shape:
            msg = f"{keys.shape[0]} ids but {codes.shape[0]} codes"
            raise ValueError(msg)
        self._keys = keys
        self._codes = codes
        self._by_code = np.argsort(codes, kind="stable")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, int]) -> CodeIndex:
        """Build the index from any id-to-code mapping."""
        encoded = [key.encode("utf-8") for key in mapping]
        if any(key.endswith(b"\x00") for key in encoded):
            # Fixed-width byte arrays drop trailing NULs, so such an id could not be told apart.
            msg = "an id ending in a NUL byte cannot be stored in a CodeIndex"
            raise ValueError(msg)
        keys = np.array(encoded, dtype=np.bytes_) if encoded else np.array([], dtype="S1")
        codes = np.fromiter(mapping.values(), dtype=np.int64, count=len(encoded))
        order = np.argsort(keys, kind="stable")
        return cls(keys[order], codes[order].astype(np.int32))

    def save(self, path: Path) -> Path:
        """Write the two arrays, uncompressed so loading is a plain read."""
        with path.open("wb") as handle:
            np.savez(handle, keys=self._keys, codes=self._codes)
        return path

    @classmethod
    def load(cls, path: Path) -> CodeIndex:
        """Read an index written by `save`."""
        with np.load(path, allow_pickle=False) as arrays:
            return cls(arrays["keys"], arrays["codes"])

    def _find(self, key: object) -> int | None:
        if not isinstance(key, str):
            return None
        try:
            encoded = key.encode("utf-8")
        except UnicodeEncodeError:
            return None
        # A longer query would be truncated to the array's width and could match a stored prefix.
        if len(encoded) > self._keys.dtype.itemsize or encoded.endswith(b"\x00"):
            return None
        position = int(np.searchsorted(self._keys, encoded))
        if position < self._keys.size and self._keys[position] == encoded:
            return int(self._codes[position])
        return None

    def __getitem__(self, key: str) -> int:
        """Return the code for `key`, raising KeyError when it is unknown."""
        code = self._find(key)
        if code is None:
            raise KeyError(key)
        return code

    def __contains__(self, key: object) -> bool:
        """Return True when `key` has a code."""
        return self._find(key) is not None

    def __iter__(self) -> Iterator[str]:
        """Yield every id, in code order."""
        for position in self._by_code:
            yield bytes(self._keys[position]).decode("utf-8")

    def __len__(self) -> int:
        """Return the number of ids."""
        return int(self._keys.size)

    def agrees_with(self, mapping: Mapping[str, int]) -> bool:
        """Return True when every id in `mapping` maps to the same code here, and nothing more."""
        if len(mapping) != len(self):
            return False
        if not mapping:
            return True
        encoded = [key.encode("utf-8") for key in mapping]
        width = self._keys.dtype.itemsize
        if any(len(key) > width or key.endswith(b"\x00") for key in encoded):
            return False
        queries = np.array(encoded, dtype=self._keys.dtype)
        expected = np.fromiter(mapping.values(), dtype=np.int64, count=len(mapping))
        positions = np.minimum(np.searchsorted(self._keys, queries), self._keys.size - 1)
        found = self._keys[positions] == queries
        return bool(found.all()) and bool(
            np.array_equal(self._codes[positions].astype(np.int64), expected)
        )


@dataclass(slots=True)
class Encoding:
    """Bidirectional maps between external ids and dense integer codes.

    `n_users` and `n_items` count only train-side entities. Anything outside is "unknown", which is
    a real category with its own evaluation segment, not an error.
    """

    #: A dict when built or read for training; a `CodeIndex` when loaded for serving.
    user_to_code: Mapping[str, int]
    item_to_code: dict[str, int]
    boundaries: dict[str, int]
    seed: int
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def n_users(self) -> int:
        """Return the number of train-side users."""
        return len(self.user_to_code)

    @property
    def n_items(self) -> int:
        """Return the number of train-side items."""
        return len(self.item_to_code)

    @property
    def user_codes(self) -> dict[int, str]:
        """Reverse map, built once on first use: scoring does this in the inner loop."""
        cached = self.extras.get("_user_codes")
        if cached is None:
            cached = {code: user for user, code in self.user_to_code.items()}
            self.extras["_user_codes"] = cached
        return cached  # type: ignore[return-value]

    @property
    def item_codes(self) -> dict[int, str]:
        """Return the reverse user map, built once on first use."""
        cached = self.extras.get("_item_codes")
        if cached is None:
            cached = {code: item for item, code in self.item_to_code.items()}
            self.extras["_item_codes"] = cached
        return cached  # type: ignore[return-value]

    def as_dict(self) -> dict[str, object]:
        """Return the code space as a JSON-serialisable mapping."""
        return {
            "schema_version": CODEC_SCHEMA_VERSION,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "boundaries": self.boundaries,
            "seed": self.seed,
            **{k: v for k, v in self.extras.items() if not k.startswith("_")},
        }


def build_encoding(train_parquet: Path, boundaries: dict[str, int], seed: int) -> Encoding:
    """Assign codes to every user and item with at least one train-window interaction.

    Codes are ordered by descending interaction count and then by id, so the mapping is stable and
    frequent entities get small codes, which keeps sparse operations cache-friendly.
    """
    connection = duckdb.connect(database=":memory:")
    rows = connection.execute(
        """
        SELECT user_id, count(*) AS interactions
        FROM read_parquet(?)
        GROUP BY user_id
        ORDER BY interactions DESC, user_id
        """,
        [str(train_parquet)],
    ).fetchall()
    item_rows = connection.execute(
        """
        SELECT parent_asin, count(*) AS interactions
        FROM read_parquet(?)
        GROUP BY parent_asin
        ORDER BY interactions DESC, parent_asin
        """,
        [str(train_parquet)],
    ).fetchall()
    connection.close()

    users = {row[0]: index for index, row in enumerate(rows)}
    items = {row[0]: index for index, row in enumerate(item_rows)}
    encoding = Encoding(
        user_to_code=users,
        item_to_code=items,
        boundaries=boundaries,
        seed=seed,
        extras={"interactions": sum(int(row[1]) for row in rows)},
    )
    _logger.info(
        "encoding.built",
        extra={
            "users": encoding.n_users,
            "items": encoding.n_items,
            "interactions": sum(int(r[1]) for r in rows),
        },
    )
    return encoding


def write_encoding(encoding: Encoding, out_dir: Path) -> Path:
    """Persist the code space next to the model artefacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / CODEC_PATH
    payload = encoding.as_dict()
    payload["user_to_code"] = encoding.user_to_code
    payload["item_to_code"] = encoding.item_to_code
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def load_encoding(artifact_root: Path, corpus: str) -> Encoding:
    """Read a previously written code space. Serving must never rebuild it."""
    path = artifact_root / corpus / CODEC_PATH
    if not path.exists():
        msg = f"no encoding at {path}; run scripts/train_baselines.py first"
        raise FileNotFoundError(msg)
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != CODEC_SCHEMA_VERSION:
        msg = f"encoding schema {payload.get('schema_version')} is not supported"
        raise ValueError(msg)
    return Encoding(
        user_to_code=payload["user_to_code"],
        item_to_code=payload["item_to_code"],
        boundaries=payload["boundaries"],
        seed=payload["seed"],
        extras={
            k: v
            for k, v in payload.items()
            if k not in {"user_to_code", "item_to_code", "schema_version"}
        },
    )


def write_serving_encoding(encoding: Encoding, out_dir: Path) -> tuple[Path, Path]:
    """Write what serving loads: the code space without its user map, and the user map as arrays.

    Refuses to write unless the arrays give every user the same code the encoding does, so the
    compact form is checked against the original on the real data every time it is built.
    """
    index = CodeIndex.from_mapping(encoding.user_to_code)
    if not index.agrees_with(encoding.user_to_code):
        msg = "the user index disagrees with the encoding it was built from"
        raise ValueError(msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = encoding.as_dict()
    payload["item_to_code"] = encoding.item_to_code
    codec = out_dir / SERVING_CODEC_PATH
    codec.write_text(json.dumps(payload, sort_keys=True))
    return codec, index.save(out_dir / USER_INDEX_PATH)


def load_serving_encoding(artifact_root: Path, corpus: str) -> Encoding:
    """Read the serving form of the code space, with users as a `CodeIndex`."""
    root = artifact_root / corpus
    codec, users = root / SERVING_CODEC_PATH, root / USER_INDEX_PATH
    if not (codec.exists() and users.exists()):
        msg = f"no serving encoding in {root}; run scripts/build_serving_manifest.py first"
        raise FileNotFoundError(msg)
    payload = json.loads(codec.read_text())
    if payload.get("schema_version") != CODEC_SCHEMA_VERSION:
        msg = f"encoding schema {payload.get('schema_version')} is not supported"
        raise ValueError(msg)
    index = CodeIndex.load(users)
    if len(index) != payload["n_users"]:
        msg = f"{USER_INDEX_PATH} holds {len(index)} users, the encoding {payload['n_users']}"
        raise ValueError(msg)
    return Encoding(
        user_to_code=index,
        item_to_code=payload["item_to_code"],
        boundaries=payload["boundaries"],
        seed=payload["seed"],
        extras={
            k: v
            for k, v in payload.items()
            if k not in {"item_to_code", "schema_version", "n_users", "n_items"}
        },
    )


@dataclass(frozen=True, slots=True)
class Cohort:
    """A frozen set of evaluation users and their single held-out target.

    One target per user (their most recent test-window interaction) is what makes Recall, HR, NDCG
    and MRR comparable across sources: every source must find the same item.
    """

    name: str
    user_codes: list[int]
    target_codes: list[int]
    seed: int
    population_size: int
    filter_sql: str

    def __len__(self) -> int:
        """Return the number of users in the cohort."""
        return len(self.user_codes)

    def as_dict(self) -> dict[str, object]:
        """Return the cohort definition as a JSON-serialisable mapping."""
        return {
            "name": self.name,
            "size": len(self.user_codes),
            "population_size": self.population_size,
            "seed": self.seed,
            "filter": self.filter_sql,
        }


#: Cohort definitions. `filter_sql` runs against the partitioned Parquet with the relations
#: `train`, `valid` and `test` already bound.
COHORTS: dict[str, str] = {
    # Every user with a test-window interaction, sampled. This is the honest headline cohort: 57% of
    # it has no train history at all, and collaborative sources cannot serve those users.
    "all_test_users": """
        WITH last AS (
            SELECT user_id, arg_max(parent_asin, timestamp) AS target
            FROM test GROUP BY user_id
        ),
        known AS (SELECT user_id, parent_asin FROM train)
        SELECT last.user_id, last.target FROM last JOIN known USING (user_id)
        """,
    # Users a collaborative model can actually serve: at least two train interactions, and a target
    # that existed at training time. Paired comparisons between models are meaningful here.
    "warm_users": """
        WITH depths AS (
            SELECT user_id, count(*) AS n FROM train GROUP BY user_id
        ),
        eligible AS (SELECT user_id FROM depths WHERE n >= 2),
        last AS (
            SELECT user_id, arg_max(parent_asin, timestamp) AS target
            FROM test GROUP BY user_id
        ),
        known_items AS (SELECT DISTINCT parent_asin FROM train)
        SELECT last.user_id, last.target
        FROM last
        JOIN eligible USING (user_id)
        JOIN known_items ON known_items.parent_asin = last.target
        """,
    # `warm_users` with the target taken from the validation window instead of the test window.
    # Choices made on this cohort leave the test split untouched for a single confirmation run.
    "warm_users_valid": """
        WITH depths AS (
            SELECT user_id, count(*) AS n FROM train GROUP BY user_id
        ),
        eligible AS (SELECT user_id FROM depths WHERE n >= 2),
        last AS (
            SELECT user_id, arg_max(parent_asin, timestamp) AS target
            FROM valid GROUP BY user_id
        ),
        known_items AS (SELECT DISTINCT parent_asin FROM train)
        SELECT last.user_id, last.target
        FROM last
        JOIN eligible USING (user_id)
        JOIN known_items ON known_items.parent_asin = last.target
        """,
}


def build_cohort(
    connection: duckdb.DuckDBPyConnection,
    name: str,
    encoding: Encoding,
    processed_dir: Path,
    *,
    size: int,
    seed: int,
) -> Cohort:
    """Sample a cohort deterministically: same seed, same users, forever."""
    train = (processed_dir / "interactions" / "train.parquet").as_posix()
    valid = (processed_dir / "interactions" / "valid.parquet").as_posix()
    test = (processed_dir / "interactions" / "test.parquet").as_posix()
    for relation, path in (("train", train), ("valid", valid), ("test", test)):
        connection.execute(
            f"CREATE OR REPLACE VIEW {relation} AS SELECT * FROM read_parquet('{path}')"
        )

    if name not in COHORTS:
        msg = f"unknown cohort {name!r}; known cohorts: {', '.join(sorted(COHORTS))}"
        raise KeyError(msg)

    population = connection.execute(f"SELECT count(*) FROM ({COHORTS[name]})").fetchone()
    assert population is not None  # noqa: S101
    rows = connection.execute(
        f"""
        SELECT user_id, target FROM ({COHORTS[name]})
        ORDER BY hash(user_id || 'cohort' || {seed})
        LIMIT {size}
        """
    ).fetchall()

    # Users and targets must be paired: filtering the two lists separately would silently drop
    # different rows and score one user's target against another user's history.
    pairs = [
        (encoding.user_to_code[user_id], encoding.item_to_code[target_id])
        for user_id, target_id in rows
        if user_id in encoding.user_to_code and target_id in encoding.item_to_code
    ]
    users = [pair[0] for pair in pairs]
    targets = [pair[1] for pair in pairs]
    cohort = Cohort(
        name=name,
        user_codes=users,
        target_codes=targets,
        seed=seed,
        population_size=int(population[0]),
        filter_sql=" ".join(COHORTS[name].split()),
    )
    _logger.info(
        "cohort.built",
        extra={
            "cohort": name,
            "size": len(cohort),
            "population": cohort.population_size,
            "encoded": len(users),
        },
    )
    return cohort


def default_settings_with(root: Path) -> Settings:
    """Return settings rooted entirely at `root`."""
    return Settings(data_root=root, artifact_root=root, results_root=root)


DEFAULT_SETTINGS = default_settings
