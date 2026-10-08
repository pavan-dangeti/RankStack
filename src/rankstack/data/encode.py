"""Integer encodings and evaluation cohorts.

Model code wants dense integer ids; the pipeline stores strings. This module is the only place the
two meet, and it draws the code space from the **train partition only**, so a user or item first
seen in the test window has no code and no collaborative model can serve it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 - used at runtime by the writers

import duckdb

from rankstack.config import Settings
from rankstack.config import settings as default_settings
from rankstack.logging import get_logger

_logger = get_logger(__name__)

CODEC_PATH = "encoding.json"
CODEC_SCHEMA_VERSION = 1


@dataclass(slots=True)
class Encoding:
    """Bidirectional maps between external ids and dense integer codes.

    `n_users` and `n_items` count only train-side entities. Anything outside is "unknown", which is
    a real category with its own evaluation segment, not an error.
    """

    user_to_code: dict[str, int]
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
