# RankStack

A production-style recommendation system: given a user, it returns a ranked list of products they
are likely to want next, with a short explanation for each item and the model version that produced
it. Retrieval narrows a ~1.6M-item catalogue to a few hundred candidates, a learned ranker orders
them, and business rules enforce diversity and freshness. Evaluation is time-respecting: every
split, feature and metric is defined against a global timeline, and leakage is tested automatically.

**Status: Phase 1 — data foundation.** The system design is written up
([`docs/design.md`](docs/design.md)) and the data pipeline is implemented and tested: verified
download, schema validation that fails the build, the global-timeline split, point-in-time feature
snapshots, and a leakage suite with a negative control. Retrieval, ranking and serving land in later
phases; nothing here recommends anything yet.

- [Design document](docs/design.md) — requirements, SLOs, architecture, alternatives, risks, threat model
- [Architecture decisions](docs/adr/) — the decisions worth arguing about, and what was rejected
- [Data contracts](docs/schemas.md) — the Parquet schemas every later phase reads
- Data: Amazon Reviews 2023 (McAuley Lab), Electronics. Terms analysis and mitigations in
  `docs/design.md` §4.3.

## What the pipeline does

```bash
uv sync --all-groups                                     # locked environment, Python 3.12
uv run python scripts/download_data.py                   # fetch + verify raw files (~2.3 GB)
uv run python scripts/build_dataset.py                   # validate, split, write Parquet + manifest
uv run pytest --cov=rankstack --cov-fail-under=85        # unit, property, integration, e2e
```

Raw dataset files are never committed (`data/` is gitignored, and CI fails if one is tracked). Every
build writes `data/processed/{corpus}/manifest.json` with source hashes, partition row counts,
validation outcomes, output hashes, seed, tool versions and platform, so a run can be repeated and
audited.

## Measured dataset statistics

Amazon Reviews 2023, Electronics, 0-core. Every figure below comes from
`results/dataset_electronics_0core.json`, which CI re-reads and compares against these numbers
(`scripts/check_docs_drift.py`).

| Quantity | Value | Source |
|---|---:|---|
| Interactions after validation | 43,365,424 | `results/dataset_electronics_0core.json#/totals/interactions` |
| Distinct users | 18,286,190 | `results/dataset_electronics_0core.json#/totals/users` |
| Distinct items with metadata | 1,610,012 | `results/dataset_electronics_0core.json#/totals/items_with_metadata` |
| Train rows (`timestamp < t1`) | 35,335,971 | `results/dataset_electronics_0core.json#/partitions/train/rows` |
| Validation rows (`t1 <= ts < t2`) | 3,953,788 | `results/dataset_electronics_0core.json#/partitions/valid/rows` |
| Test rows (`t2 <= ts`) | 4,075,665 | `results/dataset_electronics_0core.json#/partitions/test/rows` |
| Items present in train | 1,334,914 | `results/dataset_electronics_0core.json#/partitions/train/items` |
| Items first seen after `t1` | 274,946 | `results/dataset_electronics_0core.json#/cold_start/new_items` |
| Users first seen after `t1` | 2,845,057 | `results/dataset_electronics_0core.json#/cold_start/new_users` |
| Rows dropped by validation | 2 | `results/dataset_electronics_0core.json#/validation/rows_dropped` |
| Build wall-clock | 125 s | `results/dataset_electronics_0core.json#/elapsed_seconds` |

Reproduce with `uv run python scripts/build_dataset.py --corpus electronics_0core` on the platform
recorded in the result file's `environment` block. Download is a separate step
(`scripts/download_data.py`); both verify file sizes and record SHA-256 hashes. The build time above
is a warm run: the first run on a cold file cache, parsing the 1.31 GB metadata file, took 233 s on the
same machine.

### What validation found in the published data

Three findings, all recorded in the result file rather than smoothed over:

- **2 interactions rated `0.0`**, which is outside the documented 1–5 scale. Dropped and counted
  (`validation.checks[rating_out_of_scale]`). Two rows out of 43.4M cannot move any metric.
- **`price` is heterogeneous**: 526,449 numeric, 1,083,247 null, 316 strings — 266 of which are the
  em-dash placeholder `—` and the rest look like `from 9.95`. The reader takes the first number in a
  string price and never coerces a missing price to zero, because "unknown price" and "free" are
  different facts for a price-affinity feature.
- **128,442 items have no category path** and 106,334 have no `main_category`. Category affinity
  treats those as unknown rather than as a bucket of their own.

The pipeline also cross-checks its split against the publisher's published split sizes, which it
re-derives rather than trusting: 35,335,971 measured against 35,300,000 published for train,
3,953,788 against 4,000,000 for validation, 4,075,665 against 4,100,000 for test — all inside the 2%
tolerance that three-significant-figure rounding requires.
## Measured model results

Every baseline and the two-tower are scored by the same harness on the same frozen cohorts, with
exact full-catalogue ranking (no sampled negatives). One held-out target per user, ties broken by
ascending item code, already-seen items excluded from the candidate set. Numbers come from
`results/evaluation_electronics_0core.json`.

HR@20, left: all test users (n=11,585, includes 57% with no train history and 41% whose target is an
item unseen at training time). Right: warm users only (n=20,000, at least two train interactions and
a target that existed at training time) — the cohort where collaborative models are comparable.

| Source | HR@20 (warm) | HR@20 (all) | Served (all) | Coverage@50 (warm) | Figure |
|---|---|---|---|---|---|
| `popularity` | 0.0069 | 0.006733 | 1.0 | 0.000043 | `results/evaluation_electronics_0core.json#/cohorts/warm_users/sources/popularity/ranking/hr@20` |
| `popularity_recent` | **0.02395** | 0.016055 | 1.0 | 0.000041 | `results/evaluation_electronics_0core.json#/cohorts/warm_users/sources/popularity_recent/ranking/hr@20` |
| `cooccurrence` | 0.00295 | 0.001985 | 1.0 | 0.339083 | `results/evaluation_electronics_0core.json#/cohorts/warm_users/sources/cooccurrence/ranking/hr@20` |
| `als` | 0.00525 | 0.00587 | 0.914631 | 0.001101 | `results/evaluation_electronics_0core.json#/cohorts/warm_users/sources/als/ranking/hr@20` |
| `two_tower` | 0.00025 | 0.000604 | 1.0 | 0.492985 | `results/evaluation_electronics_0core.json#/cohorts/warm_users/sources/two_tower/ranking/hr@20` |
| `two_tower_no_logq` | 0.00005 | 0.000604 | 1.0 | 0.511973 | `results/evaluation_electronics_0core.json#/cohorts/warm_users/sources/two_tower_no_logq/ranking/hr@20` |

The Figure column carries a machine-checked pointer into the result file for the warm-cohort
HR@20, so CI fails if this table drifts. The remaining columns are supporting context read from
the same file.

Paired bootstrap over 1,000 resamples, against `popularity_recent`, warm cohort, HR@20:
`als` -0.0187 (CI -0.0208, -0.0166), `two_tower` -0.0237 (CI -0.0259, -0.0215). Every difference
excludes zero, so **nothing learned here beats recency-weighted popularity on this corpus**, which is
the result worth stating plainly rather than burying.

### The two-tower does not work yet

The two-tower underperforms every baseline by roughly an order of magnitude, and the diagnosis
(`scripts/diagnose_two_tower.py`, results in `results/two_tower_diagnosis_electronics_0core.json`)
shows why it is not simply undertrained: on a 400,000-user subsample the training loss falls from
9.82 to 1.63 while HR@20 does not move at all (0.0055 at both 1 and 6 epochs). The sampled-softmax
objective is being minimised without improving the global ranking, which points at the negative
sampling budget — 4,096 in-batch negatives against a 1.33M-item catalogue — rather than at the
architecture or the number of epochs.

What is *not* established, and would be the next thing to measure: whether a larger negative budget,
a sequence encoder in the user tower, or an item-side text tower closes the gap. Phase 3 deliberately
records the negative result and the diagnosis rather than tuning until one configuration wins.

Two defects were found by this measurement and are fixed, with regression tests:

- `gini` returned 0.889 for a uniform distribution (wrong discrete formula).
- `exact_rank` counted ties against a *filtered* score vector, misaligning values with item codes, so
  any source with masked items produced wrong ranks.
