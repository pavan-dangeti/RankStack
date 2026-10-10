# RankStack

A production-style recommendation system: given a user, it returns a ranked list of products they
are likely to want next, with a short explanation for each item and the model version that produced
it. Retrieval narrows a ~1.6M-item catalogue to a few hundred candidates, a learned ranker orders
them, and business rules enforce diversity and freshness. Evaluation is time-respecting: every
split, feature and metric is defined against a global timeline, and leakage is tested automatically.

**Status: Phase 5 — serving.** The service answers, and every answer is explained. Two of the three
learned components lost their measurements and are documented as such rather than quietly dropped:
the two-tower retrieval model (HR@20 0.00025) and the LambdaMART re-ranker (-0.0173 against its own
stage 1). What the service actually ships is the retriever that won, fused with a personalised
source, over a degradation ladder where every failure mode returns a working list instead of an
error.

- [Design document](docs/design.md) — requirements, SLOs, architecture, alternatives, risks, threat model
- [Architecture decisions](docs/adr/) — the decisions worth arguing about, and what was rejected
- [Data contracts](docs/schemas.md) — the Parquet schemas every later phase reads
- [Serving runbook](docs/runbook.md) — operating the API, every degradation and its alert
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

### Stage 2: re-ranking

Stage 2 takes a candidate list from `popularity_recent` and item-to-item co-occurrence, scores it
with LambdaMART over 18 point-in-time features, then applies the seen filter, a freshness bonus, a
category cap and MMR. It is scored by the same harness, on the same frozen cohorts, as every
retriever — a stage-2 number measured any other way would not be comparable.

| Cohort | Candidate ceiling | Stage 1 HR@20 | Stage 2 HR@20 | Paired difference (CI) |
|---|---|---|---|---|
| warm users | 0.0818 | 0.02395 | 0.00665 | -0.01730 (-0.01960, -0.01505) |
| all test users | 0.0888 | 0.016055 | 0.005179 | -0.01088 (-0.01329, -0.00820) |

**The candidate ceiling is the important number.** Stage 1 puts the held-out item somewhere in a
~300-item list for 8.2% of warm users, and stage 2 cannot rank what stage 1 never returned. Stage 1
already converts 2.4% of that 8.2%. There was real headroom, and stage 2 gave it all back.

> **These numbers predate a bug fix and are being re-measured.** `load_item_features` resolved item
> ids through `CodeLookup`, which returned an id's position in sorted order rather than its encoding
> code. Every per-item feature was therefore attached to the wrong item. The stage-2 result below was
> produced on scrambled features and should be read as provisional until the re-run lands. The
> *direction* of the result is not in doubt — stage 2 lost by a wide margin — but the exact figures
> are not the ones this code now produces.

### Stage 2 does not work either, and the diagnosis is specific

Re-ranking hurts, measurably: -0.0173 HR@20 on the warm cohort, bootstrap probability 0.000 that
stage 2 beats stage 1. Probing 2,000 users directly, restricted to those whose target is in the
candidate list, the ranker moved the target **up 57 times and down 106 times**, and the median
position of a correct item fell from 134 to 181. The harness is not penalising stage 2; the ranker
genuinely demotes the right item.

Three measurements explain it, each checked rather than assumed:

1. **The training label is much easier than the evaluation label.** At training time the target is
   the user's most recent *train* item and candidate recall is 0.579. At evaluation it is the most
   recent *test* item and the ceiling is 0.082, a seven-fold drop. The ranker is fitted to promote
   items that a train-window retriever already finds easy.
2. **The dominant feature is anti-correlated with the target under a temporal holdout.**
   `category_affinity` takes 39% of the split gain: how much of the user's history sits in the
   candidate's category. A user's later purchase is more often a new category than a repeat, so the
   strongest feature the model found systematically demotes the item it is judged on.
3. **A truncation-window hypothesis was tested and rejected.** `lambdarank_truncation_level`
   defaults to 30 while the positive sits around position 150, so widening it to 300 should have
   helped. It did not: 0.00665 against 0.00675. The setting stays configurable because 30 is a real
   constraint on this list length, but it is not the explanation, and the commit message that claimed
   otherwise was wrong.

The likely cause is a train/serve objective mismatch: LambdaMART is fitted to reorder a list that
contains its positive 58% of the time, then asked to find that positive in a list where it appears
8% of the time, six months later. What to test next, none of it done here: a binary or
`rank_xendcg` objective over sampled negatives, or training the ranker on a candidate construction
matched to evaluation-time retrieval difficulty.

## Serving

`GET /recommend?user_id=<id>&k=<1..50>` returns a ranked list with a template-based explanation per
item, the artefact checksums that produced it, and the request id. It always answers **HTTP 200 with
a non-empty list**: unknown users, a corrupt index and a missing feature store each degrade to a
weaker but working model, and every degradation is a counted metric rather than a silent fallback.

```bash
uv run python scripts/build_serving_manifest.py          # hash artefacts; the loader verifies
uv run python -m uvicorn rankstack.serving.app:app        # http://localhost:8000
uv run pytest tests/unit/test_serving.py -q               # every degradation row, injected
```

Artefacts are hashed at load time and a mismatch refuses to start the process. A serving process
that cannot vouch for its model does not answer. Operations, alerts and the full degradation table
are in [`docs/runbook.md`](docs/runbook.md).

### What the service serves, and what that costs

Serving the best-scoring retriever alone turned out to serve **the same list to everyone**.
`popularity_recent` is a global ranking: across 2,000 warm users it returned **3 distinct top
items**. A recommender that does that is not personalising.

So the request path serves reciprocal-rank fusion of `popularity_recent` with item-item
co-occurrence (`scripts/measure_fusion.py`, `results/fusion_electronics_0core.json`):

| Served source | HR@20 | HR@10 | Distinct top-1 over 2,000 users | Figure |
|---|---|---|---|---|
| `popularity_recent` | 0.02000 | 0.01450 | 3 | results/fusion_electronics_0core.json#/sources/popularity_recent/ranking/hr@20 |
| `cooccurrence` | 0.00150 | 0.00050 | 1,911 | results/fusion_electronics_0core.json#/sources/cooccurrence/ranking/hr@20 |
| **fused (shipped)** | **0.01500** | **0.00900** | **18** | results/fusion_electronics_0core.json#/sources/fused/ranking/hr@20 |

Fusion gives up 25% of HR@20 (paired difference -0.005, 95% CI [-0.0095, -0.0010]) and buys six
times the personalisation. The trade is taken deliberately and published rather than discovered by
a user.

| Measure | Distinct top-1 items over 2,000 warm users | Figure |
|---|---|---|
| `popularity_recent` alone, if it were the only source | 3 | results/fusion_electronics_0core.json#/personalisation/distinct_top_1_global |
| **fused (shipped)** | **18** | results/fusion_electronics_0core.json#/personalisation/distinct_top_1_fused |

### Measured latency

400 requests at concurrency 8, k=20, single process (`scripts/load_test.py`,
`results/load_test_electronics_0core.json`):

| Path | p50 | p95 | p99 | Target | Figure |
|---|---|---|---|---|---|
| Warm user | 25.1 ms | 33.0 ms | 59.7 ms | 60 / 150 / 300 ms | results/load_test_electronics_0core.json#/latency/0/p50_ms |
| Cold start | 24.1 ms | 29.7 ms | 30.8 ms | p95 ≤ 250 ms | results/load_test_electronics_0core.json#/latency/1/p50_ms |
| Degraded (index raising) | 8.7 ms | 13.9 ms | 15.8 ms | p95 ≤ 20 ms | results/load_test_electronics_0core.json#/latency/2/p50_ms |

All five SLOs met, 310 req/s in one process. Every row above is re-read from the result file
by CI, so this table cannot drift from what the code produces.

| Quantity | Value | Source |
|---|---|---|
| Catalogue size served | 1,334,914 | results/load_test_electronics_0core.json#/catalogue_items |
| Artefact load time | 9.37 s | results/load_test_electronics_0core.json#/artifact_load_seconds |
| Throughput, one process | 310.4 req/s | results/load_test_electronics_0core.json#/throughput_requests_per_second_single_process |
