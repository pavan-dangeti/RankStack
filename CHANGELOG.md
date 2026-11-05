# Changelog

Every figure below is followed by the committed result file it comes from, and CI checks each one
against that file. Entries from v1.0.0 on refer to pull requests in this repository. The phase
entries before it refer to pull requests in the original repository, which is now archived; the
history is the same, and each phase is marked by an annotated tag at the commit where it landed.

## Unreleased

- Demo fixture and runbook no longer contain reviewer ids; earlier commits, the v1.0.0 tag and PR
  refs #1 to #9 still do (9 pseudonymous ids from the public dataset); fixed forward, no history
  rewrite (rationale in design §4.3).
- The serving process's memory is measured, not assumed (`scripts/measure_memory.py`); the
  runbook and Dockerfile had understated it several times over. The user map is now a sorted id
  array searched by bisection instead of a dict of fifteen million ids, which brings the process
  inside the serving image to
  2,152.0 (`results/memory_electronics_0core_linux-aarch64-container.json#/rss_after_requests_mib`)
  MiB, with a peak of
  2,372.9 (`results/memory_electronics_0core_linux-aarch64-container.json#/peak_rss_mib`) MiB.
  Latency was measured three times with the dict and three times with the arrays, interleaved on
  one machine; medians of warm-user latency, dict then arrays: p50
  23.7 ms (`results/load_test_median_electronics_0core.json#/dict/latency/warm/p50_ms`) and
  23.5 ms (`results/load_test_median_electronics_0core.json#/arrays/latency/warm/p50_ms`), p95
  30.3 ms (`results/load_test_median_electronics_0core.json#/dict/latency/warm/p95_ms`) and
  29.0 ms (`results/load_test_median_electronics_0core.json#/arrays/latency/warm/p95_ms`), p99
  69.2 ms (`results/load_test_median_electronics_0core.json#/dict/latency/warm/p99_ms`) and
  35.9 ms (`results/load_test_median_electronics_0core.json#/arrays/latency/warm/p99_ms`); throughput
  330.0 req/s (`results/load_test_median_electronics_0core.json#/dict/throughput_requests_per_second_single_process`) and
  328.1 req/s (`results/load_test_median_electronics_0core.json#/arrays/throughput_requests_per_second_single_process`). All five SLOs were
  met in every run.
- The runbook's `docker run` mounts `data/` whole: mounting `data/processed` at `/data` left the
  catalogue where the server does not look for it.

## v1.0.0

The first release: a recommender that answers, explains every answer, and shows it working at
`/demo`.

**Served.** For a known user, recency-weighted popularity fused with item-item co-occurrence at
weight 0.75 (`results/fusion_weight_sweep_electronics_0core.json#/selected_weight`), chosen on the
validation split by a rule committed before any weight was evaluated (ADR-0007) and confirmed once on
the test split. There, against popularity alone, the paired HR@20 difference is
0.0005 (`results/fusion_weight_test_electronics_0core.json#/result/paired_hr@20_vs_popularity_recent/difference`),
95% CI -0.003 (`results/fusion_weight_test_electronics_0core.json#/result/paired_hr@20_vs_popularity_recent/ci_low`)
to 0.004 (`results/fusion_weight_test_electronics_0core.json#/result/paired_hr@20_vs_popularity_recent/ci_high`).
An unknown user gets popularity alone. Warm-user latency is
23.9 ms (`results/load_test_electronics_0core.json#/latency/0/p50_ms`) at p50 and
34.6 ms (`results/load_test_electronics_0core.json#/latency/0/p99_ms`) at p99 in one process.

**Changes since the last phase (pull requests #1 to #4 and #6):**

- A cold-start privacy bug is fixed (#1): unknown users were scored as the first encoded user, a real
  customer, so every anonymous visitor saw items from that customer's history. They now get the
  global ranking alone, and tests check the list's contents.
- Fusion no longer gives rank credit to items a source scores zero (#2), which had quietly promoted
  the most popular items through the personalised source.
- The served co-occurrence weight is a setting, chosen by the pre-registered rule in ADR-0007 (#3).
  Weights above 61/80 are deferred to a fresh hold-out window.
- A demo page at `/demo` (#4): eight curated shoppers and an anonymous visitor, the served blend
  against popularity alone, and the real response's request id, timings and fallback.
- A README front page (#6) with the demo GIF, a key-results table, the architecture, how to run it,
  the findings and the limitations.
- The documentation drift check now reads figures in prose, in table cells, under table column
  paths, in the ADRs, in this changelog and on the demo page.

## phase-3-ann (PR #9)

FAISS benchmark of approximate indexes for stage-1 retrieval. No approximate index reaches the
0.95 recall@200 target. Exact `IndexFlatIP` is the recommended index for ALS, at
3.8184 ms (`results/ann_index_electronics_0core_als.json#/curves/flat/0/p99_ms`) p99 with one
thread. The two-tower vectors are the failure case: the best approximate index reaches
0.902425 (`results/ann_index_electronics_0core_two_tower.json#/curves/hnsw/53/recall_at_top_k`)
recall@200.

## phase-4-ablation (PR #8)

Ablation of the post-ranking rule layer. Warm-user HR@20 is
0.00915 (`results/two_stage_model_only_electronics_0core.json#/cohorts/warm_users/stage_2/ranking/hr@20`)
without the rules and
0.00915 (`results/two_stage_electronics_0core.json#/cohorts/warm_users/stage_2/ranking/hr@20`)
with them, so the rule layer has no measurable effect at its shipped weights.

## phase-4-remeasure (PR #7)

Stage 2 re-measured after fixing the CodeLookup bug. Warm-user HR@20 is
0.00915 (`results/two_stage_electronics_0core.json#/cohorts/warm_users/stage_2/ranking/hr@20`),
against
0.02395 (`results/evaluation_electronics_0core.json#/cohorts/warm_users/sources/popularity_recent/ranking/hr@20`)
for the best stage-1 source; re-ranking remains negative.

## phase-5-serving (PR #6)

FastAPI service that always answers: checksum-verified artefacts, a degradation ladder down to
popularity, load-test results and an operations runbook.

## phase-4-reranking (PR #5)

Stage-2 LambdaMART re-ranker with point-in-time features, candidate generation and post-ranking
rules. Re-ranking measured worse than stage 1.

## phase-3-retrieval (PR #4)

Two-tower retriever with sampled softmax and popularity correction, measured and diagnosed. It
reaches warm-user HR@20
0.00025 (`results/evaluation_electronics_0core.json#/cohorts/warm_users/sources/two_tower/ranking/hr@20`),
below every baseline.

## phase-2-baselines (PR #3)

Popularity, co-occurrence and ALS baselines with the evaluation harness. Recency-weighted
popularity is the strongest source, at warm-user HR@20
0.02395 (`results/evaluation_electronics_0core.json#/cohorts/warm_users/sources/popularity_recent/ranking/hr@20`).

## phase-1-data (PR #2)

Verified download, validated time-split pipeline, point-in-time features and a leakage test suite
with a negative control.

## phase-0-design (PR #1)

Design document, evaluation protocol, threat model, zero-cost hosting plan and six
architecture decision records.
