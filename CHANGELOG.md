# Changelog

Each phase was merged as one pull request and is marked by an annotated tag at the commit where it
landed. Pull request numbers refer to the original repository, which is now archived; the history
here is the same. Every figure below is quoted from the README, which names the committed result
file it comes from.

## phase-3-ann (PR #9)

FAISS benchmark of approximate indexes for stage-1 retrieval. No approximate index reaches the
0.95 recall@200 target. Exact `IndexFlatIP` is the recommended index for ALS, at 3.8184 ms p99 with
one thread (`results/ann_index_electronics_0core_als.json#/curves/flat/0/p99_ms`). The two-tower
vectors are the failure case: the best approximate index reaches 0.902425 recall@200
(`results/ann_index_electronics_0core_two_tower.json#/curves/hnsw/53/recall_at_top_k`).

## phase-4-ablation (PR #8)

Ablation of the post-ranking rule layer. Warm-user HR@20 is 0.00915 with and without the rules
(`results/two_stage_model_only_electronics_0core.json#/cohorts/warm_users/stage_2/ranking/hr@20`),
so the rule layer has no measurable effect at its shipped weights.

## phase-4-remeasure (PR #7)

Stage 2 re-measured after fixing the CodeLookup bug. Warm-user HR@20 is 0.00915
(`results/two_stage_electronics_0core.json#/cohorts/warm_users/stage_2/ranking/hr@20`), against
0.02395 for the best stage-1 source; re-ranking remains negative.

## phase-5-serving (PR #6)

FastAPI service that always answers: checksum-verified artefacts, a degradation ladder down to
popularity, load-test results and an operations runbook.

## phase-4-reranking (PR #5)

Stage-2 LambdaMART re-ranker with point-in-time features, candidate generation and post-ranking
rules. Re-ranking measured worse than stage 1.

## phase-3-retrieval (PR #4)

Two-tower retriever with sampled softmax and popularity correction, measured and diagnosed. It
reaches warm-user HR@20 0.00025
(`results/evaluation_electronics_0core.json#/cohorts/warm_users/sources/two_tower/ranking/hr@20`),
below every baseline.

## phase-2-baselines (PR #3)

Popularity, co-occurrence and ALS baselines with the evaluation harness. Recency-weighted
popularity is the strongest source, at warm-user HR@20 0.02395
(`results/evaluation_electronics_0core.json#/cohorts/warm_users/sources/popularity_recent/ranking/hr@20`).

## phase-1-data (PR #2)

Verified download, validated time-split pipeline, point-in-time features and a leakage test suite
with a negative control.

## phase-0-design (PR #1)

Design document, evaluation protocol, threat model, zero-cost hosting plan and six
architecture decision records.
