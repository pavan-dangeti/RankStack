# ADR-0007 — Choosing the co-occurrence weight in the served fusion by a pre-registered rule

- Status: Accepted — rule pre-registered in commit 07359c8, results added after the sweep
- Date: 2026-11-05

## Context

For known users the service serves reciprocal-rank fusion of `popularity_recent` and item-item
co-occurrence. At equal weight the blend gives up hit rate against `popularity_recent` alone, and the
loss is measurable: the paired HR@20 confidence interval on the test-split warm cohort excludes zero.
What the blend buys is a list that differs between users. Whether any blend is worth serving, and at
what weight, has so far been argued rather than decided by a rule.

This document fixes the rule **before** any weight is evaluated. It is committed on its own, ahead
of the code that runs the sweep, so the commit history shows the rule was not fitted to the results.

## Rule

**Candidates.** The fused score of an item is

    1 / (60 + rank in popularity_recent) + w / (60 + rank in co-occurrence)

over each source's top 500 non-zero scores, for w in {0, 0.1, 0.25, 0.5, 0.75, 1.0}. w = 0 is
`popularity_recent` alone. RRF constant 60 and depth 500 are unchanged from the current service.

**Selection data: validation split only.** Cohort `warm_users_valid`: users with at least two
train-window interactions whose last validation-window interaction is an item that existed at
training time; that item is the target. 2,000 users, sampled with the project seed (20261008) by the
same deterministic rule as every other cohort. Each user's history is their train window, as in every
other evaluation. Every candidate is scored on the same users. The models are those already trained
on the train window, so the validation targets are unseen by every candidate.

**Reported for each w:**

- HR@20 with a 95% bootstrap CI (1,000 resamples of users, percentile interval, project seed);
- the paired HR@20 difference against `popularity_recent` with its 95% paired-bootstrap CI
  (same resamples, same seed);
- distinct #1 items: the number of different items ranked first across the 2,000 users;
- catalogue coverage@20: the number of distinct items in the union of every user's top 20, also
  reported as a fraction of the 1,334,914-item catalogue.

**Selection.** A weight qualifies when its paired HR@20 difference CI includes zero
(`ci_low <= 0 <= ci_high`). Among qualifying weights, choose the one with the highest coverage@20;
a tie goes to the smaller weight. w = 0 differs from itself by exactly zero, so it always qualifies;
if no w > 0 qualifies, the rule therefore selects w = 0 and the service serves `popularity_recent`
alone. Nothing measured after the sweep can change the selection.

**What "qualifies" means.** A CI that includes zero says the loss is not detectable with 2,000
users. It does not say there is no loss. Five weights are compared against one baseline with no
multiplicity correction, which makes a qualifying weight easier, not harder, to find; the rule
accepts that because a false "no detectable loss" costs at most the gap a 2,000-user CI cannot see.

**Confirmation: test split, once.** The selected weight alone is then evaluated once on the existing
test-split `warm_users` cohort (2,000 users, same seed), together with `popularity_recent` as the
paired baseline, and the same four quantities are reported. The test result is reported whatever it
shows and does not feed back into the choice. The script refuses to overwrite an existing test
result.

**Serving.** The selected weight becomes the serving default as a configuration value. If it is 0,
the service serves `popularity_recent` alone to every user, known or not.

## Consequences

The rule above is unchanged since it was committed. Results, added afterwards:

**Selection.** The sweep (`uv run python scripts/select_fusion_weight.py sweep`) selected
w = 0.75 (`results/fusion_weight_sweep_electronics_0core.json#/selected_weight`). Every weight from 0
to 0.75 qualified; w = 1 did not, with a paired HR@20 difference CI from
-0.012 (`results/fusion_weight_sweep_electronics_0core.json#/by_weight/1.0/paired_hr@20_vs_popularity_recent/ci_low`)
to -0.0015 (`results/fusion_weight_sweep_electronics_0core.json#/by_weight/1.0/paired_hr@20_vs_popularity_recent/ci_high`).
The full table is in the README.

**Confirmation.** On the test split the selected weight's paired HR@20 difference against
`popularity_recent` is
0.0005 (`results/fusion_weight_test_electronics_0core.json#/result/paired_hr@20_vs_popularity_recent/difference`),
CI from
-0.003 (`results/fusion_weight_test_electronics_0core.json#/result/paired_hr@20_vs_popularity_recent/ci_low`)
to 0.004 (`results/fusion_weight_test_electronics_0core.json#/result/paired_hr@20_vs_popularity_recent/ci_high`):
no detectable loss there either. It was run once.

**Serving.** `Settings.fusion_cooccurrence_weight` defaults to the selected weight.

**Finding, not applied.** Coverage@20 grows slowly up to w = 0.75, from
23 (`results/fusion_weight_sweep_electronics_0core.json#/by_weight/0.0/coverage_at_20/items`) to
133 (`results/fusion_weight_sweep_electronics_0core.json#/by_weight/0.75/coverage_at_20/items`)
items, then jumps to
13,100 (`results/fusion_weight_sweep_electronics_0core.json#/by_weight/1.0/coverage_at_20/items`)
at w = 1. The cause is the RRF arithmetic, not the data: co-occurrence's first-ranked item scores
w/61 and popularity's twentieth item scores at least 1/80, so an item ranked only by co-occurrence
can reach a top 20 only when w exceeds 61/80. Every candidate weight below 1 sits under that line, so
the grid could not test a blend that brings new items into the top 20 at an acceptable hit-rate cost.
A future rule that wants to test that should place grid points between 61/80 and 1; this one is not
re-run to do so.
