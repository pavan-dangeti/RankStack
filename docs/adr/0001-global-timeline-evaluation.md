# ADR-0001 — Evaluation uses a global timeline only

- Status: Accepted
- Date: 2026-10-08

## Context

A recommender evaluated with a random train/test split, or with a per-user leave-last-out split
applied on top of a random global split, sees interactions from after the moment it is asked to
predict. The model appears to predict the future it was trained on, and every reported number is
inflated by an amount that grows with catalogue size and interaction density.

This is not hypothetical: Ji, Sun and Zhang (ACM TORS 2023) show that BPR, NeuMF, SASRec and
LightGCN all recommend items that did not exist at the prediction time on MovieLens-25M, Yelp,
Amazon-Music and Amazon-Electronics, and that the relative ranking of those baselines *changes*
depending on how much future data is included.

RankStack's evaluation must be defensible to someone who knows this literature.

## Decision

1. One split strategy only: a **global timeline** split on absolute timestamps. Train is
   `(-inf, t1)`, validation is `[t1, t2)`, test is `[t2, +inf)`, with `t1` and `t2` fixed in
   configuration and recorded in every result file.
2. `t1` and `t2` default to the publisher's published values for the corpus
   (`1628643414042`, `1658002729837`) so that results are comparable to external work, and the
   pipeline asserts that its re-derived split matches the publisher's published split sizes.
3. Leave-last-out is used **within** the test window only, to produce a single held-out target per
   user for ranking metrics. It never defines the global partition.
4. An additional earlier window is held out from training entirely as an untouched scoring set
   during development.
5. The test window is read at most twice per frozen model: once at freeze, once at the end. All
   tuning happens on the validation window.
6. No sampled metrics. Scoring is over the full candidate catalogue, because sampling 100 negatives
   from a 1.6M-item catalogue inflates Recall@20 by orders of magnitude.
7. Leakage is tested automatically (§5.5 of the design doc), including a negative control in which
   a deliberately time-shuffled pipeline must fail the suite.

## Consequences

**Good.** Every number in the project means "this user, at this moment, given only what was known
then". Model comparisons are meaningful because every model sees the same information. Results are
comparable with published work on the same corpus and split.

**Bad.** Lower absolute scores than a leaked evaluation would report, and a smaller effective test
set (only users with a test-window interaction are evaluable). Iteration is slower because the
validation window is small relative to the training window, and hyper-parameter search on it can
overfit — which is exactly why the untouched earlier window exists.

**Risk.** Small segment sizes inside the test window (new users, new items) make confidence
intervals wide. Segments with `n` below a stated threshold are reported with their `n` and no point
estimate, rather than with a noisy one.

## Alternatives considered

- **Random split** — rejected: not defensible, and every reviewer knows why.
- **Per-user leave-last-out without a global cut** — rejected: still leaks across users; it is the
  protocol this ADR exists to avoid.
- **Publisher's `last_out_w_his` files as the primary protocol** — rejected as the primary because
  it does not define a prediction moment in wall-clock time; kept as the comparability protocol
  (§5.6 of the design doc).