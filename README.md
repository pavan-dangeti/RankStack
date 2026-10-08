# RankStack

A production-style recommendation system: given a user, it returns a ranked list of products they
are likely to want next, with a short explanation for each item and the model version that produced
it. Retrieval narrows a ~1.6M-item catalogue to a few hundred candidates, a learned ranker orders
them, and business rules enforce diversity and freshness. Evaluation is time-respecting: every
split, feature and metric is defined against a global timeline, and leakage is tested automatically.

**Status: design phase.** The system design, the evaluation protocol, the threat model and the
dataset decision are written up and under review. Results, the demo and the serving service arrive
in later phases.

- [Design document](docs/design.md) — requirements, SLOs, architecture, alternatives, risks, threat model
- [Architecture decisions](docs/adr/) — the decisions worth arguing about, and what was rejected
- Data: Amazon Reviews 2023 (McAuley Lab), Electronics. Terms analysis and mitigations in
  `docs/design.md` §4.3.