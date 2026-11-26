# PointMaze R12 latent-radius eight-episode corner extension

This is an isolated consumed-development extension, not confirmation or
factorial evidence. It runs only latent-radius observation memory with the
nearest-recorded passage adapter on the historically exposed `r01_c06` and
`r06_c06` corner goals.

Each goal receives all eight distinct starts in the locked nine-goal spatial
development panel. Every episode retains the benchmark horizon of 20 commands
(2,000 primitive steps). Task memory persists across the eight episodes exactly
as it did across the three episodes of the R12 factorial.

The purpose is to distinguish delayed corner acquisition from a stable failure
to value the corners. The evaluator therefore reports the first contact episode
and the minimum primitive-state distance to the hidden goal in every episode.
It also records, post hoc, the latent posterior probability at the recorded
support candidate nearest the evaluator goal, its probability rank, and total
posterior mass inside the evaluator contact region. The goal coordinate and
contact radius used for these diagnostics are never passed to the planner,
executor, posterior update, passage adapter, or candidate scorer.

All models, roadmap, executor, candidate support, radius prior, posterior,
search, and passage rules are unchanged from the corresponding
`latent_radius__nearest_recorded` R12 factorial arm. Results are written to a
separate directory and must not be pooled with the four-arm factorial.
