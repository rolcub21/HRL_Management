# E4 — Certified-frontier ranking ablation

E4 asks how much management quality comes from ranking after the exact
recoverability certificate has fixed the admissible action frontier. On a new
30-instance 92k panel it holds candidate generation, exact certification, the
recovery-witness liveness guard, normalized mode aggregation, deterministic
tie-breaking, and macro execution fixed. Only the candidate merit changes:

- reproducible random-safe ranking (five predeclared hash seeds in the full run);
- a frozen rank/urgency/distance operational heuristic;
- the frozen VCG1.1 operational critic `Q_op`;
- conditioned VCG at predeclared `lambda = {.05, .10, .20}`.

Random and heuristic ranking are policy-seed-free, so they are not needlessly
duplicated across neural model seeds. The seed-0 VCG configuration is only their
executor/liveness carrier. Learned arms use all three final model seeds.

## Pilot

The pilot runs seed 0 on the first five frozen instances and one of the five
random-ranking seeds. Its 30 rows are only a debugging and mechanism check:

```bash
bash experiments/conditioned_vcg/E04_safe_frontier_ranking_92k/run.sh run-pilot
```

It verifies that every ranker receives only exact-SAFE candidates and passes
through the identical selector. An unfavorable result is not a reason to tune
the predeclared design.

## Full experiment

```bash
bash experiments/conditioned_vcg/E04_safe_frontier_ranking_92k/run.sh run-all
```

The resumable full grid contains 540 rows: 150 random-safe, 30 heuristic-safe,
90 `Q_op`, and 270 conditioned rows. Generated instances, ledgers, report, and
table are written to `results/vcg-conditioned-e04-ranking-ablation-92k/`.

## Paper figure and table

After all 540 rows pass authentication, render the paper-facing two-view
figure and tightened table without running any additional episode:

```bash
bash experiments/conditioned_vcg/E04_safe_frontier_ranking_92k/run.sh figures
```

The left view retains the full ranking landscape, including the extreme
random and heuristic controls. The right view expands the learned VCG region
and displays each of the three model-seed trajectories plus their aggregate.
This represents seed variation directly rather than treating 90 crossed
model-instance rows as independent training replicates.
