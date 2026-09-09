# Exhaustive live recovery trees on the 5x5 environment

This additive, evaluation-only experiment executes the complete declared
finite disturbance tree, modulo the declared absorbing-terminal equivalence,
through the authoritative 5x5 environment for one committed stored-block
cohort. It uses frozen policies and performs no training, checkpoint
selection, lambda selection, or instance selection.

## Controlled roots

All six arms start from exactly the same authenticated pre-action state for an
EpisodeInstance:

- nominal, one-step, and recursive recovery filters; and
- handling lambda 0 and 0.2.

Roots come from the previously authenticated snapshot panel. The source VCG
lambda-zero trajectory physically reached each state before robustness was
computed. Admission is then closed exogenously, its stored labels and
witness-derived primitive budget are committed, and the six arms are crossed
on the same live environment clone.

The staged panels are:

- `smoke`: seed 89000, occupancy 2, epoch 14, budget 17;
- `pilot`: all eight predeclared occupancy-2 roots; and
- `full`: all 30 predeclared roots at occupancies 2, 4, 6, or 8.

The smoke and occupancy-2 pilot are intentional. A dense eight-block segment
can have millions of raw disturbance paths before state convergence, so the
full panel should not be launched until the smaller DAG sizes are inspected.

## Live branch protocol

At every reachable active boundary, the runner:

1. enumerates the exact nominally safe live frontier;
2. applies the declared recovery filter at the carried remaining budget;
3. invokes the unchanged frozen selector once;
4. enumerates every member of the declared delay/stop envelope;
5. deep-copies the unchanged parent environment for every outcome;
6. executes the certified path and disturbance literally through `env.step`,
   except declared post-terminal steps represented by the absorbing quotient;
7. verifies path, duration, physical successor, cohort, and budget; and
8. replans independently from every realized successor.

The branch stops only when the committed cohort is empty or a classified
failure occurs. A full causal live-state key deduplicates convergent suffixes;
path multiplicities are retained in the DAG recurrence.

Global terminal delivery is explicit: the environment forbids actions after
termination, so nonnominal abstract post-terminal outcomes are recorded as
aliases of the one absorbing terminal transition. They are never silently
omitted. The report separates abstract outcomes, live executions, live
equivalence classes, and terminal aliases.

## Status semantics

- `PASS`: all declared branches are accounted for and every leaf completes
  the cohort without mismatch, violation, deadlock, or unresolved cutoff.
- `FAIL`: at least one concrete modeled disturbance path is a reproducible
  counterexample.
- `UNKNOWN`: the finite solver or live DAG reached a declared resource cap.
- `INVALID`: cloning, source authentication, or model/executor correspondence
  failed.

`UNKNOWN` is neither success nor failure. The node cap is part of the
authenticated contract; changing it requires a new output directory.

Because the disturbance set has no probability measure, the report does not
average uniformly over leaves. It reports universal completion, status/path
counts, and completed-path extrema with disturbance witnesses.

## Run in stages

From the repository root:

```bash
bash experiments/vcg_exhaustive_live_recovery_tree_5x5/run.sh prepare-smoke
bash experiments/vcg_exhaustive_live_recovery_tree_5x5/run.sh run-smoke
```

If the smoke tree closes without `UNKNOWN` or `INVALID`, run the eight-root
occupancy-2 pilot:

```bash
bash experiments/vcg_exhaustive_live_recovery_tree_5x5/run.sh prepare-pilot
bash experiments/vcg_exhaustive_live_recovery_tree_5x5/run.sh run-pilot
```

Only after inspecting pilot DAG sizes and wall times should the dense panel be
considered:

```bash
bash experiments/vcg_exhaustive_live_recovery_tree_5x5/run.sh prepare-full
bash experiments/vcg_exhaustive_live_recovery_tree_5x5/run.sh run-full
```

The root ledgers are atomic and resumable across method/lambda/root cells.
A partially explored individual root is rerun; a completed authenticated root
is reused.

## Claim boundary

This experiment can support exhaustive live confirmation only for the
declared finite disturbance set and the committed stored cohort. It is not a
claim about full dynamic episodes, Accept/Defer errors, future-arrival
uncertainty, block-clock perturbations, route-interior deviations, arbitrary
real-world disturbances, or hardware execution.
