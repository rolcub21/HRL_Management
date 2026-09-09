# Recursive robust VCG kernel on reduced exact yards

This deterministic, no-training experiment constructs the recursive object
that the one-step bounded-execution probe intentionally did not claim.

It solves two finite closed-admission games:

1. A resource-augmented 4x4 recovery game where every strict macro may take
   zero or one extra primitive step. Remaining primitive budget is part of the
   state, so execution delays accumulate across macros.
2. A 1x5 two-block corridor where the terminal agent position may be the
   nominal endpoint or a clear traversable radius-one neighbor.

For each game it computes:

- the nominal completion kernel;
- the one-step robust set whose outcomes need only remain in the nominal
  kernel; and
- the least fixed-point recursive robust completion kernel and robust rank.

Here "kernel" means the least fixed-point robust-completion winning set (a
reachability attractor), not the greatest fixed-point viability kernel that
would permit safe but nonterminating cycles.

The policy comparison exhausts every declared disturbance branch. The
recursive controller ranks only the part of the robust invariant frontier
whose outcomes cannot increase robust rank, using
`q_operational - lambda * q_rehandles`. After one same-rank macro, it forces a
strictly rank-decreasing action. This bounded-liveness guard prevents cycles
and completes within twice the initial robust rank. The fixed merits are
transparent toy values, not neural-network predictions.

Run from the repository root:

```bash
bash experiments/vcg_recursive_robust_kernel_reduced_yard/run.sh
```

The report is written to
`results/vcg-recursive-robust-kernel-reduced-yard/recursive-robust-report.json`.

## Claim boundary

This is an exact recursive robust-completion result for reduced,
closed-admission recovery games. Duration uncertainty and endpoint uncertainty
are solved as separate games, not as a joint product set. The experiment does
not cover future arrivals, full dynamic episodes, learned-policy performance,
continuous disturbances, or hardware execution.

The recursive games use repository-native strict macros whose routes are safe
by construction; they do not add perturbed interior tube states. The separate
`vcg_robust_execution_probe.py` covers unsafe-tube and UNKNOWN rejection. Thus
this experiment establishes recursive endpoint/duration containment, while
the earlier probe supplies the one-step tube evidence.
