# Frozen R74/R15 aligned multiseed rebuild

This protocol rebuilds TF-SN preparation independently for seeds 0, 1, and 2
in AntMaze and PointMaze.  It changes no frozen decision equation.  Its purpose
is to replace the historical, domain-specific preparation mixtures with the
single collection and fitting procedure in
`FROZEN_R74_METHOD_SPECIFICATION.md`.

Each domain and seed receives exactly 36,000 reward-free commands:

- 24,000 bootstrap commands sampled uniformly from the exact xy coordinates
  observed so far;
- 6,000 commands selected by the frozen acquisition controller under an
  all-negative consequence stream; and
- 6,000 commands selected by the frozen return router toward synthetic anchors
  drawn uniformly from occupied landing-scale support cells.

All executed transitions enter the consequence fit and recorded graph.  The
alignment controllers are the already frozen R74 AntMaze and R15 PointMaze
controllers.  They use the current frozen artifacts only to select collection
commands; the rebuilt artifacts are fitted solely to transitions executed by
the seed's declared executor.

Preparation seed controls resets, bootstrap sampling, grouped holdout, and
model initialization.  PointMaze seed `k` uses the frozen executor checkpoint
with training seed `k`.  AntMaze has one frozen E549-5M executor, so seeds vary
TF-SN preparation only.

No benchmark goal, benchmark reward, success radius, maze geometry, or
confirmation-panel observation enters collection or fitting.  Synthetic return
anchors are preparation labels drawn from reward-free support and are never
reported as task evidence.

## Required comparisons

After seed 0 is rebuilt:

1. rerun the consumed PointMaze 24-task development gate and compare it with
   the frozen R15 24/24 result;
2. rerun the consumed AntMaze original-nine development panel and compare
   acquisition at episodes 10 and 20, conditional return, and per-task rows
   with R75; and
3. audit the AntMaze task-6 false shortcut from
   `(9.6958466, -1.2592292)` to `(9.7764997, 4.7351899)`.  Report its old and
   rebuilt nearest bridge weights, the reach-kernel-weighted family mean, and
   the maximum continuous `kernel * weight` contribution.  The kernel uses the
   empirical reach covariance on both sides and has no cutoff.  Maze geometry
   is diagnostic only.

Seed 0 is an equivalence check, not a tuning gate.  A difference is reported;
it does not authorize changing a setting before seeds 1 and 2.
