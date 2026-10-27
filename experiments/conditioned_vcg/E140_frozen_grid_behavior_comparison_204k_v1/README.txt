E140 FROZEN TRAINED CHECKPOINT BEHAVIOR COMPARISON
================================================
This evaluates REAL trained seed-0/seed-1 episode-42 checkpoints. Not scratch
policies, not training, and NOT the untouched test panel. Same six E138 greedy
development instances and release-fixed dates for every method. Because this
panel was used during learning, results are descriptive development evidence,
not independent held-out-test claims or statistically powered comparisons.

    cd /home/ai_diagnosis/HRL_Management
    bash experiments/conditioned_vcg/E140_frozen_grid_behavior_comparison_204k_v1/run.sh prepare
    bash experiments/conditioned_vcg/E140_frozen_grid_behavior_comparison_204k_v1/run.sh run

Prepare authenticates upstream source and freezes compact INFERENCE snapshots
of seeds0/1 at episode42. Original resumable checkpoints are hashed/read-only,
never overwritten. Replay and optimizer are excluded from evaluation snapshots.
Once frozen they do not track a changing latest.pth. Snapshot/hash/endpoint are
recorded in each row; evaluation snapshots cannot resume training.

30 episodes: six regimes x [Dynamic PSLAP, GA, certified Dynamic, VCG seed0,
VCG seed1]. One CPU thread; no GPU allocation for scoring or learning, so seed2
may continue. Shared host CPU/memory contention may still affect latency.
Every row has the SAME 15-minute wall cap, 6000-step/1000-decision horizon and
2000-node certificate budget. These are separately labelled extended-budget
evaluations; original E137/E138 timeouts remain untouched. Worst-case cap budget
7.5 hours for 30 rows, not an expected-time estimate. Long run is user launched.
Native plain-baseline compound-versus-elementary certified macro mismatch stays
explicit; this is not a new identical-action-set benchmark.

Greedy epsilon zero, no optimizer or replay writes. Check online AND target
weights and gradient counters unchanged after each learned-policy episode.
Full episode recorder: primary J/components, sensitivity J_W, completion,
release MAE, within +/-20, early/late counts, admission wait (incoming only),
rehandles, steps/delivery, primitive travel/WAIT/handling counts, macros and
per-job timestamps. Incomplete costs are observed only and never ranked against
completed costs; all failures/timeouts remain in the table/denominator.

Rows are reused on rerun, including saved failures. Different budgets/source
require a new output version; no quiet retry. Ctrl-C/SIGTERM preserves the
current partial row and exits. Hard kill can lose only the current row.
Do not run two writers for the same coordinate.

    bash experiments/conditioned_vcg/E140_frozen_grid_behavior_comparison_204k_v1/run.sh status

Outputs: e140-table.md, e140-report.json, rows/ with action/job traces, models/
with frozen inference checkpoints and provenance. Baselines are run on these
exact development workloads; do not copy costs from E137's different instances.

When seed 2 finishes (episode60 OR a recorded stop), optionally:
    bash experiments/conditioned_vcg/E140_frozen_grid_behavior_comparison_204k_v1/run.sh add-seed2

That freezes and evaluates its own terminal endpoint, reusing existing rows and
baselines. The table explicitly labels the checkpoint episode. If seed2 stops
at a different budget than42, do not pool this as a matched-episode three-seed
estimate. No automatic seed2 snapshot while it is still running. No Stage B,
training continuation, sealed test generation or opening is performed here.
