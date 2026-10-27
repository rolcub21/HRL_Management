E137 USER-LAUNCHED GRID ROLLOUT READINESS
=======================================
This finishes the development episode recorder and exposes bounded full-loop
checks for the user to run. NO TRAINING, NO OLD CHECKPOINT LOADING, NO SEALED TESTS.
Uses exactly one previously generated calibration instance per six regimes.
All rows use external release-fixed dates and the same J/J_W accounting.

Methods: informed Dynamic PSLAP, informed GA (existing repaired assignment source,
16 population, 10 generations, fixed seed), certified Dynamic, and SCRATCH VCG.
Scratch VCG is solely an untrained integration control, NOT a quality comparison
or evidence for/against the learned architecture. No model weights are saved.
The fast E123-v2 input/state-sharing implementation remains in the certified
shell. CPU, one PyTorch thread; no TD updates or GPU required for this gate.

    cd /home/ai_diagnosis/HRL_Management
    bash experiments/conditioned_vcg/E137_grid_rollout_readiness_201k_v1/run.sh prepare
    bash experiments/conditioned_vcg/E137_grid_rollout_readiness_201k_v1/run.sh run

There are 24 coordinates, run sequentially, each capped at 300 seconds, 1000
decisions and 6000 primitive steps; certificate search is capped at 2000 nodes.
Worst-case episode-loop budget is about TWO HOURS (plus setup/I/O). This is a
diagnostic resource budget, not a calibrated final comparison/search budget.
UNKNOWN/frontier exhaustion, failed macros, physical violations, decision/horizon
limits and timeouts are retained. No automatic promotion to training occurs.

For an incremental check, or to run methods in separate tmux sessions:
    bash experiments/conditioned_vcg/E137_grid_rollout_readiness_201k_v1/run.sh run --method dynamic_pslap
    bash experiments/conditioned_vcg/E137_grid_rollout_readiness_201k_v1/run.sh run --method ga
    bash experiments/conditioned_vcg/E137_grid_rollout_readiness_201k_v1/run.sh run --method certified_dynamic
    bash experiments/conditioned_vcg/E137_grid_rollout_readiness_201k_v1/run.sh run --method scratch_vcg_interface
Prefer sequential runs initially: concurrent runs would confound latency.
Optional --fill 0 --load 0.6 selects just the empty low-load coordinate.

    bash experiments/conditioned_vcg/E137_grid_rollout_readiness_201k_v1/run.sh status

Completed rows AND saved failures are reused, not rerun automatically. To retry
failed/interrupted rows, add --retry-failed; original attempts are archived and
completed rows cannot be overwritten. Ctrl-C / SIGTERM saves the current partial
row and exits, without continuing to the next method. A hard kill cannot save
the in-progress row; previous atomic row files remain intact. This is row-level
resumption, not an in-episode checkpoint. Do not run two writers on the same row.
Changing source or budgets requires a fresh --output and a separate contract.

Reports: e137-report.json, e137-table.md; per-row primitive traces in rows/;
per-coordinate latest progress in progress/. Console progress prints roughly
every 30 seconds at macro boundaries. An expensive frontier may be silent until
it completes or reaches the wall cap. Report completion/failure before costs.
Incomplete rows have observed partial costs but are ineligible for cost ranking.
An interrupted primitive can produce an accounting_error: retain that failure,
do not fabricate a consistent completed trajectory.

IMPORTANT MACRO LIMITATION
The certified pair has the exact same certificate, guard and Defer preparation.
The native plain baselines use compound retrieval options, potentially including
several route-clearing relocations. VCG uses elementary certified recovery
macros. The underlying physical movement/handling checks are shared here, but
this is NOT YET an identical parameterized macro action set. This readiness
panel exposes execution/clock failures; it cannot establish the final grid's
interface-fairness or learning-contribution claim. Decide whether to adapt native
rule ranking to the common elementary macro library before a final comparison.

LIVENESS / TRAINING FREEZE
The 6000-step horizon is an empirical execution cap, not a uniform completion
proof. Current recoverability certificate protects admitted work, not arbitrary
future admissions or a uniform next-progress deadline. Do not inherit E123's
old 3600 bound, call observed completion a theorem, or silently change guards.
After these user-run results, settle that claim and macro alignment, then freeze
mixture sampling, disjoint manifests, exploration/budget/checkpoint selection,
final GA/search budgets and statistical hypotheses. E137 freezes only this
diagnostic's budgets, not the eventual mixture-training settings. Stage B waits.
