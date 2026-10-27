E126: release-clock-informed existing baselines
==============================================

Question: does E123 A retain E125's apparent advantage when the existing
Dynamic PSLAP and GA rules receive the release-fixed due date d=arrival+stay?
No new GA fitness, scheduler preference, search budget or learning is added.

This is separate from E125 and modifies no shared Shipyard source file. Scoped
planning adapters replace remaining time at all relevant boundaries:
  - stored-yard snapshots, retrieval slack, and retrieval-triggered Defer;
  - inbound assignment previews/commits and physical candidate-mask views;
  - ordinary and capacity-aware GA arrived-pending-job views;
  - projected post-admission states, aging both existing and inbound jobs.

Physical release dates, stay durations, admission timestamps and execution
remain native. Only already-arrived job metadata is supplied. Planning a
future completion of an already-arrived job is allowed; inspecting unreleased
jobs is not. Per-coordinate audits record planning-clock calls and example
inputs, and verify the physical job schedule was not mutated.

Same five E123 development EpisodeInstances, ID-checked against all fixed
late-window validations. Dynamic PSLAP runs once per instance; GA uses the
same four random replicates, fitness and population/generation settings as
E125. All trajectories use the same J=2T+E+0.1L+10R, release dates and +/-20
tolerance. Equipment work excludes WAIT; total steps includes WAIT.

Reuse frozen E123 A results at 96,102,108,114,120, averaged within instance;
also show the prespecified final checkpoint 120. Average GA replicates within
instance. Five instances are the paired sample size, not 25 checkpoint rows.
Nominal paired t intervals remain exploratory, without a multiplicity-adjusted
paper claim. No independent test, training, seed selection or checkpoint
selection. Historical VCG and new baseline latencies are not controlled.

Run (CPU, one thread; rows are resumable):
  bash experiments/conditioned_vcg/E126_release_clock_informed_baselines_190k_v1/run.sh run-all
  bash experiments/conditioned_vcg/E126_release_clock_informed_baselines_190k_v1/run.sh status

Accepted output:
  results/vcg-conditioned-e126-release-clock-informed-baselines-190k-v1-repair2/

Earlier diagnostic outputs are retained, not pooled into this comparison.
The original partial attempt was stopped by the GA pending-clock coverage
assertion, exposing its separate capacity-aware queue implementation. Repair1
covered that path; a test-fixture API omission was then fixed. Repair2 reruns
all coordinates after all tests pass, with unchanged clock/control logic from
repair1. Outcomes are not used to choose settings or tasks.

Result: all 25 accepted baseline trajectories strictly complete, with no
rehandles. Release-informed Dynamic PSLAP J=69.84; GA J=73.89; E123 A's fixed
late-window J=79.14 and final checkpoint J=76.38. The information correction
removes and reverses E125's apparent cost advantage on this panel. This does
not establish universal baseline superiority or assess heavily occupied yards.
