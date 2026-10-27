E125: completed E123 A versus matched Dynamic PSLAP and GA
=======================================================

This is a small development reference, not an untouched test or training run.
E123 A seed 0 is frozen after 120 episodes. Reuse its already-saved greedy
validation on seeds 187800--187804. Authenticate the source contract, completed
summary, episode-120 candidate and fixed late-window validation digest.

Use the exact same EpisodeInstances for the unchanged E116 baseline adapters:
empty 8x8, 36 jobs, release gap 30, Poisson(80) stays. Preparation verifies every
instance ID against every source checkpoint's realized validation trajectory.
Dynamic PSLAP runs once per instance; GA uses its established four stochastic
replicates per instance. Thus 25 new baseline executions, no VCG reruns or fits.
A3C is omitted because no matching size/workload retraining is available here.

Objective: J = 2*T + E + 0.1*L + 10*R. Dates are fixed at release. Earliness
and tardiness are beyond +/-20; L counts non-WAIT primitive actions; R counts
physical rehandles. Total steps and steps/delivery also include WAIT. The
baseline controller itself is unchanged, not newly optimized for this J.
Incomplete executions retain their completion outcome but have no comparable J.

Primary VCG statistic: average checkpoints 96,102,108,114,120 within each
instance before averaging five paired instances. Episode 120 is also reported,
not selected as a best checkpoint. Average GA replicates within each instance.
Paired nominal 95% t intervals are descriptive over n=5, not n=25; they are not
Holm-adjusted confirmatory results. No independent test panel is opened.

Commands (CPU, one thread; existing repository .venv):
  bash experiments/conditioned_vcg/E125_e123_matched_baseline_reference_189k_v1/run.sh prepare
  bash experiments/conditioned_vcg/E125_e123_matched_baseline_reference_189k_v1/run.sh run-all
  bash experiments/conditioned_vcg/E125_e123_matched_baseline_reference_189k_v1/run.sh status

Coordinates are individually saved and reused. Source E123 outputs are read
only. Do not change source summaries or candidate files during this reference.
Reports are under results/vcg-conditioned-e125-e123-matched-baseline-reference-189k-v1/.
The JSON includes per-instance paired differences and baseline amortized
decision latency. Historical cached VCG timings and new baseline timings are
not a controlled inference-speed benchmark.

Observed reference: all methods completed; every method had zero rehandles.
A's fixed-window J is lower on this panel, while its outside wait and storage
MAE are higher. This supports further evaluation, not a general superiority,
cross-seed replication, or rehandling/preference-control claim. This low-load
panel does not by itself establish behavior when storage access is stressed.
