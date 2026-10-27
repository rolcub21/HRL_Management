E144 ARRIVAL-AWARE WARM START, V2
================================
Original v1 code, contracts and episode0 artifacts remain intact.
Run v2 explicitly; never resume the older 360-episode command for this stage.

  MPLCONFIGDIR=/tmp/e144-mpl .venv/bin/python -B -m experiments.conditioned_vcg.E144_warm_start_mixture.run_arrival train --through-episode 60 --device cuda

Status: same module, replace train... with status.
Output: results/vcg-conditioned-e144-warm-start-mixture-arrival-v2-repair1/
Failed arrival-v2 episode0 artifacts remain untouched. No training episodes
had completed. Repair1 corrects a rollout-interface mismatch: the old E138
learner reset Defer streak on arrival-only observed_event, whereas E137/E141
and the certified comparator reset it only on flow progress. Both training
and validation now use the comparator's flow-event counter. No cost/ranking
tolerance was widened. The episode0 exact-cost reproduction gate remains.

Two fixed, current-boundary observations:
  count of incoming releases in (t-120,t], divided by120 primitive steps;
  primitive steps since last observed incoming release (or episode start).
Initial inventory is excluded. Delivered incoming jobs remain in arrival
history. Unreleased timestamps, future dates, regime IDs, fill/load flags and
simulator arrival-rate parameters are not exposed. Startup denominator stays
120; existing elapsed-time context lets the model distinguish startup.
This history is a useful partial observation, NOT a proof of a Markov state.

The old context encoder and all original weights are unchanged. A separate
2-to-context-embedding linear projection, without bias, starts exactly zero.
Its output adds to the old context embedding; source predictions are preserved
at initialization, while gradients can train the new connections. Every replay
record stores the observed context in its immutable ArrivalState. Target
network uses the same inputs/architecture; all four heads retain one shared
selected successor. Native objective/certification/guard/loss/fast caches remain.
Input units remain source3600, execution cap6000. No cost/head normalization.
Optimizer/replay fresh. Same mixture exploration .90->.05/6000 decisions.

First stage: exactly60 new adaptation episodes =10 per regime; pause regardless
of outcome. No automatic additional seeds or training past60. Validation at
0,6,30,60 on six fixed workload development cases plus five native controls.
Endpoint0 must reproduce completed E141 original seed0 J within0.001 before
training. Timeouts retained as failures; correctness/reproduction errors stop.

Continue eligibility is fixed now, not tuned after seeing results:
- all11 cases complete at0,30,60;
- easy/.60 J at30 AND60 within10% of episode0 (~73.1);
- sum J of four congested cases at60 <=95% of episode30, and below episode0.
Then continuation may be considered, never automatically launched. A failed
gate means stop/reassess, not proof of forgetting or of any particular remedy.
Native controls and each hard cell remain individually reported; a sum can
hide individual regressions and is not a per-cell superiority claim.

All available input information remains observational: baselines could compute
the same release history. Their existing rules are unchanged; we do not expose
advance information to the learned method alone. Test panel sealed. Baseline
macro mismatch and one changed-workload validation instance/cell remain limits.
