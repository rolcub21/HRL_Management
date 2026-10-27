E130 EXACT MACRO COST PREFLIGHT
==============================

Run from the repository root:
    bash experiments/conditioned_vcg/E130_exact_macro_cost_diagnostic_194k_v1/run.sh run
Analyze an existing result:
    bash experiments/conditioned_vcg/E130_exact_macro_cost_diagnostic_194k_v1/run.sh analyze

No training, replay insertion, optimizer steps or native controller changes.
Frozen E123 A episode-120, first historical validation instance 187800; this is
development evidence, not a fresh test. Audit at most 32 native decision states
and three lexically spread candidates per guard-admissible action type. Execute
alternatives only on physical clones; preserve Python/NumPy/Torch RNG states.
An alarm bounds the diagnostic to 180 seconds. Existing source checkpoints and
earlier experiment sources are never edited.

Costs are four unweighted components, matching E115/E123:
    tardiness accrual; earliness at delivery; equipment work; rehandles.
J = 2*T + E + 0.1*work + 10*rehandles; tolerance 20; gamma 1.
The predictor uses only released unfinished jobs, current routes and bound
recovery paths. It never reads unreleased-job durations or the future schedule.
Arrivals encountered by cloned execution belong to the observed outcome, not
to the predictor's information. The declared exactness gate tests both costs
and duration; passing this bounded test would not constitute a global proof.

RESULT
------
229 candidate executions: 27 Accept, 16 Defer, 93 Deliver, 93 Reconfigure.
All 229 component vectors matched. All 213 route-bound candidates also matched
duration. Two event-sensitive Defer candidates ended at the next arrival:
step 28: predicted 10, observed 2; step 54: predicted 10, observed 6.
Their costs remained zero, but the resulting time/observation was not the
predicted ten-step afterstate. Thus the full exact-macro gate did not pass.

IMPORTANT COVERAGE LIMIT: none of these physical samples accrued positive
tardiness. Earliness, equipment work and rehandles were nonzero in 44, 213 and
93 samples respectively. Unit tests cover overdue-job accrual, but that is not
a substitute for positive-tardiness physical cases before training.

NEXT TRAINING DESIGN (not implemented or launched by this preflight)
-----------------------------------------------------------------
Use a known computable cost K(s,a) and learn the residual R, not old Q plus K:
    Q_cost(s,a) = K(s,a) + R(s,a).
    target_R = C_observed - K_current
               + gamma**actual_duration * (1-terminal)
                 * (K_next_selected + R_target_next_selected).
All four heads use the SAME next candidate, chosen by the unchanged grouped
selector on the combined merits -J(K+R). Guard-forced choices remain forced.
Both action selection and TD selection must include K, including conservative
preference anchoring. Never add K to the old trained total-Q predictions.

For a validated route-bound macro, K is its exact complete immediate vector.
For event-sensitive waiting, a safe initial choice is K=0 and let the residual
learn its immediate-plus-future cost. A zero-cost wait can be recognized when
no released job can accrue cost throughout the cap; do not extrapolate that
special case to waiting while jobs are overdue. Use actual duration in replay.
This hybrid keeps predictable movement costs exact without claiming that all
waiting durations or future arrivals are known.

Before a matched training pilot: exercise overdue physical states and route
repair/arrival interference; test terminal residual targets, no-double-counting,
and common next-action indices for all heads. Keep grouped selection, objective,
certificate, guard, instances, initialization, exploration and budget identical
between total-Q and residual-Q arms. Compare greedy development validation,
not exploratory training returns. No independent test panel is opened here.
