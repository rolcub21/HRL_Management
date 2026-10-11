# Constrained VCG-SMDP V2

## Status and scope

This is a new, incompatible method family. It does not modify VCG-Dense
V1.1, the prospective V1.2 training protocol, or any saved V1.x artifact.
V2 is teacher-free: deterministic baselines never provide actions, scores,
features, targets, or feasibility labels.

The exact verifier remains the sole authority over the executable frontier.
The resource constraint ranks actions only after exact certification and can
never suppress a recovery-witness action selected by the liveness guard.

## Constrained objective

For an authenticated episode with `N_req` required blocks, V2 solves the
finite-episode constrained problem

```
maximize    J_op(pi)
subject to  E_pi[N_phys - beta * N_req / 100] <= 0,
```

where `N_phys` is the number of observed storage-to-storage physical moves and
`beta` is a predeclared allowance in rehandles per 100 required blocks. The
denominator is required workload, never successful deliveries, so incomplete
episodes cannot manufacture apparent compliance.

This is an expected constraint, not a hard per-episode action cap. A hard cap
would require a resource-indexed viability kernel and is a different method.

The operational critic estimates the dense timing/travel objective with
`gamma_op = 0.99` and frozen scale `eta = 0.01`. The physical-cost critic
estimates undiscounted finite-episode rehandles with `gamma_phys = 1`.
Remaining finite-horizon time and the Hold idle-budget state are therefore
part of the V2 representation.

## Vector critic and shared policy

The shared graph trunk has two targetable heads:

```
Q_op(s, c)    operational return
Q_phys(s, c)  expected physical rehandles, constrained nonnegative
```

Replay stores the raw vector transition

```
(s, c, R_op^(k), C_phys^(k), k, done, exact-safe next frontier).
```

The two critics are in current-boundary units, while the mixed-discount
Lagrangian is defined in episode-start units. Therefore, at primitive elapsed
time `t` and detached dual value `lambda`, action merit is

```
Z_lambda,t(s, c) = gamma_op^t * Q_op(s, c)
                   - lambda * Q_phys(s, c).
```

Using a constant `Q_op - lambda Q_phys` is forbidden for this mixed-discount
objective. The within-group and between-group temperatures are fixed in the
same scaled episode-start-return units as `Z`; `lambda` therefore has units of
scaled discounted operational return per raw physical rehandle. Changing
`eta`, either temperature, or either discount defines a different method.

Accept, Deliver, Reconfigure, and Hold are separate cardinality-normalized
control groups. The online `Z_lambda` values induce one nested within-group and
between-group regularized policy. Both target heads are evaluated under that
same policy. A return-only continuation and a cost-only continuation are
forbidden because they would train the two components under different future
policies.

The component heads remain raw expected return/cost estimators: KL/entropy
bonuses are not inserted into either component target. The method is described
as regularized policy improvement with vector Expected-SARSA/Double-Q
evaluation, not as two independent soft-Bellman fixed points.

Training samples the induced nested finite-temperature policy; no additional
epsilon policy is mixed into the dual rollouts. Deployment uses the
deterministic MAP realization: maximize the normalized group merit, then
maximize `Z_lambda` within the chosen group. The paper and audit must
distinguish the empirical MAP budget gate from the expected constraint learned
under the stochastic regularized policy.

## Dual update

After a predeclared batch of training episodes,

```
g_i = N_phys,i - beta * N_req,i / 100
lambda <- project_[0, lambda_max](lambda + alpha_lambda * mean_i(g_i)).
```

All finite-horizon training episodes enter the residual, including incomplete
episodes. Validation and deployment freeze `lambda` and perform no replay,
gradient, target-network, or dual update. If the dual is saturated while the
validation residual remains positive, the run is marked budget-infeasible;
it is not presented as converged.

## Certified Hold

Hold reuses the existing `DEFER` interface rather than changing V1 enums or
checkpoint dimensions. It is a bounded, WAIT-only, event-interruptible macro,
not a synthetic self-loop. The observed successor is exactly recertified and
stored in replay after execution.

V2 may expose a small, canonical horizon family when admitted work has
positive slack, including while an inbound block is waiting. It remains
unavailable when work is already due, the current recovery state is not
exactly SAFE, the finite horizon is exhausted, or its primitive idle-step
budget is exhausted. No unarrived schedule field is read. A retained recovery
witness always removes Hold from the live frontier.

The primitive idle budget, rather than a count of Hold decisions, is part of
the Markov state. It resets only after an observable event, Accept, Deliver,
or authenticated structural recovery progress; an arbitrary nonprogress
reconfiguration cannot reset it.

## Deployment eligibility

Checkpoint selection is lexicographic:

1. strict completion and exact-safety integrity;
2. no illegal drops, fallbacks, witness mismatches, or executor/certificate
   mismatches;
3. the predeclared workload-normalized rehandle budget;
4. maximum raw dense operational return;
5. lower MAE, lower physical rehandles, then earlier checkpoint.

If no validation checkpoint is constraint-feasible, V2 produces no
deployment-eligible checkpoint. The Lagrangian scalar is never used as a
reported performance metric or checkpoint-selection score.
