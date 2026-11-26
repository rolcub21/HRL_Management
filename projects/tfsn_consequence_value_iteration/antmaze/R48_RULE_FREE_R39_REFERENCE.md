# AntMaze R48: rules-only R39/R46 port

R48 is the isolated AntMaze port used to measure only the effect of removing
the nine environment-dependent or Ant-only rules from the active R39 plus R46
stack. Historical R39 and R46 source, artifacts, and results remain unchanged.

The shared definition is
[`../shared/method_manifest.json`](../shared/method_manifest.json). The AntMaze
task contract is
[`r48_rule_free_protocol.json`](r48_rule_free_protocol.json). The benchmark
success radius belongs exclusively to that evaluator contract. It is not a
planner, posterior, passage, candidate, mobility, reach, or model input.

## Preserved behavior

R48 preserves R39's immediate adaptive-greedy live-passage acquisition and
R46's reward-free online connection correction. The separate audits of
one-step greedy search, R46 target semantics, and empirical passage matching
precision are explicitly deferred. Changing any of them would invalidate the
rules-only comparison. Passage matching may be measured by the read-only R48
diagnostic, but that diagnostic cannot change matches, weights, candidates, or
planner behavior.

The frozen aligned R30 consequence checkpoint and executor are also preserved.
R30 is the model deployed by the active R39/R46 line. Its Ant-specific posture
head includes orientation and is the environment adapter here; replacing it
with R43 would be a consequence-model intervention, not rule removal.

## Rule-free estimators

Let

\[
\kappa_\sigma(x,y)=\exp\!\left(-\frac{\|x-y\|^2}{2\sigma^2}\right),
\]

where \(\sigma\) is the held-out landing precision of the deployed model.
Search and passage gain use the residual field built from this same kernel.
No reward-contact radius is read.

For a modeled command, endpoint stall is

\[
\hat p_{\mathrm{stall}}(s,c)
=\mathbb E[\kappa_\sigma(\phi(s'),\phi(s))\mid s,c].
\]

The expectation is analytic for every spatial Gaussian component, ensemble
members are reduced pessimistically, and mobility is evaluated only for exact
nonidentity movement requests. There is no physical displacement threshold.

Recorded reach observations have continuous mass

\[
\omega_i=
\left(1-\kappa_\sigma(c_i,\phi(s_i))\right)
\left(1-\kappa_\sigma(\phi(s'_i),\phi(s_i))\right).
\]

The shared equal-mass reach representative procedure consumes these weights
without a cutoff. Stalls and nearly zero requests therefore lose influence
continuously.

Every model connection integrates all joint spatial and posture component
means:

\[
q_m(s,c,z)=\sum_{j,k}
\pi^{xy}_{m,j}\pi^{posture}_{m,j,k}
\kappa_\sigma(\phi(\bar s'_{m,j,k}),\phi(z))
M(\bar s'_{m,j,k}).
\]

The deployed connection is \(\min_m q_m\). No sampled endpoint count remains.

Recorded reliability and R46 online correction both use the manifest's one
declared evidence prior weight. The reliability neighbor cap and all three
candidate-family budgets are also manifest values. The command horizon is
derived as \(H=T/\tau\).

If residual search mass is exhausted, acquisition returns zero gain and emits
`posterior_exhausted=true`. It never installs a uniform fallback.

## Evidence boundary

The consumed two-goal R39 gate (tasks 3 and 5, five episodes each) measures the
isolated rules-only change. The subsequent full 15-task panel uses exactly the
same R48 planner and five-episode contract. Because all 15 tasks have
historical development exposure, the panel is development evidence rather
than confirmation evidence and is not yet the final unified TF-SN method.

Fresh full-panel preparation must use a separate output root. Every prepared
artifact records the exact shared-manifest hash and method-reference hash;
reuse is rejected if either hash changes. The passage-matching diagnostic
reads only the prepared graph, bridge, and recorded-trajectory artifacts. It
reports complete unweighted and bridge-weighted continuous error quantiles in
meters and learned-\(\sigma\) units, introduces no acceptance threshold, takes
no environment steps, and accesses no task reward or hidden goal.

The matched passage prototypes affect acquisition passage-gain geometry. They
do not train the frozen R30 consequence model, define recorded-transition
reliability, or determine the exact mixture bridge connection weights. A poor
match can therefore corrupt which route appears to cover residual search mass,
but it cannot retroactively corrupt those independent estimators.
