# R72: online correction of planned hub-route outcomes

R72 retains R71's frozen E549-5M executor, R52 reward-free preparation,
area-weighted dynamic hubs, 20-command physical episodes, ten episodes per
task, and sigma-generalized destination-arrival correction for individual
model bridges.

R71 revealed a second consequence at a longer temporal scale.  A sequence of
short links can have high individual arrival probability while failing to
reach the selected hub before the physical episode ends.  R72 treats the
planner's prediction that a hub will be reached within the remaining command
horizon as a Bernoulli prediction and corrects it with the realized route
outcome.

An attempt begins when a hub is selected.  It succeeds if the Ant comes within
the deployed landing scale `sigma` of that hub before the attempt horizon
expires.  It fails only if the physical episode ends first without task
contact.  Task contact censors the attempt because acquisition stops before
the predicted route outcome can be observed.  The current R71 link-corrected
reachability is the prior mean and counts as one observation, using the same
globally declared prior weight as recorded-transition reliability.

Evidence generalizes continuously between route options:

```text
exp(-0.5 * (
    ||attempt_source - observed_source||^2 / sigma^2
  + ||attempt_hub - observed_hub||^2 / sigma^2
))
```

Only observations with the same remaining command horizon estimate the same
Bernoulli event.  This is exact conditioning, not a distance threshold or a
new scale.  There is no similarity cutoff, neighbour count, wall test, hub
blacklist, or environment-specific rule.

The corrected hub probability is

```text
(prior_weight * current_link_corrected_reach + weighted_arrivals)
----------------------------------------------------------------
           (prior_weight + weighted_attempts)
```

and hub selection remains the exact maximizer of posterior coverage times
corrected reachability.  Coverage is still a valid branch-and-bound upper
bound because the corrected probability remains in `[0, 1]`.  The hidden goal
and maze geometry are unavailable to the planner.

## Computational realization

The exact Bellman recursion is evaluated for a batch of target hubs at once
with targets as the leading tensor dimension.  It uses the already declared
R69 hub-scoring chunk of 64 as its computational batch and retains only the
previous and current Bellman layers.  Latent-radius coverage is evaluated in
the same declared chunks.  Neither batching nor rolling storage changes the
recursion, candidate set, coverage values, branch-and-bound upper bound, or
global argmax.  R71's bridge-evidence spreading remains the same NumPy
calculation.  A saved-selection equivalence audit gates use of this backend.

The nine goals are consumed development evidence.  R72 asks whether the
late-episode knots disappear when failed planned routes, rather than only
their individual links, are corrected online.
