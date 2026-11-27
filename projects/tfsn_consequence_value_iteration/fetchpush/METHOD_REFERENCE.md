# TF-SN FetchPush point-1 transfer

This adapter tests only the task-space clarification in the frozen R74 method:

\[
\phi(s) = \text{the state coordinate accepted as a goal by the supplied
executor}.
\]

For `FetchPush-v4`, the fixed executor accepts a block waypoint in the table
plane, so `phi(s)` is the block's `(x,y)` coordinate.  The remaining
goal-hidden physical observation (gripper, object-relative, joint and velocity
features) remains in the consequence-model state.  Native desired goals,
native rewards, and the evaluator's success radius are excluded from
preparation and from the planner/executor boundary.

Distances in this revision remain Euclidean in the native executor-goal
coordinates.  Reach-covariance-normalized distance is deliberately disabled;
enabling it would be the separate point-2 method change described in the
cross-domain specification.

The fixed command duration is the shared `tau=100` primitive steps.  The Fetch
episode contract is `T=200`, hence the planning horizon is derived as `H=2`.
Landing sigma, reach covariance, hubs, and latent-radius prior bounds are
estimated from reward-free preparation exactly as in the frozen method.

Observed task evidence contains only `phi(s)`, not an unobserved gripper or
velocity state.  Its graph association is therefore the nearest recorded node
in executor-goal space.  This is a representation adapter, not a spatial
threshold or an environment rule.

The `smoke` profile is implementation evidence only.  It uses a tiny data set
and may not be interpreted as FetchPush performance evidence.  The `full`
profile uses the shared 36,000-command preparation allocation.
