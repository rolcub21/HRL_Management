# R37 exact detection posterior

R37 is R36 with one correction to the hidden-goal posterior.  The task's
observation model is deterministic: reward is observed exactly when a
primitive Ant position lies within the declared contact radius \(\rho\) of the
hidden goal.  Therefore negative observations imply the exact posterior field

\[
A_t(x)=\mathbf 1\!\left\{
\min_{j\leq t}\lVert x-\phi(s_j)\rVert>\rho
\right\}.
\]

The prior is uniform over the existing reward-free reachable support.  A
support point retains unit mass until a realized primitive trajectory enters
its detection footprint, after which it has zero mass.  No spatial partition,
Gaussian tail, region shape, or additional bandwidth is introduced.

R36's batch of 16 posterior hypotheses, frozen R22 reach-value selection,
graph, models, executor, fixed command contract, and posterior-resampling logic
are unchanged.  Landing precision remains the scale for aliasing, hypothesis
routing, and return.  The known reward-contact radius is used only by the
acquisition observation model.  Hidden goal coordinates remain evaluator-only.

This is a consumed two-goal development gate, not confirmation evidence.
