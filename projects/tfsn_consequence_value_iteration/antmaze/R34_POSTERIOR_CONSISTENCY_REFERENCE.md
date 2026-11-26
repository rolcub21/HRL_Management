# R34 posterior-consistent hidden-goal acquisition

R34 is R33 with one correction to the lifetime of a sampled hypothetical
goal.  All R31 artifacts, the R30 consequence model, the supplied-goal R22
planner, executor, graph, candidates, command duration, physical-episode
horizon, landing scale, and return phase remain unchanged.

R33 drew a hypothetical goal from the normalized residual spatial belief at
each physical reset and retained it for the whole episode.  This wasted
commands after the Ant visited that hypothesis without observing reward.

R34 treats the draw as a posterior particle.  Let

\[
A_t(x)=1-\max_{j\leq t}\kappa_\sigma(x,\phi(s_j))
\]

be the unnormalized survival likelihood used by the existing search field.
After a new realized command trajectory is incorporated, the current
hypothesis is retained with probability

\[
p_{\mathrm{keep}}=
\operatorname{clip}\left(
\frac{A_t(\tilde g)}{A_{t-1}(\tilde g)},0,1
\right).
\]

If it is rejected, a replacement is sampled from the normalized current
residual belief.  A detour that does not sweep the hypothesis has retention
probability one.  Visiting the hypothesis without reward drives its retention
probability toward zero.  There is no distance cutoff, posterior threshold,
wall map, posture rule, or environment-specific parameter.

One fresh posterior draw is still made at each benchmark-defined physical
reset.  Within an episode, additional draws occur only through the survival
test above.  The hidden evaluation goal remains evaluator-only.  Observing
reward switches to the unchanged R31 return phase.

This is a consumed-development gate, not confirmation evidence.
