# R33 posterior-sampled hidden-goal acquisition

R33 changes only acquisition in the frozen R31 AntMaze development system.
The aligned R30 consequence model, recorded-transition graph, reach-based
bridges, factorized endpoint model, mobility model, executor, fixed command
duration, finite physical-episode horizon, and R31 return phase are unchanged.

Before reward contact, task memory defines the residual spatial belief

\[
b_t(x) \propto 1-\max_{j\leq t}\kappa_\sigma(x,\phi(s_j)).
\]

The prior support is the set of unique landing-precision lattice cells touched
by reward-free primitive trajectories.  Consequently each unit of represented
space enters the prior once; reward-free visitation density does not multiply
its probability.

At the beginning of each physical episode, R33 draws one hypothetical goal
from the normalized residual belief.  The frozen supplied-goal R22 planner is
then read against a Gaussian field centered on that draw for the complete
physical episode.  Holding the draw for the benchmark-defined episode horizon
allows the planner to execute detours.  At the next physical reset, the draw is
repeated from the posterior updated by every state actually visited so far.

The hidden evaluation goal is never an input to the sampler, planner, executor,
or learned models.  It is read only by the evaluator to detect contact.  After
contact, R33 discards the hypothetical goal and uses R31's unchanged empirical
reward-rate return phase.

This is a consumed-development test of posterior sampling.  It is not fresh
confirmation evidence.  There are no distance masks, reach thresholds,
posture rules, wall maps, environment-specific candidates, or task-time
gradient updates.
