# R46: online consequence correction on frozen R39

R46 is a narrow consumed-development component gate.  It keeps R39's graph,
offline consequence ensemble, mobility model, passage geometry, posterior,
planner objective, executor, tasks, resets, and five-episode budget frozen.
It adds only a reward-free posterior correction from consequences observed
during the current task.

After an executed command, the observation is its continuous landing-kernel
value

\[
y=\exp\!\left(-\frac{\lVert x'-g\rVert^2}{2\sigma^2}\right).
\]

There is no stall threshold or failure label.  For a new state-command query,
past observations are weighted by a Gaussian in the existing standardized
full-state/absolute-command feature space.  State scales come from the frozen
recorded graph and the command scale is the deployed landing precision
\(\sigma\).  With one unit of prior effective count, each ensemble member's
offline connection estimate is updated by the local observations.  Thus one
identical observed miss halves an offline connection estimate, while a
dissimilar command is unchanged.

The correction memory is reset between tasks and retained across physical
episode resets within a task.  It consumes neither hidden-goal coordinates nor
task reward, performs no gradient update, and does not alter an episode's
termination.  Commands continue under the benchmark's unchanged fixed-duration
contract even after the Ant becomes immobile.

The decisive gate is task 3: after the deterministic command-2 miss is
observed in episode 1, episode 2 should avoid issuing that exact command from
the repeated reset trajectory.  Task 5 is retained as the second consumed
development goal.  This is component evidence only, not confirmation evidence.
