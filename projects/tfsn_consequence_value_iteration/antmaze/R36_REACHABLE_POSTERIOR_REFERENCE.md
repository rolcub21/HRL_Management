# R36 reachable posterior-region acquisition

R36 is R35 with one change to posterior-hypothesis selection.  A single draw
from the residual hidden-goal posterior can lie beyond what the frozen R22
planner can reach in the commands left in the physical episode.  R36 draws a
small, fixed global batch of posterior hypotheses and pursues the member with
the greatest frozen R22 graph value from the current full-state projection.

For hypotheses \(g_1,\ldots,g_K\sim p_t(g)\), R36 chooses

\[
\tilde g_t=\arg\max_{g_k}V^{\mathrm{R22}}_R(i(s_t);g_k),
\]

where \(R\) is the number of commands remaining and \(i(s_t)\) is R22's
existing full-state graph projection.  The batch size \(K=16\) is a global
computational budget, not an AntMaze distance, mask, threshold, or cap on
physical reach.  R22's learned graph and consequence weights decide
reachability.

The detection-footprint posterior, sequential survival update, executor,
graph, learned models, command duration, return phase, and evaluation protocol
are unchanged from R35.  The hidden goal is evaluator-only.  This is a
consumed two-goal development gate and not confirmation evidence.
