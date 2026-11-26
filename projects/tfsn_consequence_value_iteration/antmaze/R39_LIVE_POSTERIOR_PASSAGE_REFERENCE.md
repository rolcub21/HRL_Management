# R39 live posterior-passage acquisition

R39 rejects R38's offline graph-route proxy.  Acquisition scores the actual
live commands considered by the frozen R22 decision layer from the Ant's
current full state.  Candidate generation, consequence inference, empirical
passage matching, executor, and command duration are unchanged.

For candidate command \(g\), let \(\hat p(g\mid s_t)\) be the pessimistic
ensemble probability that the executor realizes its matched passage, and let
\(T(s_t,g)\) be that reward-free empirical passage.  The acquisition score is

\[
 Q_t(g)=\hat p(g\mid s_t)
 \frac{\sum_x A_t(x)\,
   \mathbf 1\{d(x,T(s_t,g))\leq\rho\}}
      {\sum_x A_t(x)}.
\]

Here \(A_t\) is R37's exact hard posterior on the reachable support and
\(\rho\) is the declared contact radius.  The numerator is therefore the
exact residual posterior mass that this physical command is predicted to
rule out.  The posterior is updated after every command, so already covered
mass is zero at the next decision and receives no repeated credit.  This is
adaptive greedy maximization of detection coverage; it introduces no region,
distance cutoff, Gaussian search tail, or planning scale.

Recorded and model-bridge candidates use the same locally matched empirical
passages as R31.  Consequence probabilities are evaluated from the live full
state.  The graph's acquisition continuation values may propose the existing
alias-advance candidates, but they do not enter the final acquisition score.
Return is exactly the unchanged R31 return planner.

This is a consumed two-goal development gate.  Hidden goal coordinates remain
evaluator-only and the confirmation panel remains sealed.
