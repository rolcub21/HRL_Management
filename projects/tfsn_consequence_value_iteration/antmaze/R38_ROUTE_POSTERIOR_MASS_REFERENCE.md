# R38 route posterior-mass selection

R38 retains R37's exact deterministic search posterior and changes only how
the batch of posterior hypotheses is ranked.  R36/R37 ranked a hypothesis by
the supplied-goal value of reaching that single point.  That criterion does
not measure how much hidden-goal probability a route would rule out.

For each of the same 16 posterior hypotheses, R38 first computes the frozen
R22 greedy graph route from the current projected state for the remaining
command horizon.  It then evaluates the union of the route's reward-free
empirical command passages under the exact detection model.  A posterior
support point is covered by a passage when its distance from that passage is
at most the declared contact radius \(\rho\).  A support point contributes at
most once, so loops receive no repeated credit.

If edge \(e\) is reached with probability \(p_e\), its newly covered residual
posterior mass is weighted by \(p_e\).  Recorded-edge reliability and frozen
model-bridge connection weights supply these probabilities.  The selected
hypothesis maximizes

\[
  \frac{1}{\sum_x A_t(x)}
  \sum_e p_e
  \sum_{x\ \text{first covered by }e} A_t(x).
\]

Recorded edges use their recorded primitive passages.  Model bridges use the
already frozen locally matched reward-free empirical passages.  The passage
lattice is the existing landing-precision lattice; it is computational only
and introduces no search region or spatial threshold.  The detection tube is
set only by the task's declared contact radius.

The posterior, hypothesis batch size, graph, supplied-goal route values,
consequence model, executor, fixed command contract, return phase, and
development tasks are unchanged.  Hidden goal coordinates remain
evaluator-only.  This is a consumed two-goal development gate, not
confirmation evidence.
