# AntMaze R51 destination-arrival correction

R51 changes one component of the frozen R48 AntMaze development planner.  An
online observation for edge (e) is

\[
o_e = \mathbb{1}\{\lVert \phi(s') - \phi(d_e)\rVert_2 \leq \sigma\},
\]

where (s') is the actual endpoint, (d_e) is the selected graph-edge
destination, and \(\sigma\) is the deployed consequence model's held-out
landing scale.  The existing unit-effective-count Bayesian update then updates
the edge's connection probability with this Bernoulli observation.

The previous Gaussian outcome was not the event represented by the edge
weight: it gave nonzero positive evidence outside the landing region.  A
repeatably immobile endpoint could therefore raise a dead command's corrected
weight.  Under R51, every repetition that misses the edge destination is a
failure and drives the posterior connection probability down.

No reward, hidden goal, posture threshold, movement threshold, or
environment-specific distance is used.  R48's models, graph, fields,
candidates, similarity kernel, and prior weight are unchanged.

PointMaze R11 does not import the AntMaze correction.  It has a separate
movement-and-endpoint-mobility implementation, so this revision does not alter
the frozen PointMaze code or results.
