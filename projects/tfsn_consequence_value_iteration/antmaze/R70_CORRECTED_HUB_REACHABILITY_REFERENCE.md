# R70: corrected hub-reachability replay

R70 is a focused mechanism replay of consumed AntMaze development task 8.
It retains R69's E549-5M executor, R52 preparation, dynamic hub bank, R51
destination-arrival correction, 20-command physical episodes, ten episodes,
and evaluation seed. It changes hub reachability so the hub scorer and router
use the same online evidence. It is a development experiment, not benchmark
evidence.

The replay logs the full 29-dimensional state at every command boundary.  At
each hub selection it then recomputes the selected hub's finite-horizon
reachability after applying all preceding R51 observations to every modeled
bridge.  For bridge prior weight `w`, online kernel support `n`, online
arrival mass `y`, and the globally declared prior weight `a`, the corrected
weight is

```text
(a w + y) / (a + n).
```

This is exactly the pessimistic R51 member correction for a stored bridge:
the Bayesian update is a positive affine function of each member's connection
probability, so taking the least optimistic member commutes with the update.
Recorded transitions remain unchanged because they are witnessed preparation
edges, not consequence-model bridges.

R70 selects the hub maximizing posterior coverage times corrected
reachability over the complete data-derived hub bank. It does this exactly by
branch-and-bound. Posterior coverage is an upper bound on the score because
reachability is at most one. Hubs are evaluated in decreasing coverage order;
evaluation stops only when every remaining hub's coverage is no larger than
the best exact score already found. Thus there is no hub count, score cutoff,
or environment-specific threshold.

Its decisive questions are:

1. Does hub 2415's corrected reachability fall after failed attempts, and does
   the planner leave the pocket for another region?
2. If it remains trapped, do later failures move among nearby model bridges
   whose R51 kernel support remains small?

Maze geometry labels whether a selected modeled bridge's straight segment
intersects a wall.  Geometry is diagnostic-only and is unavailable to the
planner, executor, consequence model, hub scorer, and router.  The full-state
log is retained for a later reward-free alignment round if failures must be
fed back into the consequence model.

