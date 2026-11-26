# PointMaze R9 exact detection-footprint development test

R9 changes one acquisition input relative to the frozen R8 development
planner.  It interprets the benchmark's 0.4 m reward-contact radius as the
known negative-observation model for hidden-goal search.  Before contact, the
residual search posterior is therefore the binary field

`A_t(x) = 1[min_j ||x - x_j|| > 0.4 m]`.

The goal coordinate remains hidden.  The executor, consequence ensemble,
road map, movement-conditioned reach bank, bridge weights, finite horizon,
continuous advance weighting, return field, task panel, starts, and seeds are
unchanged from R8.  Return continues to use the deployed consequence model's
landing precision.

This is a consumed-development method test, not confirmation evidence.  It is
an explicit deviation from the current PointMaze protocol, which labels the
0.4 m radius evaluator-only.  The report must therefore state that the radius
was read by acquisition and list it as an oracle/task-contract input until the
canonical method reference is deliberately revised.

Each episode reports both the maze cells physically visited for the first time
and the free-cell centers newly ruled out by the exact detection posterior.
These diagnostics test whether later carried episodes move into regions that
earlier episodes did not cover.
