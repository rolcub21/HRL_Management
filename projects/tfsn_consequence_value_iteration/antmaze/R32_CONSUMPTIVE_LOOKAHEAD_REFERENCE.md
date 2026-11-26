# R32: two-command consumptive acquisition lookahead

R32 tests one consumed-development change over R31.  During hidden-goal
acquisition, the fixed-field R31 value is first used to rank the unchanged
live candidates.  For the four highest-ranked candidates, R32 consumes that
candidate's matched empirical passage from the residual-search field and
recomputes the existing bounded value-iteration tail on the consumed field.
The candidate is then rescored with its unchanged pessimistic connection
probability.  This makes the next imagined command see that the first passage
has already been searched; approximation begins after that next command.

The search-field update is

`A'(x) = A(x) * (1 - max_y kappa_sigma(x, y))`,

where `y` ranges over the empirical passage samples.  The deployed landing
precision supplies `sigma`; no new spatial threshold is introduced.

Lookahead depth 2 and beam width 4 are global computational settings intended
to remain identical in every domain.  The graph, R30 aligned consequence
model, executor, candidate set, command duration, physical horizon, return
planner, tasks, seeds, and carried-memory protocol are exactly R31.  Only the
already-consumed tasks 3 and 5 are used by the gate.  The confirmation panel
remains sealed, so the result is development evidence only.

