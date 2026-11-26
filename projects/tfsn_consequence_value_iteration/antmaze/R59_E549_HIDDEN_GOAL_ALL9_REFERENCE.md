# R59: E549/R52/R51 hidden-goal original-nine panel

R59 evaluates the frozen E549 5M executor, executor-matched R52 consequence
model and graph, and R51 destination-arrival online correction on the complete
original nine-goal consumed AntMaze development panel (task IDs 0--8; frozen
goal indices 2--10).  It expands the outcome-selected two-task R53 diagnostic
without changing the deployed method or accessing confirmation goals.

Each task receives five carried episodes.  Every physical episode starts from
its frozen reset row and contains ten fixed 100-step commands.  Search and
online consequence memory persist across the five episodes of a task and are
cleared between tasks.  Before first contact, the goal coordinate is visible
only to the evaluator.  After contact, the unchanged return field is built
from observed evidence.  The evaluator-only success radius is never read by
the planner, executor, or learned models.  No task reward is used for learning
and no task gradient occurs.

Report acquisition by episode one and episode five, pooled conditional return
after first contact, mean end-to-end episode success, coverage diagnostics,
and selected-destination arrival.  This is consumed-development component
evidence, not confirmation evidence.
