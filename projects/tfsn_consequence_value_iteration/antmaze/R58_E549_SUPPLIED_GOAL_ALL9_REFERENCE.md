# R58: E549/R52 supplied-goal original-nine panel

R58 evaluates the frozen E549 5M executor, executor-matched R52 consequence
model and graph, and R51 destination-arrival online correction on the original
nine consumed AntMaze development tasks (task IDs 0--8; frozen goal indices
2--10).  It expands the outcome-selected two-task R56 diagnostic into the
complete original development panel without accessing confirmation goals.

The exact task goal is supplied to the planner at time zero through the same
goal field used by R56.  Each task receives one physical episode of ten
commands, each held for the deployed 100-step duration.  The reset row, goal,
planner seed construction, evaluator-only success radius, executor, learned
models, and online correction are unchanged.  The low-level executor and
learned models receive only the selected command, never the supplied task
goal.  No task reward is used for learning and no task gradient occurs.

This is a supplied-goal routing-and-execution component panel on consumed
development tasks.  It is not hidden-goal or confirmation evidence.  With one
deterministic episode per task, it measures whether the frozen stack completes
each route under the benchmark budget; it does not estimate a per-goal success
probability.
