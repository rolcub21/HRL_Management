# R53: E549/R51 two-task AntMaze audit

R53 repeats the frozen R51 tasks 2 and 13 audit after replacing the Farama
executor and all executor-dependent preparation with frozen E549 at five
million training steps.  The destination-arrival online correction is
unchanged: after an executed command, the Bernoulli observation is whether the
actual endpoint fell within the R52 landing scale of the selected graph
destination.

The tasks, reset rows, task seeds, five carried episodes, fixed 100-step
commands, hidden-goal evaluation, and selection rule are the same as the prior
Farama/R51 audit.  The two tasks were selected because they tied for the most
sub-0.03 m moves in the earlier R48 result.  Consequently this is a consumed
component audit, not a performance estimate.

The intended causal contrast is the deployed walker system: R51/Farama uses
Farama-dependent preparation, while R53/E549 uses only R52 E549 transitions
for its consequence model, graph, mobility, bridges, and passages.  No task
reward, hidden goal, evaluator radius, route, or oracle command is available to
the planner, executor, or learned models.
