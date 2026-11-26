# R69: E549 dynamic hubs with a longer evaluation budget

R69 is a consumed-development budget diagnostic on AntMaze tasks 1, 2, and
8.  It changes only the physical evaluation budget relative to R68:

- 20 fixed-duration commands per physical episode;
- 100 primitive steps per command, hence 2,000 primitive steps per episode;
- 10 physical episodes per task.

The executor is the frozen E549-5M checkpoint.  The consequence model,
recorded graph, mobility model, bridges, empirical passages, landing scale,
area-weighted latent goal prior, full data-derived dynamic hub bank, supplied
goal router, and R51 online destination-arrival correction are the R52/R68
components without retraining or task-time gradients.

Because the physical episode contains 20 commands, R69 recomputes the exact
finite-horizon hub-reachability layers through horizon 20.  It must not reuse
R68's horizon-10 reachability tensor as if it described the longer budget.

This is not a benchmark claim.  Tasks 1, 2, and 8 are already consumed
development tasks.  The purpose is to determine whether R68's remaining
failures are primarily caused by the episode count or the within-episode
command budget.  If all three goals are acquired, subsequent diagnostics may
reduce these two budgets separately while leaving the method fixed.

The hidden goal is unavailable to the planner, executor, consequence model,
hub scorer, and router.  Task reward is used only to update the latent
detection posterior and switch from acquisition to return.  The benchmark
success radius remains evaluator-only.

