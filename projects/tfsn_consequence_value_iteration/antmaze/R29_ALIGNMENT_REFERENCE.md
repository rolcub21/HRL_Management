# R29: reward-free R28 consequence alignment

R29 collects one additional reward-free preparation round with the frozen R28
acquisition planner.  Its purpose is narrow: expose the factorized consequence
model to the long commands selected by R28, after the R28 gate showed that all
three ensemble members were optimistic on those commands.

The collection has no hidden goal, reads no task reward, and performs no task
gradient updates.  Each preparation task starts with empty task memory.  That
memory persists across physical resets within the task, matching the carried
hidden-goal evaluation.  Physical state resets every ten fixed-duration
commands; every command is held for exactly 100 primitive steps.

R28 itself is frozen during collection.  In particular, the empirical graph,
reach-based bridges, factorized R8 consequence ensemble, mobility model,
empirical passage geometry, landing precision, and bounded consumption-aware
search backup are unchanged.  The resulting command transitions are data for
a later consequence-model refit; they are not a result for the hidden-goal
task and do not access the sealed confirmation panel.

