# R77: task-local recorded-link growth for return

R77 is a consumed-development replay of AntMaze task 6.  It starts from the
frozen R75 acquisition planner and changes return only.

Every executed command supplies a reward-free transition `(start state,
command, realized endpoint state)`.  R77 retains those exact states and the
command as a task-local recorded link.  For the existing R75 value iteration,
the link endpoints are associated continuously with the nearest recorded-map
nodes by the deployed landing kernel.  There is no distance cutoff.  The
resulting edge reliability is the product of the source and endpoint
associations.  The task-local links are additional options: no offline node,
edge, bridge, or value option is removed.

During return, R77 uses the same node field, recorded edges, model bridges,
max reduction, additive continuation, and finite horizon as R75.  Task-local
recorded links enter that same Bellman backup.  They are also live command
candidates and are evaluated from the current full state by the deployed
consequence model and R51/R71 online link correction.

Return-route outcomes provide a second, route-level reliability posterior.
For a source state, observed evidence target, and exact remaining horizon,
similarity is the R52 empirical-reach Gaussian on both source and target.  The
prior reliability is one observation of success.  This reliability multiplies
the otherwise unchanged R75 value at that state and horizon inside each
Bellman sweep.  With no route evidence its value is exactly one, so R75 is
recovered exactly.  There is no similarity cutoff.

The hidden goal is used only by the evaluator to emit task contact.  It is not
visible to the planner, executor, consequence model, map-growth mechanism, or
route posterior.  The return target is the first coordinate at which positive
task evidence was actually observed.

Before evaluation, R77 must establish:

1. R75 values are exactly preserved with no task-local links or route evidence.
2. A failed route lowers similar return values while leaving alternatives
   positive.
3. Adding task 6's already observed acquisition transitions makes the reset to
   observed-contact reachability positive in the exact graph audit.
4. Acquisition replay remains exactly identical to R75.
