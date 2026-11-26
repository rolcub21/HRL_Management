# R56: E549/R52 supplied-goal stalled-pair audit

R56 isolates routing and execution from hidden-goal acquisition on the two
consumed development tasks used by R53 (tasks 2 and 13).  The frozen E549 5M
executor, E549-matched R52 consequence model and graph artifacts, and R51
destination-arrival online correction are unchanged.  The only intervention
is that the exact task goal is supplied to the planner at time zero through
the standard Gaussian goal field at the deployed R52 landing scale.

Each task receives one physical episode of ten fixed 100-step commands from
the same reset row used by R53.  R51 online observations are updated after
each command within that episode.  The low-level executor and learned models
receive only the command selected by the planner; they never receive the task
goal.  The evaluator success radius is used only to report contact and is not
read by the planner, executor, or learned models.  No task reward is used for
learning and no task gradients occur.

This is a supplied-goal component audit on two outcome-selected, consumed
development tasks, not hidden-goal evidence or a performance estimate.  Its
interpretation is deliberately binary: success supports the claim that the
E549/R52 routing-and-execution stack can solve the task once the destination
is known; failure locates a bottleneck in that stack before acquisition is
considered.
