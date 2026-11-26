# R30: R29-aligned factorized Ant consequence model

R30 changes only the reward-free training distribution of the factorized
consequence model.  It keeps the R8 spatial/posture factorization,
architecture, optimizer, ensemble size, mixture count, and training schedule.
The original reward-free rounds 0--2 are retained and the R29 commands chosen
by the frozen R28 acquisition planner are added.

The split is by whole R29 preparation task.  Sixteen of the eighty tasks are
selected deterministically and never used for fitting or model selection.
They measure the selected-command calibration failure that motivated R29.
The remaining R29 tasks and every original round-0--2 command form the
training set.  Neither hidden goals nor task rewards enter either set.

The matched diagnostic keeps the frozen R28 connection radius of
0.09150784158775013 m solely so old and aligned models are compared against
the exact same observed event.  It reports both ensemble-mean and pessimistic
connection probabilities, separately for all held-out commands and for the
predeclared diagnostic subset whose commanded distance is at least 6 m.

This stage is component evidence only.  It does not replace R28 artifacts or
rerun hidden-goal evaluation.  If held-out long-command calibration improves,
the deployed landing precision and all model-dependent road-map artifacts
must be rebuilt together before a task-time claim.

