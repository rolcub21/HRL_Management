# R31: deploy the R29-aligned consequence model in the frozen R28 loop

R31 answers one consumed-development question: does the reward-free R29
alignment round improve the same two-goal hidden-goal gate when the aligned
R30 consequence model is actually deployed?

The task-time method remains R28.  Acquisition uses empirical passage gain
and the bounded consumption-aware first-contact backup.  Return uses the R22
finite-horizon planner plus exact positive online evidence candidates.  The
executor, fixed 100-step command contract, candidate budget, horizon, two
development goals, five carried episodes, and evaluator-only contact rule are
unchanged.

R31 changes only artifacts that depend on the deployed consequence model:

1. Landing precision is recomputed from R30's held-out R29 commands using the
   existing estimator: ensemble- and mixture-weighted within-mode radial
   spread.
2. The recorded-transition graph is rebuilt from the unchanged original
   reward-free command records using R30's state normalization and the new
   landing precision.  R29 held-out commands are not inserted into the graph.
3. Empirical-reach bridge destinations are rebuilt and rebound to R30.
4. Endpoint-mobility labels/model and bridge weights are recomputed with R30.
5. Empirical passage geometry is rebuilt at the deployed landing precision.

R28 and every earlier artifact remain immutable.  The R29 alignment data are
reward-free and contain no hidden evaluation goal or environment reward.  The
two hidden goals used by the R31 gate are already-consumed development goals;
the confirmation panel remains sealed.  Therefore R31 is development evidence
only, not confirmation evidence.

After the two-goal gate, the unchanged R31 runtime may also be evaluated on
all 15 already-consumed development goals to determine whether the observed
failure mode is general.  This remains development evidence only: no planner,
model, seed, episode count, or preparation artifact changes, and the sealed
confirmation panel is not accessed.

