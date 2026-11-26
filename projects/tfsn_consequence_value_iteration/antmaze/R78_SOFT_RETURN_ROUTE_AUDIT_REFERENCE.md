# R78: soft-return route-correction preflight

R78 is an offline, consumed-development audit of AntMaze task 6.  It does not
run the environment and it does not define an adopted planner revision.

The audit tests route-level evidence on top of R75's soft return values.  R75
uses an additive return field, so its values may exceed one.  They are not
Bernoulli probabilities and are never clipped into `[0, 1]`.  Instead, a
route-outcome posterior supplies a multiplicative reliability:

```text
R_r(x) = (1 + weighted route arrivals) / (1 + weighted route attempts)
V_corrected_r(x) = R_r(x) V_R75_r(x)
```

The prior is the shared one-observation prior.  Similarity is the continuous
Gaussian Mahalanobis kernel over the route-prefix source and observed evidence
target, using R52's empirical reach covariance on both sides.  Evidence only
transfers between equal remaining-command horizons.  There is no cutoff,
wall rule, coordinate rule, or task-specific exclusion.

The structural task-6 preflight reconstructs the soft observed-evidence field
at the deployed landing scale and rolls out the greedy graph policy before
route evidence, after one failed return episode, and after two.  Maze geometry
is used only after the rollout to label whether a selected segment intersects
a wall; it is not available to the value computation or route correction.

Four gates are reported:

1. With no route evidence, corrected values are bitwise identical to R75.
2. Failed evidence lowers similar values smoothly without clipping them.
3. Alternative routes remain positive.
4. After one or two task-6 failures, the best graph rollout stops selecting a
   wall-crossing shortcut and takes an available non-crossing route.

An environment replay is permitted only if all four pass.  A failed fourth
gate means route-level reliability merely moves the optimizer to another
member of the same false-shortcut family; the next remedy belongs in aligned
consequence data or in a principled model-level generalization, not in a maze
coordinate rule.
