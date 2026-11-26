# R73: fixed-sigma control for R72 hub-route correction

R73 is the matched fixed-sigma arm for R72.  It keeps the frozen E549-5M
executor, R52 consequence artifacts, area-weighted dynamic hub bank, R71
sigma-generalized link correction, R72 hub-route correction, evaluation seed,
and the full 20-command by ten-episode development budget.

The only method difference is acquisition observation memory.  R72 integrates
an unknown detection radius `rho` under a log-uniform prior.  R73 instead uses
the original fixed-scale soft memory

```text
A_t(g) = 1 - max_{x in traversed path} kappa_sigma(g, x),
```

where `sigma` is the held-out landing scale of the deployed R52 consequence
model.  The finite goal support retains R65's equal total prior mass per
occupied sigma cell.  Normalizing the residual mass after each passage does
not change hub rankings; it only represents the same soft memory as a
posterior over the finite support.

No evaluator success radius, hidden goal, maze geometry, distance cutoff, or
new scale is available to the planner.  Return behavior is identical to R72.

R72 and R73 therefore isolate latent-radius inference versus fixed-sigma
memory after the route-level failure correction is held constant.  The nine
goals remain consumed development evidence.
