# R74: empirical-reach route correction

R74 retains R72's frozen E549-5M executor, R52 reward-free preparation,
latent-radius acquisition memory, dynamic area-weighted hubs, exact global hub
argmax, sigma-generalized link correction, 20-command physical episodes, and
ten episodes per consumed development task.

R72 corrected a planned hub-route outcome with a Gaussian kernel at landing
precision.  Its failures recur in spatial pockets that are wider than landing
precision: after one hub route fails, a nearby start or destination receives
almost no evidence and is treated as a new option.  R74 changes only this
route-level evidence geometry.

The route kernel is

```text
exp(-0.5 * (
    (source - observed_source)^T Sigma_reach^-1
        (source - observed_source)
  + (hub - observed_hub)^T Sigma_reach^-1
        (hub - observed_hub)
))
```

`Sigma_reach` is the weighted covariance of realized one-command displacement
in the frozen R52 preparation.  It uses R52's already declared continuous
movement weight

```text
(1 - kappa_sigma(command, start))
* (1 - kappa_sigma(endpoint, start)).
```

Thus both route sources and route destinations generalize at the executor's
empirical one-command reach.  There is one data-derived covariance, no chosen
radius, multiplier, cutoff, neighbour count, wall test, or maze-specific
input.  Link-level destination-arrival evidence remains at landing precision,
because it predicts one landing rather than a multi-command route.

The exact batched latent scorer is inherited from R72 under its saved CUDA
equivalence certificate.  R74 changes the route-evidence inputs shared by the
scalar and batched Bellman realizations, not the Bellman recursion, candidate
set, branch-and-bound rule, or argmax.

The nine goals are consumed development evidence.  The matched run asks
whether a failed route from a pocket suppresses the neighboring route family,
restores late-episode coverage, and preserves R72's successes on tasks 2, 3,
5, and 6.
