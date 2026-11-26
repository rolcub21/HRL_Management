# R76: task-6 return-route correction replay

R76 is a focused, consumed-development component replay of task 6 from R75.
It retains the E549-5M executor, R52 preparation, R74 acquisition planner,
twenty-command physical episodes, twenty episodes, R51 destination-arrival
link correction, and the R74 empirical reach covariance.

The only behavioral change is post-contact.  A completed return episode is an
observed route outcome.  For every command-boundary state on that attempted
route, R76 records whether the remaining policy reached the first observed
positive-evidence location before the physical episode ended.  The outcome is
therefore defined by observed task evidence, not by the hidden goal coordinate.

At the next return decision, exact finite-horizon reachability to that observed
evidence is recomputed from the R51-corrected bridge weights.  Route outcomes
correct this state-to-evidence reachability with the same one-observation prior
and the same continuous two-sided Mahalanobis kernel as R74:

```text
exp(-0.5 * (
    (source - observed_source)^T Sigma_reach^-1
        (source - observed_source)
  + (evidence - observed_evidence)^T Sigma_reach^-1
        (evidence - observed_evidence)
))
```

`Sigma_reach` is derived from frozen R52 preparation exactly as in R74.  Route
evidence transfers only between equal remaining command horizons.  There is no
chosen radius, cutoff, wall test, maze geometry, or task-specific exclusion.
Acquisition never reads the return correction.

The replay must reproduce R75 task 6 acquisition exactly through first contact.
Its question is only whether route-level correction converts the fourteen
post-contact failures.  It is not confirmation evidence and cannot support an
all-task claim.
