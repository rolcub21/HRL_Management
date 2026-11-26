# R68: dynamic area-weighted hub refinement

R68 is the second half of the R67 hub-routing test.  It retains the E549 5M
executor, R52 consequence preparation, R51 destination-arrival correction,
area-weighted latent-radius posterior, supplied-goal router, and consumed
development tasks 1, 2, and 8.

R67 deployed only the first twenty hubs ranked under the starting posterior.
That fixed regional repetition but left no finer target after the Ant reached
the correct region.  R68 removes that declared diagnostic restriction.  Its
candidate set is every reward-free landing-scale representative in the R52
road map.  The candidate set is data-derived; it has no requested target,
distance cutoff, or environment-specific count.

For every candidate hub, R68 precomputes the exact finite-horizon reach
probability from every recorded graph node for every remaining command count.
This is a one-time reward-free computational artifact.  It is stored as a
memory-mapped float32 table and built resumably.  At task time, every time a
hub is reached or a physical episode resets, R68 recomputes each candidate's
posterior contact probability and selects

```
argmax_h posterior_contact_probability(h)
         * reach_probability(current_full_state, h, commands_remaining).
```

Scoring is exact over the whole data-derived candidate set.  Chunking changes
only memory use and is declared computational-only.  There is no shortlist,
reachability cutoff, local radius, or fallback action.  Consequently, finer
hubs can become optimal after the route to a coarse hub consumes posterior
mass.

The hidden task goal and benchmark contact radius remain evaluator-only.
Task reward is used only to update the hidden-goal observation and switch to
the unchanged return phase.  R68 is a consumed-development component gate,
not confirmation evidence.

