# PSLAP GA reproduction contract

The independent GA baselines are inspired by Park and Seo,
“Mathematical modeling and solving procedure of the planar storage location
assignment problem,” *Computers & Industrial Engineering* 57(3), 1062–1071,
2009, <https://doi.org/10.1016/j.cie.2009.04.010>.

## Verified paper-level semantics

- The planning horizon is discretized into periods and the inbound/outbound
  schedule is known.
- Outbound objects are released and inbound objects are stored in each period.
- The primary objective is the number of obstructive object moves under planar
  movement.
- The paper uses a GA because the mathematical model is NP-hard.

No official source repository was located during the implementation audit.
Available public metadata and indexed text do not expose every evolutionary
operator or parameter needed for byte-for-byte reproduction.

## Explicit reproduction choices

### Offline full-schedule reference

- A chromosome assigns one unique environment storage cell to each episode
  block. This stronger uniqueness rule guarantees online feasibility but does
  not claim to reproduce every reuse constraint in the paper's mathematical
  model.
- A block's schedule interval is
  `[arrival_step, arrival_step + storage_steps_needed]`.
- Outbound events precede inbound events at equal timestamps.
- Fitness lexicographically minimizes infeasible events, obstructive moves,
  and route length. It never uses environment return or timing reward.
- The environment's actual pickup cell, walls, storage cells, and exit gate are
  used. This adapts the paper's simplified one-side rectangular yard to the
  experimental environment.
- The default search uses population 30, 30 generations, two elites,
  tournament size 3, one-point crossover with uniqueness repair, crossover
  probability 0.8, mutation probability 0.1, and seed 0.
- The best assignment is executed by `OnlinePSLAPPolicy` with the exact same
  travel-time-aware retrieval dispatcher used by `dynamic_pslap`.

This implementation is named `pslap_ga_2009_offline`. It reads every block's
arrival and required storage duration before execution and must be reported as
an information-advantaged reference. The historical `pslap_ga_2009` CLI name
is an alias for this method.

### Rolling online baseline

`pslap_ga_2009_rolling` reoptimizes at every inbound assignment epoch. Its
optimizer receives only an immutable snapshot of currently stored inventory
and blocks whose arrival time has already been reached. Existing placements
remain fixed; the chromosome assigns unique currently free cells to the known
pending blocks. Fitness lexicographically minimizes infeasible known placements,
predicted minimum-obstruction egress over known inventory, and route length.

The rolling fitness is an explicit online adaptation, not an algorithm claimed
by the 2009 paper. Its RNG is local to the search, and the decision-index offset
makes repeated reoptimizations deterministic without coupling them to episode
generation. Tests verify that changing unarrived blocks' durations cannot alter
the current rolling decision.

Both GA fitness functions are planning surrogates, not observed online metrics.
Action-time storage and proactive dispatch can change which blocks overlap in
the live yard. `compare_pslap_baselines.py` therefore reports both
`ga_predicted_obstructive_moves` and actual `obstructive_moves`, plus assignment
fallbacks. Claims about operations must use the actual metric; disagreement
between the two is itself a model-mapping diagnostic.

The historical `example/genetic_algorithm.py` is retained for provenance but
is not this baseline. It evaluates total reward while storing and immediately
retrieving each block, which removes the overlapping occupancy responsible for
the PSLAP objective.
