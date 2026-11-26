# R67: area-weighted hub-routed hidden-goal development gate

R67 is a consumed-development component gate on AntMaze tasks 1, 2, and 8.
It keeps the frozen E549 5M executor, the E549-matched R52 consequence
preparation, and R51 destination-arrival correction.  It changes acquisition
only.

The acquisition candidate bank is the first twenty reward-free,
area-corrected hubs reported by R65.  Twenty is a computational limit for
this short component gate, not a claimed final hub count.  No distance,
reachability, or score cutoff is applied.

At the start of a physical episode and after reaching a hub, R67 scores every
hub by

```
posterior contact probability at hub
    * probability of reaching hub from the current full state
      within the commands remaining in the episode.
```

The posterior is the exact latent-radius posterior used by R61, except that
its spatial prior is corrected as in R65: each occupied landing-scale cell
has equal total prior mass.  The reach probability is computed by the frozen
R52 recorded-edge/bridge Bellman recursion.  The chosen hub is then supplied
to the unchanged R51 supplied-goal router until the realized trajectory comes
within the data-derived landing scale of that hub.  Every primitive state
traversed without contact updates the search posterior.  The next hub is then
selected from the updated posterior and actual current state.

The hidden task goal and the benchmark contact radius are evaluator-only.
They are never passed to the hub selector, supplied-goal router, consequence
model, or executor.  Task reward is used only as the hidden-goal observation
that switches acquisition to the unchanged return phase.  There are no task
gradient updates.

The report separates memory-update, hub-scoring, and routing wall time for
every decision.  A one-decision timing command performs no environment step.
The five-episode command is development evidence only and uses the already
consumed tasks 1, 2, and 8; it is not confirmation evidence.

