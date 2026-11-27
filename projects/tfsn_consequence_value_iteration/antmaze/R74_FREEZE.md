# Frozen AntMaze method: R74

The canonical cross-domain equations, setting provenance, and rebuild
procedure are in
[`../FROZEN_R74_METHOD_SPECIFICATION.md`](../FROZEN_R74_METHOD_SPECIFICATION.md).
The machine-readable inventory is
[`../shared/frozen_r74_settings.json`](../shared/frozen_r74_settings.json).

The frozen AntMaze TF-SN method is **R74**, method revision
`antmaze_e549_5m_reach_scale_hub_route_correction_r74`.

R75 is not a planner revision.  It runs the unchanged R74 planner with twenty
episodes per task instead of ten.  Its consumed nine-goal development panel is
therefore the budget curve reported for the frozen method:

- acquisition: 0/9 by episode 1, 3/9 by episode 5, 5/9 by episode 10, and 9/9
  by episode 20;
- pooled conditional return after contact: 0.8715596330;
- mean end-to-end success across all twenty episodes: 0.5777777778;
- online destination-arrival rate: 0.9017006821.

These are consumed development results, not confirmation evidence.

## Known task-6 failure

Task 6, goal `(10, 6)`, was acquired in episode 6 (zero-indexed contact episode
5).  It then failed all fourteen return episodes, so its conditional return was
0/14.  Every return episode remained below the wall near `(10, -0.75)`; across
280 return commands the largest realized y-coordinate was `-0.5239`.

The failure is a measured consequence-model/routing error at that wall:

- the return planner issued fifteen bridge commands from below the wall to
  destinations with y greater than 4 m;
- those selected bridges had offline connection estimates from `0.0067` to
  `0.4787` and online-corrected estimates from `0.0061` to `0.4441`;
- evidence against one crossing caused a neighboring crossing to replace it;
- the R78 preflight showed that a principled multiplicative route posterior
  preserved R75's additive values but still selected a neighboring
  wall-crossing shortcut after two failed episodes;
- the supplied-goal R58 panel independently tested the exact `(10, 6)` goal and
  also failed, finishing near `(10.30, -0.73)` with a minimum goal distance of
  6.67 m.

Thus task 6 is reported as **acquired, then return blocked by a model error at
one wall**.  It is not reported as a hidden-goal acquisition failure.  R76,
R77, and R78 are rejected component diagnostics and are not part of the frozen
method.

## Method boundary

The frozen deployment uses the E549 5M chained AntMaze executor and R52
reward-free preparation.  R74's route correction uses the empirical
movement-weighted reach covariance derived from R52.  It has no route
similarity cutoff, wall input, maze-coordinate rule, planner access to the
benchmark success radius, task gradient update, or hidden-goal input.

Future planner-selected reward-free alignment may rebuild the preparation and
remove the false wall-crossing family, but that would be a new method revision.
