# Continuous TF-SN cross-domain method reference

The frozen reference revision is `tfsn_continuous_reference_r11`. PointMaze is
the executable reference; AntMaze must adopt the same decision principles and
global computational settings. Domain adapters may supply only the state and
action schema, reward-free data, executor checkpoint, and benchmark episode
length.

The complete fixed-setting inventory is
[`shared/method_manifest.json`](shared/method_manifest.json). Every entry is
classified as data-derived, global computational, global decision principle,
or task knowledge. Implementations must import global settings from that file;
they may not introduce a domain-local distance, threshold, mask, candidate
budget, horizon, or cap.

## Detection model decision

The frozen method does **not** receive a detection radius. A benchmark success
radius is evaluator-only: it reports contact but is forbidden to the planner,
executor, consequence model, graph, reach construction, task fields, and
online correction.

Search clearing and return localization use the deployed consequence model's
held-out landing scale `sigma`. This is learned separately in each domain from
reward-free data under the same estimator. Thus PointMaze's 0.4 m and
AntMaze's 0.8 m contact radii do not enter either planner.

Supplying a known observation/detection model is a legitimate future ablation,
but it is a different declared method revision. It must not be folded into an
R11 domain adapter.

## Shared realization

- A command lasts the globally declared 100 primitive steps. The planning
  horizon is the benchmark episode length divided by that duration.
- Reach is the empirical realized-displacement measure, continuously weighted
  by requested and realized advance at the learned landing scale.
- Recorded transitions are retained. Model bridges integrate all spatial and
  remaining-state mixture outcomes, and ensemble predictions are reduced
  pessimistically.
- Acquisition uses a bounded consumable-field backup and depth-two consumptive
  lookahead. Return uses the persistent reward-rate field.
- Online consequence evidence persists across physical resets within one task
  and resets between tasks.

The detailed executable realization is
[`pointmaze/METHOD_REFERENCE.md`](pointmaze/METHOD_REFERENCE.md). Its frozen
artifacts, evidence, manifest audit, and known limitations are bound in
[`pointmaze/R11_FREEZE.json`](pointmaze/R11_FREEZE.json).

## Evidence boundary

PointMaze R11 is frozen from the exposed nine-goal development panel. The
confirmation panel remains sealed for this revision. Its two known failures,
`spatial_diagnostic_native_goal_r01_c06` and
`spatial_diagnostic_native_goal_r06_c06`, are part of the freeze rather than
discarded as noise.
