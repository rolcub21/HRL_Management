# Continuous TF-SN reference implementation: PointMaze

PointMaze is the reference implementation for continuous-domain TF-SN revision
`tfsn_continuous_reference_r11`. AntMaze must adopt the same decision rules;
domain adapters may provide only state/action schemas, reward-free data,
executor checkpoints, and the benchmark episode length.

The cross-domain definition is recorded in
[`../METHOD_REFERENCE.md`](../METHOD_REFERENCE.md), and the immutable PointMaze
freeze record is [`R11_FREEZE.json`](R11_FREEZE.json).

All fixed method and computational settings are loaded from
`../shared/method_manifest.json`. The benchmark success radius is evaluator
only. It is never read by the planner, consequence model, reach construction,
task field, or online correction.

The executor receives an xy command for the shared command duration
`tau = 100`. The finite horizon is always the benchmark episode length divided
by `tau`; PointMaze therefore derives 20 commands from 2,000 primitive steps
rather than storing a PointMaze-specific horizon.

### Detection-radius decision

R11 removes the detection/contact radius from the method. The PointMaze
benchmark's 0.4 m success radius is read only by the evaluator to decide
whether contact occurred. It is not task knowledge available to acquisition,
return, the graph, the consequence model, reach, or online correction.

Acquisition instead clears its Gaussian search field at `sigma`, the deployed
consequence model's held-out landing scale. This is a data-derived scale, not
an environment-specific radius. AntMaze must use the same rule. A future
experiment that supplies a detection model would be a declared method change,
not an implementation detail of R11.

## Data-derived scales and candidates

The landing scale `sigma` is the held-out mixture-weighted within-mode spatial
spread of the deployed consequence ensemble. Search clearing, reward-rate
localization, alias association, and the stall kernel use this learned scale.

Reach is an empirical measure over realized reward-free displacements. Every
record is weighted continuously by

`(1 - kappa_sigma(command, start)) * (1 - kappa_sigma(endpoint, start))`.

There is no distance threshold or sigma multiplier. Near-zero requests and
near-zero outcomes remain represented but carry vanishing reach mass.

The road map retains every recorded command-boundary state and transition.
Recorded-edge reliability is the shared kernel posterior of realized-delta
repeatability. Its neighbor cap and one conservative unobserved
pseudo-observation are declared once in the shared manifest. The learned model
does not receive an extra vote in this estimator; its predictions enter through
the bridge consequence weights.

## Consequence and mobility

The factorized consequence ensemble predicts landing position and remaining
state. Stall probability is `E[kappa_sigma(endpoint, start)]`.

Mobility labels are evaluated over the continuously weighted empirical reach
candidates, with no fixed movement radius. Bridge values integrate every joint
spatial/posture mixture component mean using its joint probability. Endpoint
sampling is not used.

After every task-time command the planner records the continuous movement
outcome. The following command supplies evidence about whether the preceding
endpoint remained mobile. Similar reward-free observations update command
viability and endpoint mobility with the shared one-observation prior. Memory
persists across physical resets within a task and resets between tasks.

## Planning

Acquisition clears the Gaussian search field at the learned landing scale.
Return reads the Gaussian reward-rate field at that scale. Acquisition uses the
bounded consumable-field backup; return uses the additive persistent-field
backup. Recorded transitions and model bridges use the same reliability
semantics.

The live decision uses continuous alias/advance weighting. Acquisition applies
depth-two consumptive lookahead. The top-four prefilter is retained solely as a
declared global computational approximation; changing or removing it changes
the shared manifest, not a domain adapter.

The consequence ensemble is reduced pessimistically. The hidden goal
coordinate, evaluator success radius, environment geometry, and task reward
are not planner inputs.

## Development and confirmation

The nine exposed spatial tasks are development-only. The E373 confirmation
panel remains sealed until this reference implementation is accepted and the
same runtime has been ported to AntMaze.

R11 acquires seven of nine development goals and returns successfully in every
post-contact episode. The two stable known failures are
`spatial_diagnostic_native_goal_r01_c06` and
`spatial_diagnostic_native_goal_r06_c06`. Neither was contacted in eight
episodes in R8 or R11. They remain explicit limitations of the frozen version,
and are the natural later diagnostic for the value of supplying a detection
model. They must not be silently removed from the development report.

## Supplied-goal control

The separate supplied-goal protocol installs a Gaussian bump at the provided
goal at episode start and runs the unchanged R11 return planner. Acquisition is
disabled, and each episode resets online task memory. It uses the exact E373
26-goal, five-environment-seed, 20-episode matrix so its episodes pair directly
with the existing MEGA evaluation. This is descriptive rather than
equal-budget: current R11 uses 4.1M reward-free transitions including executor
training, while the available MEGA actor uses 2.9M.
