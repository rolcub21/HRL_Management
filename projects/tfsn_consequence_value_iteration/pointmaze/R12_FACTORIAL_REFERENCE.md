# PointMaze R12 latent-detection x passage factorial

This is an isolated consumed-development screen. It does not modify or
supersede frozen PointMaze R11, and it is not confirmation evidence.

The task set is the two historically exposed R11 corner failures
`spatial_diagnostic_native_goal_r01_c06` and
`spatial_diagnostic_native_goal_r06_c06`, with three episodes per goal. The
four arms cross sigma-kernel versus latent-radius observation memory with
nearest-recorded versus shared learned passage prediction.

Candidate goals are the unique xy projections of recorded R11 roadmap states,
with a uniform prior. `rho_min` is the deployed held-out landing sigma and
`rho_max` is the exact support diameter. Posterior exhaustion is fatal and has
no uniform fallback.

The learned passage estimator is the same
`shared.learned_passage.PassageMLP`, loss, target rule, optimizer, grouped
split, ensemble reduction, and early-stopping rule used by the AntMaze screen.
Only the state adapter and reward-free PointMaze trajectories differ.

Frozen R11's global finite-horizon values cannot be reused inside this
factorial: they were precomputed with the sigma observation field and matched
passages, which would leak both control factors into the changed arms.
Acquisition therefore uses one declared local rule in every arm: immediate
candidate scoring followed by depth-two consumptive lookahead over the top
four candidates. Candidate proposal is observation-neutral reliability only.
This means the control arm is not a bitwise R11 reproduction. After contact,
the frozen R11 return planner is used unchanged.
