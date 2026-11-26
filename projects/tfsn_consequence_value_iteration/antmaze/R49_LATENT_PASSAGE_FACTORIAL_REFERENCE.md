# AntMaze R49 latent-detection × passage factorial

R49 is a consumed-development causal screen, not a new frozen TF-SN result.
It leaves R48 unchanged and evaluates two independent acquisition factors on
AntMaze tasks 1 and 4: sigma-kernel versus latent-radius observation memory,
and nearest-recorded versus the shared learned passage predictor.

Candidate goals are exactly the unique xy projections of recorded roadmap
states. Their prior is uniform over that set. No grid is constructed. For the
latent-radius factor, `rho_min` is the deployed held-out landing sigma and
`rho_max` is the exact Euclidean diameter of this support. The prior density is
proportional to `1/rho`; each goal's radius interval and passage-contact
probability are integrated analytically.

Every actually observed primitive reward updates the joint posterior. A reset
state is updated only after the evaluator has computed its actual binary
reward. If all candidate mass reaches zero, the run writes a
`MODEL_FALSIFIED` artifact and raises. A uniform or other fallback is forbidden.

AntMaze trains passages through the domain-agnostic
`shared.learned_passage.PassageMLP`, `passage_loss`, and
`train_passage_ensemble`. Its adapter provides only the state vector, xy
projection, commands, and reward-free primitive xy trajectories. Architecture,
loss, target construction, grouped split, optimizer, schedule, and ensemble
reduction come from the one R12 manifest. PointMaze is not trained or evaluated
in this screen.

R39 immediate adaptive-greedy search and R46 online consequence semantics are
held fixed. The advance-family proposal is prefiltered only by recorded/model
reliability; using the old Gaussian search value there would leak the control
memory into the latent-radius arms. All final candidate scores use the selected
factorial memory and passage source.

This screen consumes previously exposed tasks and supports only component-level
causal diagnosis. It is not confirmation evidence and does not supersede the
R39 full-panel result or R48 rule-removal result.
