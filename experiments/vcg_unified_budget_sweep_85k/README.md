# Unified VCG developmental budget sweep

This experiment uses one shared developmental training seed to estimate four
operating points of the same unified VCG architecture:

- `lambda0`: lambda is fixed at zero;
- `budget-10`: learned lambda with B_N=10 rehandles/100 deliveries;
- `budget-8`: learned lambda with B_N=8;
- `budget-5`: learned lambda with B_N=5.

All arms train for 300 episodes with identical initialization, training
EpisodeInstances, behavior RNGs, replay RNG, policy semantics, and terminal
validation grid. Both critic heads train in every arm. Only lambda control and
the declared budget differ. The existing opened 85000..85011 panel is used;
86xxx remains unopened.

Prepare (short, no training):

```bash
bash experiments/vcg_unified_budget_sweep_85k/run.sh prepare
```

Run all four arms sequentially and then analyze:

```bash
bash experiments/vcg_unified_budget_sweep_85k/run.sh run-all
```

Or run one arm at a time:

```bash
bash experiments/vcg_unified_budget_sweep_85k/run.sh run-vcg
bash experiments/vcg_unified_budget_sweep_85k/run.sh run-budget 10
bash experiments/vcg_unified_budget_sweep_85k/run.sh run-budget 8
bash experiments/vcg_unified_budget_sweep_85k/run.sh run-budget 5
bash experiments/vcg_unified_budget_sweep_85k/run.sh compare
```

The report includes return, MAE, steps, rehandles, the one-seed point-estimate
MAE/rehandle frontier, and for each constrained arm terminal J_N, g=J_N-B_N,
lambda, lambda*g, and an inactive/active/unconverged classification. It reports
both the validation-panel diagnostics and the final training-block diagnostics.
This is a post-hoc developmental sweep, not a seed-stability or final-panel
claim.

## Recorded outcome

The sweep completed, but every positive-budget arm (`B_N=10,8,5`) was
classified as unconverged at the terminal point. In particular, positive
multipliers coexisted with substantially negative primal residuals. These runs
must not be described as converged active-budget solutions or as an identified
budget frontier. They motivated the direct fixed-weight diagnostics summarized
in [Unified VCG](../../docs/UNIFIED_VCG.md).
