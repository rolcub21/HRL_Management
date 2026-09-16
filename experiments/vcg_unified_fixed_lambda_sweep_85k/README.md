# Unified VCG fixed-lambda development sweep

This experiment maps the handling/timing trade-off directly without relying
on dual convergence. It reuses the completed 300-episode lambda-zero arm and
trains four new arms with fixed lambda values 0.05, 0.10, 0.20, and 0.30.
The grid was chosen after observing the dual-budget development sweep and is
therefore explicitly post-hoc and development-only.

Every new arm uses lambda=0 for the shared 20-episode critic warm-up, then its
declared lambda for episodes 21..300 and terminal evaluation. Model seed,
training EpisodeInstances, behavior RNGs, replay RNG, architecture, policy,
and the opened 85000..85011 validation grid are unchanged. Both critic heads
continue to train. No budget, complementarity, or KKT claim is made.

Prepare (short):

```bash
bash experiments/vcg_unified_fixed_lambda_sweep_85k/run.sh prepare
```

Run the four new arms sequentially and compare all five operating points:

```bash
bash experiments/vcg_unified_fixed_lambda_sweep_85k/run.sh run-all
```

Or run arms individually:

```bash
bash experiments/vcg_unified_fixed_lambda_sweep_85k/run.sh run-lambda 0.05
bash experiments/vcg_unified_fixed_lambda_sweep_85k/run.sh run-lambda 0.10
bash experiments/vcg_unified_fixed_lambda_sweep_85k/run.sh run-lambda 0.20
bash experiments/vcg_unified_fixed_lambda_sweep_85k/run.sh run-lambda 0.30
bash experiments/vcg_unified_fixed_lambda_sweep_85k/run.sh compare
```

The output is `fixed-lambda-comparison.json`. It reports all five points,
paired differences from lambda=0, the one-seed MAE/rehandle point-estimate
frontier, and whether rehandles decrease monotonically with lambda. This is a
development-only diagnostic; 86xxx remains untouched.

## Recorded outcome

The retrained operating points were non-monotone in the handling weight. For
example, rehandles/100 were 1.56 at `lambda=0`, 18.49 at `0.05`, 10.16 at
`0.20`, and 8.33 at `0.30`. Because each arm followed a separate optimization
path, this sweep did not isolate policy scalarization. It is evidence of
training-path sensitivity, not the final trade-off curve. The next experiment
therefore held weights fixed.
