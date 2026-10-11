# Unified VCG development comparison

This stages one V2.3-derived controller with two multiplier modes:

- `vcg`: both critic heads train, but lambda is fixed at zero.
- `vcg-handling-constraint`: the same controller learns lambda for B_N=20.

Everything else is shared, including model seed 14, the 200 training
EpisodeInstances, action/replay RNG namespaces, policy semantics, Hold,
discounts, and the opened 85xxx validation instances.  Only the terminal
episode-200 checkpoint is evaluated (48 rows per arm); no best-of-look
selection is used.  This is a development experiment and does not open 86xxx.

## Recorded outcome

The run completed. On the 48-row-per-arm development grid, the adaptive arm
changed rehandles/100 from 10.94 to 4.43 and MAE from 16.81 to 18.61. This
demonstrated a trade-off, but it did not establish an active `B_N=20`
constraint: both validation handling rates were already below 20. Treat this
as development history, not the final handling-cost result.

Fast preparation:

```bash
bash experiments/vcg_unified_lambda_pair_85k/run.sh prepare
```

Long runs (run these yourself, sequentially or as separate commands):

```bash
bash experiments/vcg_unified_lambda_pair_85k/run.sh run-vcg
bash experiments/vcg_unified_lambda_pair_85k/run.sh run-constrained
bash experiments/vcg_unified_lambda_pair_85k/run.sh compare
```

See [the unified VCG evidence index](../../docs/UNIFIED_VCG.md) for the later
frozen-weight selection and confirmation.
