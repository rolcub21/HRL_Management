# Unified VCG paired seed stability

This development experiment repeats the unified VCG comparison with fresh
model seeds 15, 16, and 17. Seed 14 remains development-only and is excluded
from the primary three-seed aggregate.

For each model seed, both arms share initial weights, 200 training
EpisodeInstances, behavior-action RNGs, replay RNG, and the terminal 12x4
validation grid. The only treatment difference remains lambda control:

- `vcg`: lambda is fixed at zero while both critics are trained.
- `vcg-handling-constraint`: lambda is updated against the B=20 budget.

Run all six training jobs sequentially and then aggregate:

```bash
cd /path/to/HRL_Management
bash experiments/vcg_unified_lambda_pair_seed_stability_85k/run.sh run-all
```

Reissuing `run-all` skips arms that have a complete training summary. It
stops on a partial arm directory rather than overwriting it.

To resume manually after a stopped job, run only the missing arm and then the
comparison, for example:

```bash
bash experiments/vcg_unified_lambda_pair_seed_stability_85k/run.sh run-constrained 16
bash experiments/vcg_unified_lambda_pair_seed_stability_85k/run.sh compare-seed 16
bash experiments/vcg_unified_lambda_pair_seed_stability_85k/run.sh compare-all
```

Outputs are written under
`results/vcg-unified-lambda-pair-seed-stability-85k/`. The experiment reuses
the already-opened 85000..85011 development panel and does not open a new
final panel.

## Recorded outcome

The original stability rule passed: the adaptive arm reduced aggregate
rehandles/100 from 9.29 to 7.99, with reductions for two of three seeds. MAE
increased from 19.87 to 21.59, and seed 16 had a small rehandle increase. The
adaptive dual paths are therefore developmental. The completed `lambda=0`
terminal checkpoints for seeds 15--17 were subsequently reused, without
retraining, for the frozen-weight experiment chain documented in
[Unified VCG](../../docs/UNIFIED_VCG.md).
