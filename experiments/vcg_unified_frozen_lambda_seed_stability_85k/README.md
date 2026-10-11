# Frozen-lambda seed-stability check

This small development experiment tests the frozen-policy candidate
`lambda=0.05` against `lambda=0` on the three already-trained unified VCG
model seeds 15, 16, and 17.

It performs no training. Within each model seed, both lambda settings use
the exact same episode-200 weights, the already-opened EpisodeInstances
85000..85011, and the same four action RNGs per instance. The run therefore
contains 3 x 2 x 12 x 4 = 288 inference-only rows.

The continuation criterion is deliberately simple: all rows must be safe
and complete; the equal-seed aggregate must reduce physical rehandles;
at least two of three individual model seeds must reduce rehandles; and the
aggregate MAE point increase must be at most 2.0. This is a development
robustness check, not a final-panel or budget-convergence claim.

Run:

```bash
cd /path/to/HRL_Management
bash experiments/vcg_unified_frozen_lambda_seed_stability_85k/run.sh prepare
bash experiments/vcg_unified_frozen_lambda_seed_stability_85k/run.sh run
```

The run is resumable by completed 48-row ledger and writes
`frozen-lambda-seed-stability.json`. It does not access 86xxx instances.

## Recorded outcome

The continuation rule passed, strongly: all 288 rows were safe and complete,
and rehandles fell for all three frozen model seeds.

| Frozen policy | Return | MAE | Steps | Rehandles/100 | Within +/-20 |
|---|---:|---:|---:|---:|---:|
| `lambda=0` | 77.15 | 19.87 | 187.44 | 9.29 | 62.85% |
| `lambda=0.05` | 85.18 | 19.32 | 190.26 | 6.94 | 63.02% |

The paired aggregate changes were -2.34 rehandles/100 (nominal 95% CI
`[-4.09, -0.60]`) and -0.55 MAE. This remained a development selection check;
the prospective confirmation is documented in
[Unified VCG](../../docs/UNIFIED_VCG.md).
