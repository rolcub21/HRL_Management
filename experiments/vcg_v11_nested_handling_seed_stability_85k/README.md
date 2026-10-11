# Nested VCG 1.1 handling: seed-stability continuation

This additive development experiment follows the completed seed-0 pilot. It
does not modify or rerun that pilot.

The selected operating point is frozen at `lambda=0.025`, the smallest value
that passed the seed-0 gate. For VCG 1.1 model seeds 1 and 2, the experiment:

1. authenticates the selected operational checkpoints (episodes 450 and 375);
2. reads historical replay only through those selected episode cutoffs;
3. fits one detached handling-cost head per seed while keeping Qop frozen;
4. evaluates only `lambda=0.025` on serialized instances `85000..85011`;
5. reuses the authenticated VCG 1.1 lambda-zero rows; and
6. aggregates model seeds 0, 1, and 2 with equal weight.

Only 24 new evaluation rows are executed. There is no Qop or environment
retraining, no new lambda sweep, and no 86k/87k access.

## Stability criterion

All 72 reused/new method rows must be strict-safe-complete. Both genuinely new
seeds 1 and 2 must individually save at least two rehandles over their 96
deliveries while keeping MAE within +2 and dense return within -20 of their
own lambda-zero endpoints. The equal-seed aggregate must reduce rehandles and
meet the same MAE/return tolerances.

This remains a development replication of a historical-behavior Monte Carlo
handling predictor. It is not an independent test-panel confirmation or a
dual-convergence claim.

## Run

```bash
bash experiments/vcg_v11_nested_handling_seed_stability_85k/run.sh run-all
```

Individual resumable phases are:

```bash
bash experiments/vcg_v11_nested_handling_seed_stability_85k/run.sh prepare
bash experiments/vcg_v11_nested_handling_seed_stability_85k/run.sh train-seed1
bash experiments/vcg_v11_nested_handling_seed_stability_85k/run.sh train-seed2
bash experiments/vcg_v11_nested_handling_seed_stability_85k/run.sh evaluate-seed1
bash experiments/vcg_v11_nested_handling_seed_stability_85k/run.sh evaluate-seed2
bash experiments/vcg_v11_nested_handling_seed_stability_85k/run.sh analyze
```

Outputs go to
`results/vcg-v1-1-nested-handling-seed-stability-85k-development/`.
