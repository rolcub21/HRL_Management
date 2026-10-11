# Strictly nested VCG 1.1 handling pilot

This is the fast development check for the architecture we actually want:

```text
VCG 1.1 operational controller (frozen)
                    + optional detached handling-cost head
merit = Qop - lambda * QN
```

At `lambda=0`, the wrapper calls the original VCG 1.1 selector directly. It
does not evaluate `QN` and does not reconstruct V1.1's policy. The pilot first
checks that direct and wrapped execution have the same behavior digest on a
sentinel 85k episode, and reuses the already-authenticated twelve seed-0 V1.1
rows as the lambda-zero endpoint.

`QN` is fit offline from the completed replay already stored in V1.1's seed-0
`latest.pth`. Each decision is labelled with the number of physical
reconfiguration events remaining in that episode. The 500 complete episodes
are split 400/100; the frozen V1 graph/action encoders only produce detached
features, and only the separate cost tower is optimized. There is no new
environment training and no use of the 86k or 87k panels.

The positive-lambda pilot evaluates the fixed grid
`0.01, 0.025, 0.05, 0.10, 0.20` on the exact serialized instances
`85000..85011`. It advances only if a value:

- completes all 12 rows strictly and safely;
- saves at least two physical rehandles across the 96 deliveries;
- increases MAE by no more than 2 steps; and
- decreases dense return by no more than 20.

This is a development screen for a frozen-policy Monte Carlo cost predictor,
not yet a converged constrained-RL or active-budget result.

## Run

After installation under `experiments/vcg_v11_nested_handling_85k/`:

```bash
bash experiments/vcg_v11_nested_handling_85k/run.sh run-all
```

`run-all` executes: prepare, one sentinel parity pair, offline cost fitting,
then 60 positive-lambda evaluation rows. It does not rerun the 12 lambda-zero
rows. Individual commands are also available:

```bash
bash experiments/vcg_v11_nested_handling_85k/run.sh prepare
bash experiments/vcg_v11_nested_handling_85k/run.sh parity
bash experiments/vcg_v11_nested_handling_85k/run.sh train-cost
bash experiments/vcg_v11_nested_handling_85k/run.sh sweep
```

Outputs go to
`results/vcg-v1-1-nested-handling-seed0-85k-development/`.
