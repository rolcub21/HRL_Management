# E255: E195 temperature-scaled 5x5 diagnostic

E255 is a fresh seed-0 rerun of E195. It changes only the four Boltzmann
temperatures. Their fixed divisor is the ratio between the three-seed mean
selected development-validation costs at 8x8 and 5x5:

```text
2390.296296 / 73.522222 = 32.511209
```

Consequently, the within-mode temperatures change from `0.1` to
`0.0030758623`, and the mode temperature changes from `1.0` to `0.0307586228`.
The six source summaries, selected checkpoints, selected costs, and hashes are
part of the authenticated contract. No test-panel result is used.

Architecture, reward scaling, objective, optimizer, replay, E195's original
epsilon schedule, 300-episode budget, certification, guard, training days,
validation cadence, and checkpoint selection remain unchanged.

Prepare and self-check without running an episode:

```bash
bash experiments/conditioned_vcg/E255_e195_temperature_scaled_5x5_314k_v1/run.sh prepare
bash experiments/conditioned_vcg/E255_e195_temperature_scaled_5x5_314k_v1/run.sh self-check
```

Train the one predeclared seed:

```bash
MPLCONFIGDIR=/tmp/e255-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E255_e195_temperature_scaled_5x5_314k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 300 --device cuda
```

After training, compare all development checkpoints with E195 seed 0:

```bash
bash experiments/conditioned_vcg/E255_e195_temperature_scaled_5x5_314k_v1/run.sh analyze
```
