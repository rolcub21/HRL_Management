# E195 — E191 method on scale-matched 5×5 long-stay workloads

E195 changes only the physical scale and its workload normalization. It keeps
E191's network, eight public queue/progress inputs, exact objective
`2T + E + 0.5W + 0.1L + 10R`, gamma 1, certification, mask-only progress
guard, checkpointing, and exact acceleration paths.
Every learned tensor, optimizer, replay buffer, counter, guard state, and
certificate cache starts fresh.

The 5×5 yard has eight storage cells and eight jobs. E191's expected stay per
cell is `700 / 35 = 20`; E195 therefore uses Poisson-160 stays so that
`160 / 8 = 20`. The unchanged deterministic release gaps 100, 40, and 25
produce the same nominal 20%, 50%, and 80% occupancy regimes. Episodes start
empty, and validation uses three paired-duration cases per regime.

This is an episode-matched scale comparison, not a job- or decision-matched
one: E195 has eight jobs per episode while E191 has 36. The old successful
5×5 checkpoint is not used for initialization, and its old dense objective
and gamma 0.99 are not used for learning. A frozen counterfactual comparison
with that historical model remains a separate evaluation.

Exploration retains E191's linear 0.90-to-0.05 rule, but its decision decay is
scaled by the fixed jobs-per-episode ratio: `18,500 × 8 / 36 = 4,111`
decisions. Copying 18,500 literally would leave the much shorter 5×5 run far
above its epsilon floor at episode 300. This normalization preserves E191's
exploration exposure per training job and is declared in the contract.

Run the episode-free checks first:

```bash
bash experiments/conditioned_vcg/E195_e191_method_5x5_longstay_254k_v1/run.sh \
  self-check
```

Run three resumable pilot episodes on seed 0:

```bash
MPLCONFIGDIR=/tmp/e195-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E195_e191_method_5x5_longstay_254k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 3 --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e195-e191-method-5x5-longstay-254k-v1/seed-0-run.log
```

If all three finish strictly, resume the same output. Validation runs every
20 episodes on the fixed nine-case development panel:

```bash
MPLCONFIGDIR=/tmp/e195-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E195_e191_method_5x5_longstay_254k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 300 --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e195-e191-method-5x5-longstay-254k-v1/seed-0-run.log
```

The 2,000-step limit is an error cap, not an action rule. The sealed test
panel remains closed, and this protocol does not authorize Stage B.
