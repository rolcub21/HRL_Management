# E247 — objective-matched Kim-inspired placement A3C

This experiment is a **Kim et al. (2020)-inspired placement A3C
adaptation**, not an exact reproduction.  It learns only the storage cell for
an arriving block.  Admission, delivery, defer, and blocked-state physical
reconfiguration use the same rule-based scheduler as the Dynamic PSLAP and GA
baselines.

## Locked fairness contract

- Placement candidates are all cells in the shared **physical** candidate
  mask.  No VCG certificate filters the candidates.
- The VCG progress guard and witness fallback are absent.
- The environment's exact operational objective is used:
  `J = 2T + E + 0.5W + 0.1L + 10R`.
- With `gamma=1`, each placement is trained from its realized future
  `-0.01 J` return under the common baseline scheduler.
- The five-channel state is an explicit extension of the paper's relative
  dwell grid: occupancy, storage mask, traversability, and exit proximity are
  needed to represent this repository's walls, irregular storage cells, and
  exits.
- Training runs for at least 6,000 and at most 11,000 episodes.  After 6,000,
  it stops only after eight 250-episode validation checks show less than 0.5%
  material improvement.  The learning curve and selected checkpoint remain.
- Sealed evaluation uses five stochastic trials per instance.  Completion is
  reported over every trial; J is summarized only among strict completers.

## Commands

Prepare without opening either sealed panel:

```bash
bash experiments/conditioned_vcg/E247_kim2020_objective_matched_a3c_306k_v1/run.sh prepare
```

Train one seed (repeat for seeds 0, 1, and 2 at each size):

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E247_kim2020_objective_matched_a3c_306k_v1/run.sh \
  train-seed --size 5x5 --model-seed 0 --through-episode 11000 --device cuda
```

After training is finalized, evaluate that seed:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E247_kim2020_objective_matched_a3c_306k_v1/run.sh \
  evaluate-seed --size 5x5 --model-seed 0 --device cuda
```

The same commands with `--size 8x8` train and evaluate the larger yard.
Evaluation uses the E201 5×5 and E222 8×8 sealed instance identities, so its
rows can join the existing Dynamic PSLAP and GA tables directly.
