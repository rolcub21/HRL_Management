# E226 — E215 seed-0 guard-free training

E226 changes one behavior from E215 seed 0: it removes the Reconfigure mask,
the retained-witness fallback, and the selector bypass. The critic chooses
from the complete exact-SAFE frontier, and the Bellman backup uses that same
frontier. Certification, the objective, neural inputs, Defer limits, training
days, seed, exploration, replay, validation, checkpoint selection, and all
accelerations remain matched to E215.

The passive four-value non-progress context remains an input so that removing
the guard does not also change the network architecture.

Prepare and test without launching training:

```bash
MPLCONFIGDIR=/tmp/e226-mpl \
bash experiments/conditioned_vcg/E226_e215_seed0_guard_free_training_285k_v1/run.sh \
  self-check --device cpu

MPLCONFIGDIR=/tmp/e226-mpl \
bash experiments/conditioned_vcg/E226_e215_seed0_guard_free_training_285k_v1/run.sh \
  smoke --device cuda
```

The official training command is intentionally separate:

```bash
MPLCONFIGDIR=/tmp/e226-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E226_e215_seed0_guard_free_training_285k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 60 --device cuda
```

The early read is fixed at episodes 20, 40, and 60. Continue only after
reviewing `analyze` against the matched E215 seed-0 checkpoints.

