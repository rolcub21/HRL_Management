# E220: E215 episode-300 Stage B, seed 0

E220 applies the frozen E214/E216 Stage-B procedure to the finalized E215
episode-300 seed-0 Stage-A checkpoint.  Stage A, certification, the progress
mask, and all accelerations remain frozen.

Training uses four rounds of nine on-policy development episodes.  Each round
crosses the 20%, 50%, and 80% workload levels with continuous alpha draws from
10--20, 20--30, and 30--40.  The round-start future-rehandle head is frozen
during collection and supplies the eta-return bootstrap; only current-round
data are fitted.

Endpoint evaluation uses three new development EpisodeInstances.  The learned
future head is compared at alpha 20 and 40 against an exact immediate-only dial
built on the same E215 checkpoint, with alpha 10 retained as exact Stage-A
delegation.  No sealed test panel is opened and the result is not yet multiseed
paper evidence.

Run the resumable training and then the paired endpoint evaluation:

```bash
MPLCONFIGDIR=/tmp/e220-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E220_e215_ep300_stage_b_279k_v1/run.sh \
  train --through-round 4 --device cuda

MPLCONFIGDIR=/tmp/e220-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E220_e215_ep300_stage_b_279k_v1/run.sh \
  evaluate --device cuda
```

Each completed collection episode and each round endpoint is durable, so the
training command can be rerun after interruption.
