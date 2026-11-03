# E204 — E191-centered rehandle Stage B

E204 freezes the three E202-selected E191 controllers and learns only a
preference-conditioned estimate of downstream physical rehandles.  E191's
operational critic and exact certified action frontier remain frozen.

For a deployment rehandle weight `lambda`, candidates are ranked by

```text
Q_E191(s,c) - 0.01 * (lambda - 10)
    * (immediate_rehandle(c) + predicted_future_rehandles(s,c,lambda))
```

Thus `lambda=10` delegates action selection exactly to E191.  The other
declared weights are `0, 5, 20, 40`.  This is a surrogate away from 10: E191's
base values were learned under the continuation policy for weight 10.

The collection design is development-only.  Each of four rounds contains the
full 5-lambda by 3-workload factorial (15 complete episodes).  Five episodes
per round—one at every lambda—are held out from that round's fit.  All E191
certification, mask-only guard, relocation-family proof, timing-invariant
cache, graph-input cache, identity-deduplicated scoring, and path cleanup stay
active.  The terminal Stage-B checkpoints are not paper evidence; a new panel
after E203 is required.

## Preflight and pilot

```bash
cd /home/ai_diagnosis/HRL_Management
set -o pipefail

MPLCONFIGDIR=/tmp/e204-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/run.sh \
  prepare

MPLCONFIGDIR=/tmp/e204-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/run.sh \
  verify-base --model-seed 0 --device cuda

MPLCONFIGDIR=/tmp/e204-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/run.sh \
  verify-center-parity --model-seed 0 --device cuda

MPLCONFIGDIR=/tmp/e204-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/run.sh \
  pilot-collect --model-seed 0 --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e204-e191-centered-rehandle-stage-b-263k-v1/seed-0-run.log
```

The pilot is round 1 episode 1 and is reused by the full run.

## Training

Run one seed per GPU/server when possible:

```bash
MPLCONFIGDIR=/tmp/e204-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/run.sh \
  train-seed --model-seed 0 --through-round 4 --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e204-e191-centered-rehandle-stage-b-263k-v1/seed-0-run.log
```

Replace `0` with `1` or `2` for the other frozen E191 model.  Runs resume only
at fitted round boundaries; completed collection episodes inside an interrupted
round are cached and reused exactly.

After all seeds finish:

```bash
bash experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/run.sh \
  analyze
```
