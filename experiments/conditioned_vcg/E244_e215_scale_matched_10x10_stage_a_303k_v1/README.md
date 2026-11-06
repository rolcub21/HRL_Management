# E244: final Stage A at 10×10

E244 trains the final E215/E218 Stage-A method from fresh random initialization
on the scale-matched 10×10 long-stay yard first defined by E209.

The method is unchanged: the corrected objective
`2T + E + 0.5W + 0.1L + 10R`, exact certification, the mask-only progress
guard, the same network, replay, optimizer, and decision-preserving
accelerations. The yard has 63 storage cells and 64 jobs. Mean stay is 1260
steps, preserving 20 steps per storage cell, and arrival gaps 100, 40, and 25
define the nominal 20%, 50%, and 80% workloads.

The important training correction relative to E209 is the final E215/E218
exploration schedule. Its 57,300-decision decay for 36 jobs is normalized to
64 jobs: `round(57300 * 64 / 36) = 101867`. Training lasts 300 episodes, with
fixed nine-case development validation every 20 episodes and a checkpoint
every episode. The sealed test panel remains closed.

## Preflight

```bash
cd /path/to/HRL_Management
MPLCONFIGDIR=/tmp/e244-mpl \
bash experiments/conditioned_vcg/E244_e215_scale_matched_10x10_stage_a_303k_v1/run.sh \
  self-check --device cpu
```

## Recommended launch sequence

Run a three-episode seed-0 smoke first. The same command is resumable, so the
episode-20 and episode-300 commands continue the existing checkpoint.

```bash
export E244_RESULTS="$PWD/results/vcg-conditioned-e244-e215-scale-matched-10x10-stage-a-303k-v1"
mkdir -p "$E244_RESULTS"
set -o pipefail

MPLCONFIGDIR=/tmp/e244-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E244_e215_scale_matched_10x10_stage_a_303k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 3 --device cuda 2>&1 | \
tee -a "$E244_RESULTS/seed-0-run.log"

MPLCONFIGDIR=/tmp/e244-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E244_e215_scale_matched_10x10_stage_a_303k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 20 --device cuda 2>&1 | \
tee -a "$E244_RESULTS/seed-0-run.log"

MPLCONFIGDIR=/tmp/e244-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E244_e215_scale_matched_10x10_stage_a_303k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 300 --device cuda 2>&1 | \
tee -a "$E244_RESULTS/seed-0-run.log"
```

After the seed-0 smoke passes, seeds 1 and 2 may run in parallel on separate
GPUs or servers by changing `--model-seed`. Each seed has its own independent
checkpoint directory. Do not point two processes at the same seed/output.

Allow at least 10 GB free per server when retaining large resumable
`latest.pth` checkpoints and logs. Selected checkpoints are much smaller, but
the resumable replay-bearing files are not.

