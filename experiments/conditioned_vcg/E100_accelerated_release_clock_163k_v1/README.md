# E100: E98 training with an exact graph-input cache

Continues E98 (release-clock Stage A, seed 0) with one change: how Q-network
graph inputs are built. Reward, certification, guard/defer rules, replay,
optimizer, exploration, validation and the frozen E98 contract are unchanged;
E100 calls E98's `train()` with its output directory redirected.

## Why

Profiling one E98 TD update (128 transitions, ~18.8k continuation candidates,
~19k unique states, RTX A6000) gave ~5.3 s steady state and ~10 s after a
target refresh. About 90% was Python building graph inputs
(`encode_recovery_state`, padding, list-to-tensor); the GNN forward is small.
Inputs do not depend on network weights, so they need to be built only once
per state.

## What `fast_inputs.py` does

- One static template per yard geometry (9 geometry feature columns plus the
  edge list), built by the reference encoder.
- Per state, the 6 state-dependent columns (agent, occupied, time known,
  scaled/absolute remaining time, overdue) are computed with the reference's
  float64 arithmetic and stored once on the device.
- Each forward gathers template + dynamic rows and passes tensors directly to
  the unchanged graph encoder. Online and target networks share the cache.
- Patched paths: TD continuation scoring (`v3._deduplicated_no_grad_scores`)
  and `ViabilityGraphQNetwork.forward` (action selection and the chosen-action
  gradient path). The V4 target-score cache is unchanged.
- Keys are object identity with the state retained (no id reuse); capacity 4M
  states (~6 GB float32), clear-and-rewarm when full.

## Verification (`benchmark.py`, checkpoint episode 11, CUDA)

- Inputs bit-identical (`torch.equal`) to `padded_batch_tensors(pad_yard_graphs(...))`
  for all 33,684 states in two sampled batches.
- TD predictions, targets, losses and all gradients: max difference 1.7e-5
  versus reference; the reference differs from itself by up to 1.3e-5 on the
  same batch (CUDA `index_add_` atomics). Target-refresh parity also checked.
- Fully warm update: 0.58 s versus ~5.3 s (≈9×). Short benchmark with partly
  cold cache: 2.2× cold, 3.0× median warm.

Certification-frontier time (~250 s/episode in E98) is not accelerated and is
expected to dominate after this change.

## Usage (repository root)

```bash
PY=.venv/bin/python
MPLCONFIGDIR=/tmp/e100-mpl $PY -m experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1.benchmark \
  --checkpoint results/vcg-conditioned-e98-release-clock-feasibility-163k-v1/training/latest.pth \
  --output results/vcg-conditioned-e100-accelerated-release-clock-163k-v1/benchmark-gpu
$PY -m experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1.run seed    # copy E98 checkpoint dir
$PY -m experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1.run train   # resume to episode 112
```

Evaluation: E98's `evaluate.py` reads `e98.run.OUT` (comparators and
`training/`). To evaluate E100, copy E98's `comparators/` into the E100 output
directory and run `evaluate compare` with `e98.run.OUT` patched to it.
