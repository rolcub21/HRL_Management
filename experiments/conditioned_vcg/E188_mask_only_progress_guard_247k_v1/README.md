# E188 — choice-preserving progress guard

E188 is a paired one-factor training arm against E187. Everything is held
fixed—including the E78 warm sources, fresh heads, instances, objective,
queue context, exploration, certification, acceleration, and validation
panel—except the progress guard.

At the nonprogress threshold, E188 removes Reconfigure and lets the critic
choose among every remaining certified Accept, Deliver, and Defer. Only when
none remains does it execute one step of a retained exact completion witness;
it rechecks the full progress subset at the next decision boundary.

Training is fresh and paired by model seed and episode instance. It must never
resume an E187 checkpoint. Validation occurs every 20 episodes on the same
five development instances, enabling paired behavior comparisons at episodes
20, 40, 60, and later checkpoints. The sealed test panel remains closed.

The contract asserts the matched E187 fields in code. It preserves 300
episodes, model seeds 0/1/2, the common training-instance seed base, the five
development instances, the exact objective, E78 encoder warm start, queue
representation, epsilon 0.90→0.05 over 18,500 decisions, and every existing
certificate/graph/TD acceleration. Checkpoints are retained every episode.

Prepare and run the CPU self-check before launching a seed:

```bash
MPLCONFIGDIR=/tmp/e188-mpl \
bash experiments/conditioned_vcg/E188_mask_only_progress_guard_247k_v1/run.sh \
  self-check --device cpu
```

```bash
cd /home/ai_diagnosis/HRL_Management
set -o pipefail

MPLCONFIGDIR=/tmp/e188-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E188_mask_only_progress_guard_247k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 300 --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e188-mask-only-progress-guard-247k-v1/seed-0-run.log
```

Use `compare` after any shared validation checkpoint to write a paired E188
versus E187 development report. Running E187 and E188 simultaneously on one
GPU is logically safe but may reduce throughput or exhaust GPU memory.

```bash
MPLCONFIGDIR=/tmp/e188-mpl \
bash experiments/conditioned_vcg/E188_mask_only_progress_guard_247k_v1/run.sh \
  compare
```

The paired report includes J and all five objective components, storage- and
release-clock timing, admission wait, peak inventory, action counts, steps,
rehandles, and guard-forced versus voluntary deliveries for each exact shared
validation day. Negative `E188 minus E187` is better only for cost metrics;
peak inventory and action counts are descriptive behavior measurements.

