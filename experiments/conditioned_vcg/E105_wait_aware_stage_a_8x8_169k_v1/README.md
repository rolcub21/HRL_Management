# E105: wait-aware Stage-A 8x8 pilot

E105 asks whether Stage A can learn timely admission when the reward directly
charges excess outside waiting. It retains the original VCG-Dense v1.1
architecture, E78's base-80 storage-clock delivery reward, certification,
guard, replay, optimizer, gamma, exploration schedule, workload, and E100's
exact graph-input acceleration.

The one change is an accrued primitive reward:

```text
-0.5 * number of released, unstored jobs whose outside age is at least 80
```

For a completed job with outside wait `W`, this sums exactly to
`-0.5 * max(0, W - 80)`. This is an objective change, not policy-invariant
reward shaping. It deliberately makes admission delay visible before delivery.

The pilot uses model seed 0 only, validates every six episodes on E78's five
reused development instances, reports every checkpoint, and does not select a
best checkpoint. It records mean outside wait, cap breaches, release-clock and
storage-clock MAE and signed error, steps, rehandles, guard activity, and exact
wait-penalty accounting. Stage B and an independent test are closed.

The borrowed trainer's startup banner still contains a hard-coded
`objective=40...` label. That label is stale: the authenticated E105 contract
and environment use E78's base-80 delivery term plus the wait charge above.

## Run from the repository root

First run the episode-free checks:

```bash
bash experiments/conditioned_vcg/E105_wait_aware_stage_a_8x8_169k_v1/run.sh prepare
bash experiments/conditioned_vcg/E105_wait_aware_stage_a_8x8_169k_v1/run.sh self-check
```

Run the initial 12-episode mechanism screen on CUDA, then summarize it:

```bash
bash experiments/conditioned_vcg/E105_wait_aware_stage_a_8x8_169k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 12 --device cuda
bash experiments/conditioned_vcg/E105_wait_aware_stage_a_8x8_169k_v1/run.sh analyze
```

If it remains strictly complete with finite metrics, resume the same run to
episode 36 and analyze again:

```bash
bash experiments/conditioned_vcg/E105_wait_aware_stage_a_8x8_169k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 36 --device cuda
bash experiments/conditioned_vcg/E105_wait_aware_stage_a_8x8_169k_v1/run.sh analyze
```

The episode-36 development gate requires strict completion, mean outside wait
at most 256, storage-clock MAE at most 20, at least 70% within +/-20, no more
than 1200 mean steps, and exact reward accounting. Passing would authorize a
fresh matched confirmation design—not Stage B or a paper claim by itself.
