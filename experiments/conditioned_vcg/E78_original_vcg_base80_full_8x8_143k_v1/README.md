# E78: full 8×8 base-80 original-VCG Stage A

This development protocol trains the **original VCG-Dense v1.1 Stage-A
architecture** on the empty-start 8×8/36-job workload with the E70 delivery
reward `80 - 1.5|e| - 0.5(|e|-20)+`. The primitive-step discount remains
`γ = 0.99`. No encoder, hierarchy, certification rule, optimizer, or epsilon
schedule is changed. This is not the E60 architecture and is not Stage B.

The full work-matched budget is **112 episodes per model seed** (4,032 jobs
per seed). Model seeds are 0, 1, and 2. Seed 0 continues the authenticated
E70 base-80 run already at episode 30 under its original frozen contract;
seeds 1 and 2 start independently under E78. Reusing that prefix avoids
discarding 30 expensive episodes. E78 checks the E70 contract and the
immutable first-30 validation record before training or analysis.

The stability analysis is declared in advance: all validation checkpoints
from episodes 60, 66, …, 108 and the final episode 112 are reported. Terminal
and late-window MAE, return, completion, rehandles, and steps are primary;
the trainer's best-development checkpoint is recorded only descriptively.
There is no checkpoint cherry-picking and no test panel is opened. Stage B
will be considered only after these Stage-A trajectories are inspected.

## Commands

From the repository root, preparation and self-check run no episodes:

```bash
bash experiments/conditioned_vcg/E78_original_vcg_base80_full_8x8_143k_v1/run.sh prepare
bash experiments/conditioned_vcg/E78_original_vcg_base80_full_8x8_143k_v1/run.sh self-check
```

Run the three seeds, preferably sequentially on one GPU. Each command is
resumable after an interruption and defaults to episode 112:

```bash
bash experiments/conditioned_vcg/E78_original_vcg_base80_full_8x8_143k_v1/run.sh train-seed --model-seed 0
bash experiments/conditioned_vcg/E78_original_vcg_base80_full_8x8_143k_v1/run.sh train-seed --model-seed 1
bash experiments/conditioned_vcg/E78_original_vcg_base80_full_8x8_143k_v1/run.sh train-seed --model-seed 2
bash experiments/conditioned_vcg/E78_original_vcg_base80_full_8x8_143k_v1/run.sh analyze
```

`--through-episode N` can be supplied to stop safely at an earlier episode
and resume later. The command requires CUDA explicitly; it will not silently
substitute another device. `analyze` can be run between seeds and writes a
partial report until all three reach episode 112. The report is
`results/vcg-conditioned-e78-original-base80-full-8x8-143k-v1/e78-report.json`.
The underlying dense trainer has a hard-coded `objective=40...` console banner;
that banner does not reflect this run's authenticated base-80 reward. Check the
training summary's `objective_spec` or the E78 contract instead.

This establishes whether base-80 training itself is reproducible across
three model seeds. It does **not** by itself prove base 80 is causally more
stable than base 40: only seed 0 has a matched base-40 control. A fresh,
untouched evaluation panel and any Stage-B operating-profile comparison are
separate later steps.
