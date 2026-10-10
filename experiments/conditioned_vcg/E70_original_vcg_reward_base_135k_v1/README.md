# E70: original-architecture reward-base test

This development-only test repeats the E66 additive delivery-base change on
the **original target-size VCG-Dense Stage A v4**, not the redesigned E60
controller. The fresh arm trains from scratch at 8×8/36 jobs with base 80.
The historical base-40 seed-0 run supplies the matched control. Architecture,
gamma 0.99, training/validation instance seeds, epsilon schedule, optimizer,
certified frontier, guard, and v4 acceleration remain unchanged. Only the
delivery constant changes from 40 to 80; the lateness slopes remain 1.5 and
0.5. The full budget stays 112 episodes; this screen inspects checkpoints
through episode 30 without committing to the full budget.

Run from the repository root. Preparation and self-check do not run episodes:

```bash
bash experiments/conditioned_vcg/E70_original_vcg_reward_base_135k_v1/run.sh prepare
bash experiments/conditioned_vcg/E70_original_vcg_reward_base_135k_v1/run.sh self-check
```

You run the training. CPU is permitted for this development test, but may be
slow. `train` resumes its own latest checkpoint if called again:

```bash
bash experiments/conditioned_vcg/E70_original_vcg_reward_base_135k_v1/run.sh train --through-episode 12 --device cpu
bash experiments/conditioned_vcg/E70_original_vcg_reward_base_135k_v1/run.sh analyze
```

If useful, continue to episode 30 and analyze again:

```bash
bash experiments/conditioned_vcg/E70_original_vcg_reward_base_135k_v1/run.sh train --through-episode 30 --device cpu
bash experiments/conditioned_vcg/E70_original_vcg_reward_base_135k_v1/run.sh analyze
```

The report is `results/vcg-conditioned-e70-original-vcg-reward-base-135k-v1/e70-report.json`.
It compares 6-episode checkpoints on the same five development instances.
For strictly complete trajectories, base-80 undiscounted return is recentered
by subtracting 40 per delivery. This is a one-seed mechanism test, not a
confirmation or paper result. A favorable result should be checked with more
seeds and untouched instances before making claims about the architecture.
