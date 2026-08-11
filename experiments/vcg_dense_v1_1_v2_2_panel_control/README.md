# VCG-Dense V1.1 control on the opened V2.2 panel

This development-only control answers one narrow question: was MAE below 20
achievable on the exact EpisodeInstances used to calibrate constrained V2.2?

It authenticates the finalized selected-best VCG-Dense V1.1 checkpoints for
model seeds 0, 1, and 2. It regenerates EpisodeInstances 85000--85011 and
requires their instance IDs, schedule IDs, and canonical SHA-256 hashes to
match V2.2's authenticated validation manifest. Each selected checkpoint is
then evaluated once on each instance with epsilon zero, no learning, the dense
V1.1 objective, and gamma 0.99: 3 model seeds x 12 instances = 36 deterministic
rollouts.

The report includes each model seed, an equal-model-seed aggregate, delivery
positions 1--8 plus first-two/later splits, strict completion/exact-safety
checks, and a descriptive comparison with V2.2's closest candidate look at
episode 80. This comparison is not a final-performance or inferential claim.
The 86000--86029 final panel remains unopened.

From the repository root, run:

```bash
bash experiments/vcg_dense_v1_1_v2_2_panel_control/run.sh
```

The run is sequential and writes one authenticated ledger after every
rollout. If interrupted, resume the exact same output with:

```bash
bash experiments/vcg_dense_v1_1_v2_2_panel_control/run.sh --resume-existing
```

Outputs are written under
`results/vcg-dense-v1-1-v2-2-panel-control-12instance/`. The primary files are
`control-report.json`, `control-audit.json`, and `control-runs.csv`.
