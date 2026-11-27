# TF-SN on OGBench AntMaze Large

This directory stages the first gate in the frozen evaluation campaign. It is
development-only: it does not reset or inspect any of OGBench's five native
AntMaze Large task rows.

The method is frozen R74. OGBench supplies the environment, reward-free
`navigate` dataset, and the official CRL executor implementation. The CRL
checkpoint is shared unchanged by TF-SN, SGM, and the direct-executor
ablation. CRL training is executor preparation; it does not receive a hidden
evaluation goal.

Unlike the historical SG E376 experiment, this port has no scaled maze
constants, fixed spatial grid, support-point budget, waypoint annulus, or
hand-set routing horizon. `sigma`, reach covariance, latent-radius bounds, and
hubs retain their frozen data-derived definitions. OGBench's 1,000-step
episode and the global 100-step command contract imply exactly 10 commands.

## Current gate

Run the static check first:

```bash
cd /home/ai_diagnosis/HRL_Management
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/run.sh verify
```

Download and audit the official reward-free dataset:

```bash
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/run.sh download
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/run.sh preflight
```

Then train only executor seed 0:

```bash
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/run.sh train-executor-seed0
```

This is the official OGBench 1.2.1 CRL command for AntMaze Large, with task
evaluation disabled during training. It writes a separate checkpoint under
`results/ogbench_antmaze_large_r18/` and does not touch the discontinued
Gymnasium rebuild.

After this checkpoint exists, the next gate is a chained 100-step command
audit on deterministic validation-dataset waypoints. Only after the executor
gate is recorded do we spend the 36,000-command TF-SN preparation budget.
