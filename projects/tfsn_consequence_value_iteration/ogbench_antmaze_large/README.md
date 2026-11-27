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

Freeze and run that gate with:

```bash
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/run.sh \
  audit-executor-seed0
```

The bank uses all 100 complete held-out validation trajectories. Each chain
starts with a fresh simulator reset at its recorded source XY and receives the
ten validation states at exact 100-step intervals as commands. Only XY is
copied into the executor goal observation; native task rows and task goals are
never read. The audit consumes no reward and applies no behavioral pass
threshold. It reports continuous endpoint error, displacement, progress,
uprightness, and early-versus-late chain behavior. Contact at OGBench's native
radius is evaluator-only diagnostic information. Native goal-contact
termination is disabled for this audit so every command executes the frozen
100-step contract; success timing, environment dynamics, and policy inputs are
unchanged, and the adapter is recorded in the audit output.

The frozen seed-0 audit found 15 inverted chains out of 100, with no recovery.
The 36,000-command TF-SN preparation is therefore paused. The separate R19
executor protocol continues the same vendored CRL learner on real chained
commands and checks it at predeclared +500k and +1M environment-step budgets:

```bash
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/run.sh \
  finetune-executor-500k
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/run.sh \
  audit-finetuned-executor-500k
```

Only if the 500k gate fails should the continuation run:

```bash
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/run.sh \
  finetune-executor-1m
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/run.sh \
  audit-finetuned-executor-1m
```

The gate was frozen before fine-tuning: at most 5% of validation chains may
invert, while per-command contact on the identical bank must remain at least
37.6%. Training never reads the validation bank, native tasks, or task reward.
Its target offsets are every exact 100-step displacement in the reward-free
training trajectories, with no hand-set metric distance interval.

The R19 +500k checkpoint failed this gate despite eliminating inversions:
contact collapsed from 37.6% to 0.2%. A data-contract audit found that R19
sampled every update exclusively from online replay and bounded hindsight
goals at each 100-step command. The online actor-goal median was 0.29 m while
the commanded-target median was 7.08 m. R19 is retained as failed development
evidence and must not be continued.

R20 restarts from the original 1M offline checkpoint. It keeps the unchanged
vendored CRL loss, mixes equal offline and chained-online rows in every update,
and uses the physical 1000-step episode as the online future-goal boundary.
The settings are global data-contract choices declared before the restart:

```bash
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/run.sh \
  finetune-mixed-executor-500k
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/run.sh \
  audit-mixed-finetuned-executor-500k
```

R20 writes to `chained_mixed_finetune_r20`; it neither reads nor resumes the
failed R19 checkpoint. If its 500k gate fails, audit the mixed data contract
before invoking the separately available 1M continuation.

R20 also failed and is stopped. Before any further executor training, R21
audits the untouched official checkpoint under adaptive empirical-reach
commands. The Gymnasium AntMaze reach (about 2.6 m RMS) is not used: exact
100-step OGBench validation transitions have a 9.23 m RMS displacement. At
each actual handoff R21 selects the held-out transition whose source is
nearest in the standardized native observation space, then commands that
transition's recorded endpoint. This removes stale nominal-trajectory drift
without adding a metric distance band:

```bash
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/run.sh \
  audit-executor-adaptive-reach-seed0
```

The audit contract and its 5% maximum chain-inversion decision bar are frozen
in `EXECUTOR_ADAPTIVE_REACH_AUDIT_PROTOCOL.json`. Passing selects the original
official 1M checkpoint. Failing ends CRL fine-tuning development and sends the
executor work to the separately budgeted E549-style online recipe.

R21 failed: 18 of 100 chains inverted and none recovered. R22 therefore trains
the E549-style SAC executor from scratch in OGBench AntMaze Large. Its complete
protocol, E549 diff, and stopped-gate commands live under
`e549_executor_r22/`. Start only the first gate:

```bash
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/e549_executor_r22/run.sh \
  train-1m
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/e549_executor_r22/run.sh \
  audit-1m
```

The R22 1M checkpoint failed both frozen gates: 45 of 100 chains inverted
(maximum allowed: 5%), and evaluator-radius contact was 14.5% (minimum
allowed: 38.1%). The checkpoint and full replay buffer are retained. Under the
predeclared stopping protocol, the same from-scratch run therefore continues
to 2M rather than restarting:

```bash
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/e549_executor_r22/run.sh \
  resume-2m
bash projects/tfsn_consequence_value_iteration/ogbench_antmaze_large/e549_executor_r22/run.sh \
  audit-2m
```
