# E549 — AntMaze-native chained SAC executor

E549 trains the low-level Ant executor from scratch inside Medium Diverse
AntMaze while preserving Farama's observation, reward, SAC optimizer, and
network. Every command is held until contact or 100 primitive steps, then a new
goal is issued without resetting the Ant. Every primitive training step is part
of the reported executor budget.

The protocol and exact deviations are in `protocol.json`; the human-readable
source audit is in `FARAMA_DIFF.md`.

## Run order

Nothing is launched in the background. Run each stage explicitly:

```bash
cd /home/ai_diagnosis/HRL_Management

# Cheap preflight: freezes R47 distances, audits sources/config/interfaces,
# and executes 205 primitive smoke steps.
bash projects/antmaze_executor_recovery/experiments/549_antmaze_chained_sac_executor/run.sh verify

# Inspect the complete source diff before expensive work.
bash projects/antmaze_executor_recovery/experiments/549_antmaze_chained_sac_executor/run.sh source-diff

# From-scratch seed-42 training and first paired executor gate.
bash projects/antmaze_executor_recovery/experiments/549_antmaze_chained_sac_executor/run.sh train-500k
bash projects/antmaze_executor_recovery/experiments/549_antmaze_chained_sac_executor/run.sh evaluate-500k

# Continue the same model and replay buffer to 1M, then gate again.
bash projects/antmaze_executor_recovery/experiments/549_antmaze_chained_sac_executor/run.sh resume-1m
bash projects/antmaze_executor_recovery/experiments/549_antmaze_chained_sac_executor/run.sh evaluate-1m
```

Each gate uses 32 paired resets and the exact same ten absolute commands for
the candidate and the frozen 18M open-plane Farama executor. It reports command
success, physical inversion rate, distance covered, and endpoint error. It does
not run TF-SN or access hidden goals.

The deployment-matched executor-only adaptive screen samples every next
command from the endpoint the 5M executor actually reached. It uses the same
frozen reward-free distance bank, 100-step command horizon, 10-command physical
episode, and 32 reset seeds, but no TF-SN planner or Farama comparator:

```bash
bash projects/antmaze_executor_recovery/experiments/549_antmaze_chained_sac_executor/run.sh \
  evaluate-5m-adaptive
```

It reports success by command position and by empirical requested-distance
quintile. The latter are diagnostic bins derived from the screen itself, not
executor or planner settings.

If the 1M checkpoint is competitive, continue the same run and replay buffer:

```bash
bash projects/antmaze_executor_recovery/experiments/549_antmaze_chained_sac_executor/run.sh resume-30m
```

Every 500k checkpoint has a paired replay buffer. After an interruption later
in training, resume from the newest complete pair without restarting:

```bash
bash projects/antmaze_executor_recovery/experiments/549_antmaze_chained_sac_executor/run.sh \
  resume TARGET_TOTAL_STEPS /absolute/path/to/GoalAntMazeChained_model_N_steps.zip
```

Outputs are written under
`projects/antmaze_executor_recovery/results/experiments/549_antmaze_chained_sac_executor/`.

