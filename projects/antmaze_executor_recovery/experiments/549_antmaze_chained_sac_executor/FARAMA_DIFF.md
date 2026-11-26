# E549 change audit against pinned Farama

Upstream is pinned to commit
`f4579c7760369b106115bdafbcd00e549e850434`. The untouched source files are
vendored under `farama_original/`.

## Method changes

Only the two requested parts of the executor problem change:

1. `GoalReachAnt` uses the `AntMaze_Medium_Diverse_GR-v4` Ant and maze XML
   instead of the open-plane Ant XML. Reset positions therefore come from the
   benchmark maze and command endpoints are sampled from the benchmark's own
   goal-noise footprint inside every free cell.
2. A goal changes on contact or after exactly 100 primitive steps. The MuJoCo
   state is never reset at a command handoff. Goal direction remains uniformly
   sampled as in Farama; its radius is replayed from the 2,000 reward-free
   command requests in the completed R47 TF-SN alignment run.

The 29D observation construction, dense directional reward, action space,
SAC hyperparameters, 4x256 ReLU policy, and physical 1,000-step episode remain
unchanged. `verify_protocol.py` checks these facts before training.

## Operational-only changes

Video and WandB logging are disabled, checkpoints and replay buffers are saved
locally every 500k steps, and explicit local resume is supported. These do not
change experience, rewards, model updates, or actions.

Render the complete line diff before training with:

```bash
bash projects/antmaze_executor_recovery/experiments/549_antmaze_chained_sac_executor/run.sh source-diff
```

