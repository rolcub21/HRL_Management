# E218: E215 long-exploration seed replication

E218 trains fresh model seeds 1 and 2 with the E215 recipe.  Architecture,
objective, scenario, certification, progress mask, optimizer, replay capacity,
acceleration, training instances, validation panel, and the 57,300-decision
epsilon schedule are matched.  Only the model seed differs.

Prepare and validate without running an episode:

```bash
bash experiments/conditioned_vcg/E218_e215_long_exploration_replication_277k_v1/run.sh prepare
bash experiments/conditioned_vcg/E218_e215_long_exploration_replication_277k_v1/run.sh self-check
```

Run the two seeds independently, including on separate servers:

```bash
MPLCONFIGDIR=/tmp/e218-s1-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E218_e215_long_exploration_replication_277k_v1/run.sh \
  train-seed --model-seed 1 --through-episode 220 --device cuda

MPLCONFIGDIR=/tmp/e218-s2-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E218_e215_long_exploration_replication_277k_v1/run.sh \
  train-seed --model-seed 2 --through-episode 220 --device cuda
```

Each completed episode is resumable, so either command can be stopped after a
validation checkpoint and restarted unchanged.  The sealed test panel is not
opened.
