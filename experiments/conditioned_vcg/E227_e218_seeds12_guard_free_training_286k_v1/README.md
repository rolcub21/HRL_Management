# E227 — E218 seeds 1/2 guard-free training

E227 repeats E226's single intervention for E218 model seeds 1 and 2. It
removes the Reconfigure mask, the retained-witness fallback, and the selector
bypass. The critic and Bellman backup use the complete exact-SAFE frontier.

Everything else is matched to E218: fresh initialization by model seed,
training days, long-stay workload mixture, objective, network and inputs,
57,300-decision exploration schedule, replay, certification, accelerations,
Defer limits, validation every 20 episodes, and per-episode resumability.

On each server, prepare and smoke the seed before training it:

```bash
bash experiments/conditioned_vcg/E227_e218_seeds12_guard_free_training_286k_v1/run.sh \
  self-check

bash experiments/conditioned_vcg/E227_e218_seeds12_guard_free_training_286k_v1/run.sh \
  smoke --model-seed 1 --device cuda
```

The early-read command deliberately pauses at episode 60:

```bash
MPLCONFIGDIR=/tmp/e227-s1-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E227_e218_seeds12_guard_free_training_286k_v1/run.sh \
  train-seed --model-seed 1 --through-episode 60 --device cuda
```

Use model seed 2 and `/tmp/e227-s2-mpl` on the second server. After reviewing
episodes 20, 40, and 60, rerun the same command with `--through-episode 300`;
it resumes rather than restarting.

