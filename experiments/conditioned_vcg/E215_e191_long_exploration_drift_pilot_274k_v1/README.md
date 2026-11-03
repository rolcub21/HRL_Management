# E215: E191 long-exploration late-drift diagnostic

E215 is a fresh, matched seed-0 rerun of E191.  It changes one training
quantity: epsilon decays from 0.90 to 0.05 over 57,300 macro decisions instead
of 18,500.  E191 seed 0 used 31,252 decisions in 120 episodes, so the new floor
is calibrated to approximately episode 220.  Architecture, initialization,
objective, certification, progress mask, optimizer, 20,000-transition rolling
replay, acceleration, workload sequence, and the fixed nine-case development
panel are unchanged.

This isolates the short-exploration explanation for E191's late checkpoint
drift.  It does not yet repair replay overwriting: the buffer still fills near
episode 75.  If drift remains while epsilon is high, the next isolated arm is
retained or workload-stratified replay.

The run is resumable after every completed episode.  Prepare and self-check do
not run an episode:

```bash
bash experiments/conditioned_vcg/E215_e191_long_exploration_drift_pilot_274k_v1/run.sh prepare
bash experiments/conditioned_vcg/E215_e191_long_exploration_drift_pilot_274k_v1/run.sh self-check
```

The first matched decision point is episode 120:

```bash
MPLCONFIGDIR=/tmp/e215-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E215_e191_long_exploration_drift_pilot_274k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 120 --device cuda
```

Analyze against original E191 seed 0 on the identical development panel:

```bash
bash experiments/conditioned_vcg/E215_e191_long_exploration_drift_pilot_274k_v1/run.sh analyze
```

If the curve remains stable through 120, resume first to 220 and then to 260.
The latter provides two validation points after the new epsilon floor:

```bash
MPLCONFIGDIR=/tmp/e215-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E215_e191_long_exploration_drift_pilot_274k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 260 --device cuda
```

This is a development diagnostic.  It does not open the sealed test panel and
does not supersede E191 evidence.
