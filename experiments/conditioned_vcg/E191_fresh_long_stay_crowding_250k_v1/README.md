# E191 — fresh E189 architecture on long-stay workloads

E191 keeps the E189 network topology, eight queue-context inputs, objective,
certification, mask-only progress guard, and exact acceleration paths. Every
learned tensor, the optimizer, replay, and counters start fresh.

The only workload family has an empty 8×8 yard, 35 storage cells, 36 jobs, and
Poisson-700 storage durations. Training seeds are balanced across nominal
steady-state expected occupancy targets of 20%, 50%, and 80%, implemented by
deterministic arrival gaps of 100, 40, and 25 primitive steps. “Occupancy” is
not initial yard fill. Because episodes ramp from empty and drain at the end,
the report distinguishes the nominal target from achieved mean and peak
inventory.

The current E189 Defer rule is deliberately unchanged. It waits at most ten
steps per decision and permits sixteen consecutive Defers. The report retains
guard/action/timing telemetry so any binding of that inherited limit is
visible rather than silently repaired in the scenario experiment.

Run the non-episodic checks first:

```bash
bash experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/run.sh \
  self-check
```

Use the first three resumable episodes as the runtime smoke pilot:

```bash
MPLCONFIGDIR=/tmp/e191-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 3 --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e191-fresh-long-stay-crowding-250k-v1/seed-0-run.log
```

If all three complete normally, resume the same output to episode 300:

```bash
MPLCONFIGDIR=/tmp/e191-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 300 --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e191-fresh-long-stay-crowding-250k-v1/seed-0-run.log
```

Validation runs every 20 episodes on three development days at each workload
level. The sealed test panel is not opened. The 9,000-step limit is an error
cap, not an action rule.
