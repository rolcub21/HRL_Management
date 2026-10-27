# E116: matched E115 operator-cost baselines

E116 runs Dynamic PSLAP and GA on the exact five development instances used by
E115 validation: seeds 112800–112804, empty 8×8, 36 jobs, release gap 30, and
the same frozen Poisson-80 job durations. Every completed trajectory is scored
under E115's objective:

`J = 2T + E + 0.1L + 10R`

where `T` and `E` are release-clock tardiness and earliness beyond ±20, `L` is
all non-WAIT primitive actions, and `R` is physical rehandles. Completion is a
hard requirement; incomplete trajectories do not receive a comparable `J`.

GA uses the established four frozen stochastic replicates. Replicates are
averaged within each instance, then the five instances are weighted equally.
Analysis compares the fixed baselines with every E115 validation checkpoint,
so it does not select a lucky checkpoint from the reused development panel.

Run the resumable full comparison:

```bash
bash experiments/conditioned_vcg/E116_e115_matched_operator_baselines_180k_v1/run.sh run-all
```

Inspect progress or regenerate analysis after E115 reaches more checkpoints:

```bash
bash experiments/conditioned_vcg/E116_e115_matched_operator_baselines_180k_v1/run.sh status
bash experiments/conditioned_vcg/E116_e115_matched_operator_baselines_180k_v1/run.sh analyze
```

Each coordinate is saved separately. Re-running `run-all` reuses completed
coordinates. E116 performs no training and does not open an independent test
panel; it is a development comparison only.

