# D10 10×10 latency diagnostic

This development-only companion leaves the completed occupancy-extension
contract unchanged. It evaluates the same frozen seed-0 conditioned VCG at
`lambda=0.10` on the already frozen medium- and high-occupancy 10×10
EpisodeInstances. It performs no training.

The purpose is to distinguish two possible causes of slow certification:

- broad physical candidate frontiers;
- expensive individual exact certificate checks.

Every decision records storage occupancy, admitted workload, candidates by
macro family, cache hits/misses and certificate time by check role, and every
cache miss's search expansions and latency. The report also retains the 20
slowest checks and descriptive correlations with total frontier time.

Each episode has a predeclared one-hour wall-clock limit. Reaching it produces
a `censored_wall_clock` ledger and does **not** imply infeasibility, unsafe
behavior, or operational failure.

Run the pilots sequentially on otherwise idle hardware:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/development/D10_scalability_support_screen/run_latency_diagnostic.sh prepare
bash experiments/conditioned_vcg/development/D10_scalability_support_screen/run_latency_diagnostic.sh run-medium
bash experiments/conditioned_vcg/development/D10_scalability_support_screen/run_latency_diagnostic.sh run-high
```

`run-both` is a resumable shorthand for the last two commands, but the
separate commands make it easier to inspect the medium result before spending
time on the high-occupancy case.
