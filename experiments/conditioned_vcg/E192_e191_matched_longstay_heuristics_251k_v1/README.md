# E192 — matched long-stay heuristics for E191

E192 evaluates plain Dynamic PSLAP and rolling GA on the exact nine E191
development EpisodeInstances.  It runs no learning and opens no sealed test
panel.  The baseline rows are checkpoint-independent and are reused whenever
E191 adds a new greedy validation checkpoint.

The panel is workload balanced:

- three nominal 20% days: gap 100, mean storage duration 700;
- three nominal 50% days: gap 40, mean storage duration 700;
- three nominal 80% days: gap 25, mean storage duration 700.

Dynamic is deterministic and runs once per instance.  GA runs four replicates
per instance and is averaged within instance before the nine paired instances
are aggregated.  The total is 45 episodes.  Every trajectory uses E191's
9,000-step error cap, storage-clock objective, full admission wait, frozen
arrival schedule and frozen duration stream.  The historical short-stay
defaults in the inherited baseline CLI are deliberately overridden by the
authenticated E191 EpisodeInstance.

The output reports strict completion before quality, J and all components,
storage and release timing, rehandles, achieved mean/steady/peak occupancy,
external queue and transporter utilization.  Certified Dynamic is excluded;
it remains a separate slower arm.

## Commands

Preflight without running an episode:

```bash
bash experiments/conditioned_vcg/E192_e191_matched_longstay_heuristics_251k_v1/run.sh \
  self-check
```

Run the complete resumable CPU panel:

```bash
set -o pipefail
MPLCONFIGDIR=/tmp/e192-mpl \
bash experiments/conditioned_vcg/E192_e191_matched_longstay_heuristics_251k_v1/run.sh \
  run-all --device cpu 2>&1 | \
tee -a results/vcg-conditioned-e192-e191-matched-longstay-heuristics-251k-v1/run.log
```

The runner writes each coordinate immediately.  Repeating `run-all` reuses
completed rows.  To run the methods separately:

```bash
bash experiments/conditioned_vcg/E192_e191_matched_longstay_heuristics_251k_v1/run.sh \
  run-method --method dynamic_pslap --device cpu

bash experiments/conditioned_vcg/E192_e191_matched_longstay_heuristics_251k_v1/run.sh \
  run-method --method ga --device cpu
```

Refresh comparisons after any later E191 validation checkpoint without
rerunning a baseline episode:

```bash
bash experiments/conditioned_vcg/E192_e191_matched_longstay_heuristics_251k_v1/run.sh \
  analyze
```

