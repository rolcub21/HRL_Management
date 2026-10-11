# E25-v3 — Matched warm-start scale baselines

E25/E25-v2 used identical `EpisodeInstance` IDs but **different initial yard
states**. The VCG arm used E13's warm start: the first `K` jobs were already
stored at time zero. The baseline adapter used ordinary `SmallRoomsEnv.reset`,
which put all `K` jobs in the inbound queue. Because delivery targets are set
from placement time, the cross-method E25/E25-v2 figures and tables cannot
support relative timing, handling, or completion claims. Preserve them as
historical artifacts; do not use them as paper evidence.

E25-v3 reuses the 270 authenticated, frozen E25-v2 VCG rows and **reruns all
540 baseline rows** from E13's exact warm-start layout and clocks. It retains
the three E25 baseline adapters, checkpoints, seeds, replications, and
within-instance aggregation. No training occurs. Each new baseline ledger
records a checked time-zero state signature, initial occupancy, and initial
target range. Initial-state parity is a hard condition for the final report.
This is a matched **system** comparison, not an action-interface ablation:
the baselines retain their original duration-aware scheduler and may lack a
standalone reconfiguration action available to VCG. Their failures must be
reported separately from timing among completed runs.

The pilot covers 5×5 low and 10×10 high for one replicate of each baseline.
Completion is not required for an integration pilot: failure to complete a
matched workload is a legitimate outcome, whereas a reset mismatch is fatal.
In the first pilot, all three methods completed 5×5 low but halted at step 20
on 10×10 high with `scheduler_infeasible`, before any delivery. All six reset
audits passed. These pilot outcomes are diagnostic, not confirmation evidence.

```bash
bash experiments/conditioned_vcg/E25_v3_matched_warm_start_95k/run.sh prepare
bash experiments/conditioned_vcg/E25_v3_matched_warm_start_95k/run.sh run-pilot
```

After the pilot passes, run the confirmation sequentially on isolated CPU
resources. `run-all` resumes authenticated completed ledgers; no old E25
baseline ledger is loaded.

```bash
bash experiments/conditioned_vcg/E25_v3_matched_warm_start_95k/run.sh run-all
bash experiments/conditioned_vcg/E25_v3_matched_warm_start_95k/run.sh analyze
```

The same work can be split by method using `run-method` with method identifiers
from E25: `duration_aware_dynamic_pslap`,
`duration_aware_pslap_ga_2009_rolling_capacity_aware_partial`, and
`kim2020_a3c_spatial_adapted__stochastic`. Do not run methods concurrently if
episode wall time is to be reported comparably.

The corrected figure and table are written only under
`results/vcg-conditioned-e25-v3-matched-warm-start-95k/`. The final analysis
refuses fewer than 540 new baseline rows; `analyze-partial` is diagnostic and
explicitly not paper evidence. The comparison remains post-hoc and the VCG
rows are reused controls, not independent replications.
