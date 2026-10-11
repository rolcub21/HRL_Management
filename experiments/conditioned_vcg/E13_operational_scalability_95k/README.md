# E13 — Operational and decision-space scalability

E13 is the final frozen-policy Block 6 panel. It evaluates the same seed-0
conditioned VCG controller at `lambda=0.10` without training, scale-specific
tuning, or changed normalization. Every row uses the E14 order-preserving path
cleanup, timing-invariant certificate cache, and the D12 constructive
relocation-family strategy. A failed family construction falls back to native
exact search under the same declared budget.

The instances are the authenticated D10 occupancy-extension instances. E13
does not generate or select new workloads after observing outcomes.

## Frozen panel

The main panel is the 3x3 factorial combination of 5x5, 8x8, and 10x10 yards
with low, medium, and high initial usable-storage occupancy. All three frozen
instance seeds `95100--95102` are used: 27 rows.

The 18 companion rows isolate:

- geometry at fixed initial workload (`K=6`, `N=8`);
- total episode length at fixed 8x8 medium initial occupancy (`N=20,36,54`,
  with `N=20` supplied by the main panel);
- 6x10 versus 10x6 aspect ratio at matched capacity and occupancy.

Each row records initial, time-weighted mean, and peak usable-storage
occupancy; admitted workload; strict completion and quality metrics; physical
and certified candidates by action type; UNKNOWN and empty-frontier rates;
native searches by role; D12 attempts, proofs, misses, and construction cost;
neural and verifier timing; warm/cold latency; memory; and exceedance of the
predeclared 0.1, 1, and 5 second computational thresholds. The 1-second value
is a reproducible audit threshold inherited from D10, not a claimed industrial
service-level requirement.

Quality metrics are never used to conceal incomplete rows. The report retains
complete-case values with their sample size, but the main coordinate metric is
suppressed unless all three required rows strictly complete.

## Run order

Runs are CPU-only and each launcher invocation executes one episode per Python
process so cold-start and peak-memory measurements remain interpretable. Do
not run them concurrently with another timing experiment.

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E13_operational_scalability_95k/run.sh prepare
bash experiments/conditioned_vcg/E13_operational_scalability_95k/run.sh run-main-pilot
```

After inspecting the nine-row seed-95100 pilot, complete the main panel and
companions. Existing ledgers are authenticated and skipped, so these commands
are resumable:

```bash
bash experiments/conditioned_vcg/E13_operational_scalability_95k/run.sh run-main
bash experiments/conditioned_vcg/E13_operational_scalability_95k/run.sh run-companions
```

One coordinate can be run directly:

```bash
bash experiments/conditioned_vcg/E13_operational_scalability_95k/run.sh \
  run-one size_10x10_occ_high 95101
```

The complete analysis writes `e13-report.json` and
`e13-companion-table.md` under
`results/vcg-conditioned-e13-scalability-95k-v2/`.

Each episode has a predeclared 1,800-second wall limit. A timeout is recorded
as censoring, not infeasibility; UNKNOWN is rejection under a finite search
budget, not proof that the physical state is unrecoverable.
