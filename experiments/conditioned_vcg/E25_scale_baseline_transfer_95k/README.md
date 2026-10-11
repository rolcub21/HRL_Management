# E25 — Baseline transfer across the E13 scale grid

> **Superseded for cross-method claims.** An audit found that E25's baseline
> adapter starts the first `K` jobs in the inbound queue, whereas E13 starts
> them physically stored at time zero. Matching `EpisodeInstance` IDs did not
> match the initial states or placement-clock targets. The E25 cross-method
> figure and table are historical diagnostics, not paper evidence. Use the
> versioned [E25-v3 warm-start correction](../E25_v3_matched_warm_start_95k/README.md)
> after its full baseline confirmation completes.

E25 tests whether VCG's scale-dependent timing changes are specific to its
ranking/guard or shared by non-certified Table 3 baselines on exactly the same
workloads. It reuses the 27 completed VCG rows from E13 and newly evaluates:

- capacity-aware GA;
- duration-aware Dynamic PSLAP;
- the fully convolutional A3C adaptation.

The main grid is yard size `5x5`, `8x8`, and `10x10` crossed with low, medium,
and high initial occupancy. All methods receive the identical E13
EpisodeInstances (`95100..95102`) and the same three-cell exit geometry.

## Interpretation boundary

The A3C checkpoint remains frozen. Its fully convolutional network can accept
variable spatial dimensions, but geometry transfer and the scenario metadata
shift are enabled explicitly and recorded in every row. Default A3C loading
elsewhere remains fail-closed on geometry mismatch.

These configurations contain only two arriving jobs each; all other jobs are
initially stored. Therefore this is a matched **system-transfer comparison**,
not a pure yard-size intervention. Absolute timing trends combine geometry,
workload length, and changing initially-stored composition. The method-by-
configuration comparison is nevertheless informative because every method
receives the same workload.

VCG-PD is excluded from the main panel because it is another certified
VCG-family controller. E25 targets the independent non-certified controls that
can distinguish a common workload effect from a VCG-specific transfer issue.

## Replication and aggregation

- VCG: one reused E13 seed-0 controller row per EpisodeInstance;
- Dynamic PSLAP: one deterministic row per EpisodeInstance;
- capacity-aware GA: four optimizer seeds per EpisodeInstance;
- A3C: three model seeds by five policy RNGs per EpisodeInstance.

Nuisance/model replications are averaged within EpisodeInstance before the
three instances are averaged within each size-occupancy coordinate. Strict
completion is always displayed separately. MAE, rehandles, and steps are
conditional on strict completion and cannot conceal failed runs.

## Run order

Preparation and the three-run geometry pilot do not train any model:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E25_scale_baseline_transfer_95k/run.sh prepare
bash experiments/conditioned_vcg/E25_scale_baseline_transfer_95k/run.sh run-pilot
```

The complete panel contains 540 new rollouts and is resumable row by row:

```bash
bash experiments/conditioned_vcg/E25_scale_baseline_transfer_95k/run.sh run-all
bash experiments/conditioned_vcg/E25_scale_baseline_transfer_95k/run.sh analyze
```

The analysis produces a 4x3 paper figure with occupancy columns and rows for
strict completion, placement MAE, rehandles per 100 deliveries, and steps per
delivery. Lines encode methods and light bands show the range over the three
EpisodeInstance-level means.
