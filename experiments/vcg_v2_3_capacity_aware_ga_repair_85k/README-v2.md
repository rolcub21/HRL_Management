# Capacity-aware rolling-GA repair V2 (opened 85xxx development panel)

V2 preserves the four repaired assignment policies, source IDs, source
versions, objectives, and exact `4 x 12` opened-panel execution grid from V1.
It creates a fresh protocol/schema/output family solely to repair artifact
lifecycle validation.

V1 serialized ledgers with canonical JSON key sorting, while its reload
validator incorrectly expected `best_cost_components` mapping insertion order
to equal the objective's lexicographic order. V2 instead requires the exact
objective-specific component key set and separately authenticates
`objective_lexicographic_fields`; JSON object order is explicitly
non-semantic. The completed V1 directory is preserved as provisional input
and is never resumed or rewritten by this protocol.

This remains development-only on the already-opened `85000..85011` panel. It
does not open `86xxx` and does not authorize confirmatory or performance
claims.

## Commands

Authenticate sources and write only the fresh V2 contract/preflight:

```bash
bash experiments/vcg_v2_3_capacity_aware_ga_repair_85k/run-v2.sh prepare
```

After independent audit approval, execute the fresh exact 48-row grid:

```bash
bash experiments/vcg_v2_3_capacity_aware_ga_repair_85k/run-v2.sh run
```

Resume only V2 ledgers matching the exact V2 contract:

```bash
bash experiments/vcg_v2_3_capacity_aware_ga_repair_85k/run-v2.sh resume
```

Artifacts are isolated under
`results/vcg-v2-3-capacity-aware-ga-repair-v2-85k/`. The V1 provisional
directory `results/vcg-v2-3-capacity-aware-ga-repair-85k/` is not modified.

