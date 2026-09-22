# E3 scale-stress extension

This prospective extension asks whether E3's recoverability certificate still
prevents self-blocking under larger, deliberately pressured yards. It does not
replace the completed 5x5 E3 benchmark.

Two supported geometries are tested:

| Scenario | Usable capacity | Initially stored | Synchronized total jobs |
|---|---:|---:|---:|
| `stress_8x8_saturation_burst` | 35 | 25 (71.4%) | 35 (100% of capacity) |
| `stress_10x10_saturation_burst` | 63 | 44 (69.8%) | 63 (100% of capacity) |

All jobs arrive at time zero. The initial placements preserve the access
corridor used in E13; the burst then tests whether the controller consumes the
remaining working slack before stored jobs mature. Ten instance seeds per
geometry are serialized before any rollout. No instance is retained or removed
based on an observed certification intervention.

Within every frozen instance, the seed-0 conditioned controller at
`lambda=.10` is run twice:

- `physical_feasibility_only`: exact certificates are shadow labels and do not
  control eligibility;
- `recoverability_certified`: only exact-SAFE candidates can reach the same
  learned selector.

Both arms use the same frozen weights, E14 path cleanup/timing-invariant cache,
D12 relocation-family certificates, selector, guard, search budget, and
executor. `UNSAFE` and budget-dependent `UNKNOWN` outcomes remain separate.

This is a **designed saturation stress test**, not an estimate of self-blocking
prevalence in representative industrial workloads. Only model seed 0 is used,
because that is the frozen controller whose larger-layout compatibility was
established in E13.

## Run order

Preparation is quick and performs no rollout:

```bash
bash experiments/conditioned_vcg/E03_certification_ablation_90k/run_scale_stress.sh prepare
```

The pilot runs one paired instance on each geometry: four inference-only
episodes, no training.

```bash
bash experiments/conditioned_vcg/E03_certification_ablation_90k/run_scale_stress.sh run-pilot
```

Inspect the partial report before committing to the full panel:

```bash
bash experiments/conditioned_vcg/E03_certification_ablation_90k/run_scale_stress.sh analyze-partial
```

The full confirmation is 20 pairs / 40 episodes. Existing pilot ledgers are
authenticated and reused:

```bash
bash experiments/conditioned_vcg/E03_certification_ablation_90k/run_scale_stress.sh run-all
```

For controlled scheduling, run one geometry or one pair at a time:

```bash
bash experiments/conditioned_vcg/E03_certification_ablation_90k/run_scale_stress.sh run-scenario stress_8x8_saturation_burst
bash experiments/conditioned_vcg/E03_certification_ablation_90k/run_scale_stress.sh run-pair stress_10x10_saturation_burst 97100
```

Each episode has a predeclared one-hour wall limit. A limit is recorded as
censoring, not as proof of infeasibility. Secondary timing/handling metrics are
reported only for strict completers.

Outputs are written to
`results/vcg-conditioned-e03-scale-stress-97k/`.
