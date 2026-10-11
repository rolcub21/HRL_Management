# E25-v2 — Full VCG preference family across scale

> **Cross-method comparison superseded.** E25-v2's 270 VCG rows share E13's
> warm start and remain valid for within-VCG preference analysis. Its 540
> reused E25 baseline rows start from an all-queued yard, so the combined
> figure/table cannot compare methods on the same initial task. See the
> versioned [E25-v3 correction](../E25_v3_matched_warm_start_95k/README.md).

E25-v2 corrects the single-operating-point limitation of E25. It evaluates
the frozen seed-0 VCG controller at all ten predeclared E1 deployment
preferences on the same 27 E13 size-by-occupancy EpisodeInstances:

`0, .025, .0375, .05, .075, .10, .125, .15, .175, .20`.

All 540 E25 baseline rows are authenticated reused controls. The extension
adds 270 VCG rollouts and does not train or tune any model. The 27 historical
E13 rows at `lambda=.10` are used only for a parity audit. This isolates
whether the scale result is common to the VCG preference family or specific
to the previously shown `.10` operating point.

The controller stack is unchanged: frozen model seed 0, E14 path cleanup and
timing-invariant cache, D12 relocation-family certificates, the exact checker,
and the deployed liveness guard. Each workload contains exactly two arriving
jobs; all remaining jobs are initially stored. Cross-size trends therefore do
not isolate geometry, but comparisons among methods and preferences within a
configuration remain matched.

E13 is consumed as a frozen artifact parent because later executor work changed
the live source hashes recorded by its historical contract. In particular, the
candidate executor was corrected after E13. E25-v2 binds the current E13 and
candidate executor sources, authenticates the frozen contract, manifest,
instances, and checkpoint identity, and executes **all ten** preferences under
that same corrected implementation. It does not mix historical E13 outcomes
into the new preference comparison.

The first two endpoint pilot rows were completed under an earlier E25-v2
development contract and remain under the original output directory. The
corrected confirmation uses a separate `-v2` output directory and contract.

## Run order

Preparation and the three-row endpoint and `.10` parity pilot perform no training:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E25_v2_full_vcg_family_scale_95k/run.sh prepare
bash experiments/conditioned_vcg/E25_v2_full_vcg_family_scale_95k/run.sh run-pilot
```

The complete new grid is resumable. Based on the authenticated E13 timings,
expect approximately 12--13 CPU-hours when run sequentially on the same
hardware:

```bash
bash experiments/conditioned_vcg/E25_v2_full_vcg_family_scale_95k/run.sh run-all
bash experiments/conditioned_vcg/E25_v2_full_vcg_family_scale_95k/run.sh analyze
```

Individual preference slices can be scheduled separately, for example:

```bash
bash experiments/conditioned_vcg/E25_v2_full_vcg_family_scale_95k/run.sh run-lambda 0.05
```

Analysis is paper evidence only after all 270 new rows are present. The figure
shows every VCG preference as a separate blue-shaded line, together with the
three authenticated E25 baseline families.
