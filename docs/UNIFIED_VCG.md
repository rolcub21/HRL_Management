# Unified VCG: current experiment program

This page indexes the paper-facing viability-constrained graph (VCG) family
and separates it from the versioned controllers that led to it.

## Current architecture

The current controller freezes the authenticated VCG 1.1 operational critic
and learns a lambda-conditioned estimate of future physical rehandles. For a
certified candidate `c`, deployment uses

```text
VCG              original VCG 1.1 selector                         lambda = 0
VCG + handling   Qop_V1.1(s,c) - lambda * QN(s,c,lambda)           lambda > 0
```

The hard exact-safe frontier and recovery-witness guard determine what may be
selected before preference affects ranking. The immediate Reconfigure cost is
encoded structurally; the learned nonnegative term estimates only later
rehandles under the lambda-induced continuation policy. At lambda zero the
wrapper bypasses the handling head and invokes VCG 1.1 exactly.

Here `lambda` is the handling preference, not the environment arrival-rate
parameter. It is an inference-time preference rather than a dual variable or
KKT certificate. Full equations and training details are in
[Preference-conditioned VCG](PREFERENCE_CONDITIONED_VCG.md).

## Final 90k comparison

The final CPU-v3 protocol evaluated ten fixed preferences and four baselines
on the same serialized `90000..90029` EpisodeInstances. It contains 1,860
ledgered evaluation rows and no training, checkpoint selection, lambda
selection, redraw, or complete-case filtering.

| Method | Dense return | MAE | Steps | Rehandles/100 | Eligibility |
|---|---:|---:|---:|---:|---:|
| VCG (`lambda=0`) | **185.36** | **12.38** | 188.49 | 24.58 | 90/90 |
| VCG + handling (`lambda=.05`) | 168.02 | 13.72 | 162.72 | 8.47 | 90/90 |
| VCG + handling (`lambda=.10`) | 161.34 | 14.17 | 153.24 | 3.06 | 90/90 |
| VCG + handling (`lambda=.20`) | 157.10 | 14.48 | **152.09** | **1.67** | 90/90 |
| Historical VCG 2.3 | 108.92 | 17.80 | 230.96 | 6.25 | 360/360 |
| Capacity-aware GA | 141.73 | 14.91 | 167.77 | 13.75 | 120/120 |
| Dynamic PSLAP | -- | -- | -- | -- | 28/30; suppressed |
| Kim2020 adaptation | -- | -- | -- | -- | 448/450; suppressed |

All 900 conditioned rows are strict-safe-complete. Nine of the ten conditioned
coordinates are on the point-estimate MAE--rehandles frontier; `.075` is
dominated by `.10` because of a small local nonmonotonicity.

The two endpoints give the intended interpretation:

- `lambda=0` is the best eligible operating point for return and timing MAE;
- `lambda=.20` reduces rehandles by 93.2% relative to lambda zero;
- `lambda=.20` improves on historical VCG 2.3 in return, MAE, steps, and
  rehandles, with all four paired nominal 95% intervals excluding zero.

Dynamic PSLAP failed to complete instances 90010 and 90012. Kim2020 produced
two blocked model-seed-0 rolls on instances 90004 and 90023. Their aggregate
metrics are therefore suppressed rather than averaged over successful rows.

## Development lifecycle

| Stage | Role |
|---|---|
| [D1: joint-vector screen](../experiments/conditioned_vcg/development/D01_architecture_screen/) | showed that conditioning can create intermediate behavior but did not preserve the operational endpoint |
| [D2: anchored residual](../experiments/conditioned_vcg/development/D02_operational_anchor_screen/) | preserved lambda zero but failed the high-lambda handling endpoint |
| [D3: seed-0 conditioned screen](../experiments/conditioned_vcg/development/D03_conditioned_seed0_screen/) | froze VCG 1.1 and learned only policy-conditioned future handling |
| [D4: full-update extension](../experiments/conditioned_vcg/development/D04_full_update_convergence/) | failed the declared convergence rule and motivated damping |
| [D5: damped convergence](../experiments/conditioned_vcg/development/D05_damped_convergence/) | stabilized fitted-policy iteration to its fixed round-8 terminal |
| [D6: seed-0 damped evaluation](../experiments/conditioned_vcg/development/D06_damped_seed0_evaluation/) | confirmed the fixed terminal before multi-seed replication |
| [D7: seed-1/2 training](../experiments/conditioned_vcg/development/D07_seed_replication/) | replicated the fixed recipe without opening an evaluation panel |
| [D8: seed-1 continuation](../experiments/conditioned_vcg/development/D08_seed1_continuation/) | applied the predeclared convergence-controlled continuation to round 10 |
| [D9: fixed merit bank](../experiments/conditioned_vcg/development/D09_fixed_merit_diagnostic/) | diagnosed `Qop`, `QN`, `lambda*QN`, and merit without affecting selection |
| [D10: scalability support](../experiments/conditioned_vcg/development/D10_scalability_support_screen/) | separated yard size, occupancy, candidate breadth, and per-check latency |
| [D11: shared-search audit](../experiments/conditioned_vcg/development/D11_shared_search_opportunity_audit/) | measured repeated intermediate-state work before selecting a reuse mechanism |
| [D12: relocation-family certification](../experiments/conditioned_vcg/development/D12_relocation_family_certification/) | accepted constructive amortization component: strict completion at both tested 10x10 occupancy settings, with 21,051 family proofs and zero misses |
| [E13: operational scalability](../experiments/conditioned_vcg/E13_operational_scalability_95k/) | frozen 3x3 scale-by-occupancy panel plus geometry, episode-length, and aspect-ratio companions with E14 cleanup and D12 enabled |
| [E14: certification scalability](../experiments/conditioned_vcg/E14_certification_scalability_95k/) | frozen five-budget sensitivity panel with fresh per-arm anchors; the 20,000-node reference reuses authenticated E13 rows |

The paper-facing result is [E1: prospective 90k benchmark](../experiments/conditioned_vcg/E01_benchmark_90k/).

## Predecessor nested controller

The earlier nested controller used a detached handling predictor fitted from
historical behavior rather than a policy-conditioned future-cost model. Its
five-point 89k confirmation established the central timing--handling mechanism
and exact lambda-zero nesting. The matched 89k comparator extension then
motivated the conditioned architecture.

Those results remain valid predecessor evidence:

- [89k frontier confirmation](../experiments/vcg_v11_nested_lambda_confirmation_89k/README.md)
- [89k matched comparators](../experiments/vcg_v11_nested_all_baselines_89k/README.md)

The older VCG 2.3-based unified program is also retained as development
history. Its lambda-zero controller did not reproduce VCG 1.1 and it is not a
separate proposed final method.

## Reproduction

From the repository root, completed final ledgers can be inspected and
reanalyzed with:

```bash
bash experiments/conditioned_vcg/E01_benchmark_90k/run.sh inspect
bash experiments/conditioned_vcg/E01_benchmark_90k/run.sh analyze
bash experiments/conditioned_vcg/E01_benchmark_90k/run.sh plot
```

A clean clone must first be supplied with the authenticated checkpoints,
conditioned terminals, and result manifests named in the experiment READMEs.
Generated artifacts live below ignored `results/` paths and are not stored in
Git.

The qualitative filmstrips remain post-hoc mechanism illustrations. The
robust-certificate experiments are a separate theoretical extension and do
not change the deterministic final90 performance claim.

## Scope

The final result is specific to the tested 5x5 yard, eight blocks, arrival rate
10, Poisson stay mean 80, and three frozen operational model seeds. A frontier
over other geometries, loads, arrival processes, or stay distributions needs a
separately frozen generalization experiment.
