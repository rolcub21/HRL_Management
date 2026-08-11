# Unified VCG: current experiment program

This page records the paper-facing viability-constrained graph (VCG) family
and separates it from the versioned controllers that led to it.

## Current architecture

The current method starts from the authenticated VCG 1.1 operational
controller and adds a detached handling-cost predictor. For each frozen model
seed, deployment uses

```text
VCG                       original VCG 1.1 selector              (lambda = 0)
VCG + handling            M(s,c) = Q_op(s,c) - lambda Q_N(s,c)   (lambda > 0)
```

The hard viability layer still determines which candidates may be selected.
`Q_op` and the original graph/action encoders are frozen. `Q_N` consumes
detached features from that controller and is trained separately from
historical replay; its optimizer cannot update `Q_op`. At `lambda=0`, the
wrapper delegates directly to the original VCG 1.1 `select()` implementation
and does not call the cost head. Positive lambda values retain the same safe
frontier and VCG 1.1 mode-selection semantics, changing only the candidate
merit.

Here `lambda` is the handling preference (`lambda_h`), not the environment's
arrival-rate parameter. This is an augmentation of VCG 1.1, not teacher-student distillation and not
VCG 2.3 retraining. The current cost head is a historical-behavior Monte Carlo
handling predictor rather than a converged constrained-RL critic; lambda is an
inference-time preference, not a dual variable or KKT certificate.

## Frozen lambda family

The development sweep used the same three frozen operational checkpoints and
handling heads at

```text
lambda in {0, 0.025, 0.05, 0.10, 0.20}.
```

All five operating points advanced from the opened 85k development panel to a
new 30-instance 89k confirmation panel. The confirmation contained exactly
`3 model seeds x 5 lambda values x 30 EpisodeInstances = 450` deterministic,
strict-safe-complete rows. No training, checkpoint selection, or lambda
selection occurred on 89k.

| Frozen lambda | Dense return | MAE | Steps | Rehandles/100 |
|---:|---:|---:|---:|---:|
| `0` | **199.87** | **11.34** | 191.52 | 22.92 |
| `0.025` | 186.35 | 12.39 | 186.39 | 20.42 |
| `0.05` | 179.72 | 12.89 | 181.48 | 15.00 |
| `0.10` | 170.36 | 13.68 | 168.84 | 7.50 |
| `0.20` | 158.19 | 14.55 | **159.22** | **3.61** |

Rehandles decreased monotonically in the equal-seed aggregate. All five
sampled points were nondominated in the MAE-rehandles plane and each was
nondominated for at least two of the three frozen model seeds. This supports a
sampled operating frontier for one controller family: `lambda=0` prioritizes
timing and return, while larger values progressively prioritize handling.

The evidence remains specific to the tested 5x5 yard, eight blocks, arrival
rate 10 (exponential mean interarrival 0.1), Poisson stay mean 80, and the
three frozen model seeds.

## Matched 89k comparators

A secondary analysis evaluated the established comparators on the exact same
serialized 89k `EpisodeInstance`s. The plot retains only the two endpoint VCG
operating points.

| Method | Dense return | MAE | Rehandles/100 | Eligibility |
|---|---:|---:|---:|---|
| VCG (`lambda=0`) | **199.87** | **11.34** | 22.92 | 90/90 |
| VCG + handling (`lambda=0.20`) | 158.19 | 14.55 | **3.61** | 90/90 |
| Historical VCG 2.3 | 102.43 | 18.26 | 8.19 | 360/360 |
| Dynamic PSLAP | 24.03 | 22.68 | 3.75 | 30/30 |
| Capacity-aware GA | 139.23 | 15.11 | 14.58 | 120/120 |
| Kim2020 adaptation | -- | -- | -- | 446/450; suppressed |

At the point-estimate level, `lambda=0.20` is better than every eligible
comparator in return, MAE, and rehandles. Together, the two VCG endpoints are
the nondominated methods in the matched table. Kim2020 had four stochastic
rows in which its placements left inbound work infeasible with no strict
retrieval able to release capacity; its whole-method metrics are therefore
suppressed rather than computed from successful rows.

The comparator extension was specified after the 89k VCG confirmation outcome
was known. It is a matched post-confirmation secondary analysis, not a
preregistered statistical superiority test. The prospective claim is the
replication of the VCG lambda frontier itself; baseline dominance here refers
to matched 89k point estimates.

## Evidence lifecycle

The current nested-controller path is:

| Stage | Role |
|---|---|
| [Seed-0 nested pilot](../experiments/vcg_v11_nested_handling_85k/README.md) | established bit-exact `lambda=0` nesting and selected `lambda=0.025` for replication |
| [Three-seed replication](../experiments/vcg_v11_nested_handling_seed_stability_85k/README.md) | fitted independent handling heads for seeds 1 and 2 while keeping all operational controllers frozen |
| [Five-level development sweep](../experiments/vcg_v11_nested_lambda_frontier_85k/README.md) | tested the fixed lambda grid and authorized unseen confirmation |
| [89k frontier confirmation](../experiments/vcg_v11_nested_lambda_confirmation_89k/README.md) | confirmed all five operating points on 30 unseen instances |
| [89k matched comparators](../experiments/vcg_v11_nested_all_baselines_89k/README.md) | added historical VCG 2.3, Dynamic PSLAP, capacity-aware GA, and Kim2020 on the same instances |

The earlier V2.3-based unified program remains useful development history. It
tested adaptive budgets, separately trained fixed weights, frozen-weight
sweeps, seed stability, and a prospective 87k `lambda=0` versus `lambda=0.05`
confirmation. Those experiments motivated the handling mechanism, but their
`lambda=0` controller did not reproduce VCG 1.1. The nested VCG family above
replaced that architecture as the paper-facing endpoint.

## Reproduction

From the repository root, completed ledgers are authenticated and reused:

```bash
bash experiments/vcg_v11_nested_lambda_confirmation_89k/run.sh analyze
bash experiments/vcg_v11_nested_all_baselines_89k/run.sh plot
```

A clean clone must first be supplied with the authenticated checkpoints, cost
heads, and result manifests named in the experiment READMEs; these generated
artifacts are intentionally not stored in Git.

To execute missing rows instead, use the corresponding `run` or `run-all`
command documented in each experiment README. Generated reports, checkpoints,
and figures live below `results/`, which is intentionally ignored by Git.

The qualitative filmstrips and replay utilities describe the earlier
V2.3-based mechanism and remain post-hoc illustrations rather than evidence
for the nested VCG checkpoints.
