# VCG-Dense V1.2 development result

## Scope

These results are **development evidence only**. Validation panel A and
EpisodeInstance panel B (`80000`--`80029`) have both influenced the proposed
selection rule. No sealed test panel was opened, and neither the post-hoc nor
the guarded checkpoint set is deployment eligible.

## What the trajectory diagnostic established

The three V1.1 training trajectories contain policies that trade a small
increase in delivery MAE for substantially fewer relocations. Comparing the
final episode-500 policies with the original dense-return-selected policies
gave:

| Policy group | MAE | Relocations / 100 deliveries |
|---|---:|---:|
| V1.1 selected | 11.403 | 23.889 |
| Episode 500 diagnostic | 12.046 | 19.028 |

The paired point contrast was `+0.643` MAE and `-4.861` relocations per 100
deliveries. This is evidence of representational capacity, not a deployment
comparison: episode 500 was diagnostic, and seed 0 was an exact duplicate of
its selected checkpoint.

## Why the first V1.2 point selector was rejected

The first post-hoc rule selected episodes `475/425/500` by minimizing
validation relocations subject to a validation MAE tolerance of `+2.0`. On
panel B it reduced relocations by `9.028/100` on average, but seed 1 incurred
`+2.433` MAE relative to its V1.1 reference. The rule therefore did not preserve
the fixed timing tolerance for every independent training run.

## Guarded V1.2 development proposal

The guarded rule keeps the ranking fixed on validation panel A and traverses
that ranking on panel B. It accepts the first strict/full checkpoint whose
paired panel-B MAE cost is at most `+2.0`; the V1.1 reference is the mandatory
fallback. A candidate must also have a strictly positive paired relocation
saving on panel B.

It selected episodes `475/500/500`:

| Model seed | Episode | MAE cost vs V1.1 | Relocation saving / 100 |
|---:|---:|---:|---:|
| 0 | 475 | +1.117 | 2.500 |
| 1 | 500 | +0.800 | 6.250 |
| 2 | 500 | +1.129 | 8.333 |
| Equal-seed mean | -- | **+1.015** | **5.694** |

The aggregate guarded point is:

| Metric | Guarded V1.2 | V1.1 selected | Enhanced complete GA |
|---|---:|---:|---:|
| MAE | 12.418 | 11.403 | 14.758 |
| Relocations / 100 | 18.194 | 23.889 | 17.500 |
| Dense rescored return | 186.017 | 198.961 | 146.078 |
| First-two MAE | 21.728 | 19.750 | 28.983 |

Relative to enhanced complete GA, the guarded learned policy has `2.340`
lower MAE and `39.938` higher dense return, but `0.694/100` more relocations.
This is a timing--relocation trade-off, not dominance.

## Frozen next decision

The next valid step is a prospective replication with fresh training seeds and
fresh A/B selection panels. The operational rule succeeds only if all selected
arms are strict/full, at least two of three seeds select a non-reference
checkpoint, and the equal-seed panel-B relocation saving is positive. The
untouched final panel must then show an equal-seed MAE point cost no greater
than `+2.0` and positive relocation saving. Statistical noninferiority may be
claimed only if the prespecified one-sided 95% upper confidence bound is also
within `+2.0`.

Primary artifacts:

- `results/vcg-dense-v1-1-pareto-development-30seed/pareto-report.json`
- `results/vcg-dense-v1-2-posthoc-development-30seed/posthoc-report.json`
- `results/vcg-dense-v1-2-posthoc-development-30seed/guarded-selection-report.json`
- `results/vcg-dense-v1-2-posthoc-development-30seed/vcg-dense-v1-2-guarded-pareto.png`
