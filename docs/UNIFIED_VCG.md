# Unified VCG: current experiment program

This page is the short index for the unified viability-constrained graph (VCG)
work. The version-specific VCG documents remain the source for historical
controller development; the paper-facing endpoint is one controller with an
optional handling-cost term.

## Paper-facing variants

Both variants use the same safe candidate set, graph encoder, two-headed
critic, Hold semantics, SMDP policy mechanism, and frozen checkpoint weights:

```text
VCG                      M(s,c) = Q_op(s,c)                 (lambda = 0)
VCG + handling cost      M(s,c) = Q_op(s,c) - 0.05 Q_N(s,c)
```

`Q_op` estimates operational value and `Q_N` estimates future physical
rehandles. The viability layer still decides which candidates are allowed;
the handling weight only changes the ranking within that safe set. The final
comparison therefore concerns two operating points of one architecture, not
two independently trained proposed methods. `VCG 1.1` and `VCG 2.3` are useful
development-history labels, not the preferred final names.

The selected `lambda=0.05` is a fixed evaluation-time scalarization. It is not
presented as a converged active-budget or KKT solution.

## Evidence lifecycle

| Stage | What changed | Panel and scale | Scientific role |
|---|---|---|---|
| [Initial lambda pair](../experiments/vcg_unified_lambda_pair_85k/README.md) | separately trained `lambda=0` and adaptive `B_N=20` arms | opened 85k development panel | established a trade-off but mixed scalarization with training path |
| [Three-seed lambda pair](../experiments/vcg_unified_lambda_pair_seed_stability_85k/README.md) | repeated the paired training on seeds 15--17 | opened 85k development panel | produced the frozen `lambda=0` checkpoints used below |
| [Budget sweep](../experiments/vcg_unified_budget_sweep_85k/README.md) | adaptive budgets 10, 8, and 5 | one development seed | all constrained terminal points were classified unconverged; no active-budget claim |
| [Retrained fixed-weight sweep](../experiments/vcg_unified_fixed_lambda_sweep_85k/README.md) | separate training paths for weights 0.05--0.30 | one development seed | exposed non-monotone, path-dependent training behavior |
| [Frozen-weight sweep](../experiments/vcg_unified_frozen_lambda_sweep_85k/README.md) | changed only evaluation-time weight | 240 rows on opened 85k | selected `lambda=0.05` as the small handling-cost candidate |
| [Frozen-weight seed stability](../experiments/vcg_unified_frozen_lambda_seed_stability_85k/README.md) | applied 0 and 0.05 to the same weights within seeds 15--17 | 288 inference-only rows on opened 85k | passed the development continuation rule; rehandles fell for 3/3 seeds |
| [Prospective confirmation](../experiments/vcg_unified_frozen_lambda_confirmation_87k/README.md) | tested the frozen two-arm hypothesis | 720 inference-only rows on 30 new 87k instances | passed both predeclared primary tests |
| Qualitative casebook | replayed three selected matched pairs | six inference-only replays | explanatory behavior views; not replacement confirmation rows |

The earlier 86k baseline panel was not reused for candidate selection or this
confirmation.

## Confirmed result

The confirmation averages four action RNGs within each
model-seed/`EpisodeInstance`, averages model seeds 15--17 equally within each
instance, and uses the 30 paired instances as the statistical units.

| Frozen policy | Return | MAE | Steps | Rehandles/100 | Within +/-20 |
|---|---:|---:|---:|---:|---:|
| `lambda=0` | 80.64 | 19.62 | 188.89 | 9.13 | 63.85% |
| `lambda=0.05` | 77.47 | 19.88 | 188.81 | 8.19 | 62.85% |
| Difference (`0.05 - 0`) | -3.17 | +0.26 | -0.09 | **-0.94** | -1.01 pp |

All 720 rows were strict-safe and complete. The simultaneous one-sided 95%
upper bound was `-0.118` for the rehandle difference and `0.675` for the MAE
difference, below the predeclared bounds of zero and `+2.0`, respectively.
Thus the predeclared aggregate rehandle-reduction and MAE-noninferiority tests
both passed. Rehandles improved for model seeds 15 and 17 but increased slightly
for seed 16, so the result is an aggregate effect rather than a per-seed
guarantee.

The concise supported claim is:

> With frozen weights, adding a handling weight of `lambda=0.05` reduced
> physical rehandles while preserving timing accuracy on the prospective
> 30-instance confirmation panel.

This evidence is specific to the tested 5x5 yard, eight blocks, arrival rate
10, Poisson duration mean 80, and the frozen model seeds. It does not establish
the same operating point for other yard sizes, block counts, arrival processes,
or storage-duration regimes.

## Confirmation lifecycle repair

The first confirmation execution exposed an inherited normalization helper
whose coordinate domain covered the old 12-instance development grid rather
than the predeclared 30-instance panel. The panel had already been materialized,
but zero evaluation rows and no panel metrics were persisted or reported before
the repair.

`run_vcg_unified_frozen_lambda_confirmation.py` installs the bound mechanical
repair and `normalization-grid-repair.json` records it. Only the accepted
instance-index domain changed. The frozen instances, checkpoints, policy RNG
formula, metrics, hypotheses, and success rule did not change. This repair
should accompany any report of the prospective result.

## Reproduction and qualitative views

Run the completed confirmation protocol (completed ledgers are reused):

```bash
bash experiments/vcg_unified_frozen_lambda_confirmation_87k/run.sh run
```

Replay and render the three explanatory side-by-side GIFs:

```bash
.venv/bin/python render_vcg_unified_behavior_gifs.py --device cuda
```

The renderer refuses to write a GIF unless each replay exactly matches its
saved confirmation row. The views align macro-decision index, while each panel
shows its own simulation time. The included cases illustrate handling
avoidance, a counterexample, and different timing behavior at equal realized
rehandles. MP4 files are convenience transcodes for pausing and seeking; the
GIFs and `behavior-casebook.json` are the renderer's authenticated outputs.

Generated reports and media live below `results/`, which is intentionally
ignored by Git. Experiment contracts and commands are documented in the linked
experiment READMEs.
