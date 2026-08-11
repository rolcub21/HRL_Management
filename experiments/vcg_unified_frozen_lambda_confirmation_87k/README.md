# Unified VCG frozen-lambda confirmation

This prospective inference-only experiment confirms the selected handling
augmentation on a new 30-instance panel:

- frozen model seeds 15, 16, and 17;
- the same episode-200 weights within each seed;
- `lambda=0` versus `lambda=0.05`;
- EpisodeInstances 87000..87029;
- four shared action RNGs per model/instance;
- 720 total rows, with no training.

`prepare` authenticates the selected candidate and freezes the hypothesis but
does not generate or inspect the 87k panel. `run` opens/materializes that panel,
evaluates all rows, and analyzes the result. Completed 120-row ledgers are
resumed without rerunning them.

The predeclared primary result requires all rows to be safe and complete, a
simultaneous one-sided 95% upper bound below zero for the rehandle difference,
and the corresponding MAE upper bound below the +2.0 noninferiority margin.

```bash
cd /path/to/HRL_Management
bash experiments/vcg_unified_frozen_lambda_confirmation_87k/run.sh prepare
bash experiments/vcg_unified_frozen_lambda_confirmation_87k/run.sh run
```

The final report is `results/vcg-unified-frozen-lambda-confirmation-87k/confirmation-report.json`.

## Recorded outcome

The predeclared result passed. All 720 rows were safe and complete.

| Frozen policy | Return | MAE | Steps | Rehandles/100 | Within +/-20 |
|---|---:|---:|---:|---:|---:|
| `lambda=0` | 80.64 | 19.62 | 188.89 | 9.13 | 63.85% |
| `lambda=0.05` | 77.47 | 19.88 | 188.81 | 8.19 | 62.85% |

For `lambda=0.05 - lambda=0`, the rehandle difference was -0.94 and its
simultaneous one-sided 95% upper bound was -0.118, below zero. The MAE
difference was +0.26 and its corresponding upper bound was 0.675, below the
predeclared +2.0 noninferiority margin. This supports an aggregate effect;
rehandles improved for seeds 15 and 17 but increased slightly for seed 16.

Lifecycle note: the first execution exposed an inherited 12-instance
normalization-coordinate bound after the 87k panel had been materialized.
Zero evaluation rows and no panel metrics were persisted or reported before
the repair. `normalization-grid-repair.json` binds the mechanical 30-index
correction; it does not change the frozen panel, checkpoints, policy RNGs,
metrics, hypotheses, or success rule.

## Qualitative behavior casebook

The confirmation rows can be replayed without training and rendered at
macro-decision boundaries:

```bash
.venv/bin/python render_vcg_unified_behavior_gifs.py --device cuda
```

The renderer covers handling avoidance, a counterexample, and different timing
behavior at equal realized rehandles. It writes three GIFs and
`behavior-casebook.json` below
`results/vcg-unified-frozen-lambda-confirmation-87k-behavior-view/`; every
replay must exactly reproduce its saved confirmation row before rendering.
These are explanatory selected cases, not additional confirmation evidence.
See [Unified VCG](../../docs/UNIFIED_VCG.md) for the complete evidence map and
interpretation limits.
