# VCG dense timing--relocation Pareto development experiment

## Purpose

This experiment determines whether the dense-reward VCG V1.1 training runs
already contain policies with a better timing--relocation trade-off, or whether
the architecture needs a V2 change.

The experiment is a **development diagnostic**. Opening the panel makes it
development data; it must not later be presented as an untouched final test.

## Frozen panel and environment

- EpisodeInstance seeds: `80000` through `80029`, inclusive (30 instances).
- Every method receives the exact same saved instance and schedule.
- Yard: 5 x 5 cells; 8 blocks.
- Arrival parameters: lambda = 10 and mu = 80.
- Episode horizon: 2,000 environment steps.
- Dense delivery objective: base = 40, absolute-error coefficient = 1.5,
  outside-window coefficient = 0.5, target window = 20.
- VCG discount: gamma = 0.99.
- Learned evaluation is greedy, with no learning, teacher, critic, fallback, or
  future-schedule access. Frontier search is exact-full.
- Rolling-GA settings: seed base 310000, population 16, generations 10, egress
  weight 4.

The panel namespace was checked against prior training, validation,
development, and sealed test namespaces before it was selected.

## Compared policies

Primary learned procedure:

- finalized selected `best.pth` from model seeds 0, 1, and 2.

Capacity diagnostic:

- episode-500 `latest.pth` from model seeds 1 and 2.
- seed 0 selected episode 500, and its deployment weights must be identical to
  seed 0 `latest.pth`; the duplicate is authenticated but is not executed as an
  independent policy.

Deterministic systems, each executed once per instance:

- duration-aware + nearest-free;
- duration-aware + dynamic PSLAP;
- duration-aware + rolling 2009-style PSLAP GA;
- duration-aware + enhanced complete rolling GA.

The runner authenticates checkpoint roles, hashes, training contracts, and the
common environment/objective/search contract before opening the panel.

## Frozen estimands and decision rule

Safety is evaluated first. A policy is Pareto-eligible only when it completes
the full grid with 100% strict method success and no invalid/failure rows.

The primary lower-is-better Pareto coordinates are:

1. mean absolute delivery error (MAE);
2. physical storage relocations per 100 completed deliveries.

Dense return, first-two-delivery MAE, later-delivery MAE, tardiness, earliness,
target-window compliance, steps, and failures are guardrails or secondary
metrics. In this already-opened V1 development artifact, the legacy
`relocations` column counted completed storage-to-storage moves. It is a valid
physical operating-cost measure, but the historical `obstructive_moves` alias
does not establish that a VCG standalone Reconfigure macro cleared a target's
path. New prospective runs use the versioned decomposition
`physical_storage_relocation_decomposition_v1`: target-bound obstruction
clearances versus standalone reconfigurations, with the latter split by SAFE
direct-delivery availability. The frozen V1 CSV schema and artifacts are not
rewritten.

The EpisodeInstance is the paired scenario unit. The independent learned-method
replication unit is the training seed. The 3 x 30 learned rollouts must not be
reported as 90 independent training replications. Baseline rows are not
triplicated.

For the selected-to-final diagnostic, relocation saving is favorable when

`R_best - R_final > 0`,

and timing cost is favorable/non-inferior when

`MAE_final - MAE_best <= 2.0`.

The 2.0-step MAE margin is fixed before opening the panel (10% of the target
window). If final weights consistently save relocations within that margin,
the next step is a V1.2 objective/checkpoint-selection repair. If they do not,
the evidence favors a V2 architectural mechanism for route/space preservation.

## Uncertainty and interpretation

- Average the 30 paired instances within each model seed first.
- Expose all three learned seed estimates and summarize them with mean, sample
  SD, range, and a df=2 t interval conditional on the panel.
- Use a paired instance bootstrap only as uncertainty conditional on the three
  trained policies.
- A crossed training-seed x instance bootstrap is descriptive sensitivity, not
  a substitute for additional independent training seeds.
- Keep episode-500 policies explicitly diagnostic and deployment-ineligible.
- Do not use seed-level p-values or silently discard failed/non-finite rows.

The runner writes immutable instance hashes, checkpoint provenance, a
transactional row ledger, normalized CSV rows, an audit, and the frozen Pareto
report so the comparison can be resumed and reproduced.

## Run

From the repository root:

```bash
bash experiments/vcg_dense_pareto_development/run.sh
```

If the process is interrupted after artifacts have been written, rerun the
same command with `--resume-existing` appended to the Python invocation. Each
completed method-instance row is authenticated and reused rather than rerun.
