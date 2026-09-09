# Frozen damped conditioned-handling evaluation on 85k

This development diagnostic evaluates the fixed round-8 terminal produced by
the damped seed-0 convergence experiment. Evaluation is authorized only after
the training-only convergence gate passes.

## Work

The evaluator runs exactly 97 new simulator episodes:

```text
8 positive lambda values x 12 frozen EpisodeInstances = 96
1 lambda-zero parity sentinel                         =   1
                                                       ----
                                                         97
```

After the sentinel proves exact VCG 1.1 behavior, the existing twelve
lambda-zero rows are reused. All 108 paired rows for the old nested controller
are also reused from the authenticated parent ledger; no baseline is rerun.

The lambda grid remains:

```text
0, .025, .0375, .05, .075, .10, .125, .175, .20
```

The analysis compares the damped terminal with both the old nested controller
and the undamped round-4 terminal. Its Pareto calculation groups identical
MAE--rehandle coordinates before counting frontier points, preventing two
lambda values on one plateau from being treated as two distinct operating
points.

This is still an opened-panel seed-0 diagnostic. Even if it passes, seeds 1
and 2 must be trained and evaluated before an architecture-level conclusion.

## Run

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/vcg_v11_conditioned_handling_damped_evaluation_85k/run.sh run
```

The final report is written to:

```text
results/vcg-v1-1-conditioned-handling-seed0-damped-convergence/
  damped-evaluation-85k-report.json
```
