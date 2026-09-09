# Frozen-checkpoint lambda sweep

This diagnostic freezes the completed 300-episode unified VCG lambda-zero
checkpoint and changes only the evaluation-time merit weight:

`Q_op(s,c) - lambda * Q_N(s,c)` for lambda in 0, 0.05, 0.10, 0.20, 0.30.

There is no training or learning. Every value uses the same model weights,
the same opened 85000..85011 EpisodeInstances, and the same four action RNGs
per instance. The 240 rows therefore isolate policy scalarization from the
training-path effects observed in the retrained sweep.

```bash
bash experiments/vcg_unified_frozen_lambda_sweep_85k/run.sh prepare
bash experiments/vcg_unified_frozen_lambda_sweep_85k/run.sh run
```

The run is resumable by completed lambda ledger. It writes
`frozen-lambda-comparison.json`. This is development-only and does not access
86xxx or make a budget/KKT claim.

## Recorded outcome

All 240 rows were safe and complete with identical weights across lambda
values. The development point-estimate frontier contained `lambda=0` and
`lambda=0.05`: the latter changed rehandles/100 from 1.56 to 0.52 and MAE from
21.96 to 22.44. Its paired rehandle difference was -1.04 with nominal 95% CI
`[-2.02, -0.06]`. This post-hoc development result selected `0.05` for the
predeclared multi-seed check; it is not itself confirmation evidence.
