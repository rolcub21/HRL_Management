# VCG constrained V2.2 development calibration

This runs the frozen policy/operator-consistent V2.2 protocol: 200 training
episodes and ten 48-rollout stochastic validation looks. It neither opens the
final 86xxx panel nor authorizes a performance claim.

From the repository root:

```bash
bash experiments/vcg_constrained_v2_2/run_development_calibration.sh
```

The output directory must be new. The run writes its authenticated contract,
fixed-instance manifest, complete per-look validation ledgers, audit-only
latest checkpoint, eligible best candidate when one exists, and final summary.

