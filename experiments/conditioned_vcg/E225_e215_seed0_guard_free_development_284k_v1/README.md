# E225 — frozen E215 seed-0 guard-free development replay

E225 evaluates the finalized E215 seed-0 episode-300 checkpoint on its nine
development-validation days with every liveness intervention disabled. The
critic receives the complete exact-SAFE frontier. No Reconfigure candidate is
masked and no completion-witness action is injected or forced.

The four-value non-progress context remains in the frozen network input. It is
passive telemetry in this experiment: retaining it avoids an out-of-distribution
representation change while ensuring every executed action is selected by the
critic, except when certification leaves a literal singleton SAFE frontier.

The checkpoint already embeds its exact guard-on validation runs. E225
authenticates and copies those as the paired reference, then runs only the nine
new guard-free episodes.

```bash
MPLCONFIGDIR=/tmp/e225-mpl \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E225_e215_seed0_guard_free_development_284k_v1/run.sh \
  self-check

MPLCONFIGDIR=/tmp/e225-mpl \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E225_e215_seed0_guard_free_development_284k_v1/run.sh \
  run-all --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e225-e215-seed0-guard-free-development-284k-v1/run.log
```

Every guard-free row retains the decision-level exact-SAFE frontier size,
whether the historical guard would have masked or forced at that boundary, and
whether a selected Reconfigure was a certification singleton or a critic choice.

