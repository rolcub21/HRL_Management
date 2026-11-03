# E207 — immediate-only Stage-B ablation

E207 removes E204's learned future-rehandle prediction while retaining the
same frozen E191 critic, certified frontier, selector, guard, instances, and
centered price correction:

```text
immediate-only: Q_E191 - 0.01*(lambda-10)*I_reconfigure
E204 learned:   Q_E191 - 0.01*(lambda-10)*(I_reconfigure + N_future_hat)
```

It evaluates only `lambda=20` and `lambda=40` on E205's exact three instances.
`lambda=10` is the common E205/E191 reference and requires no duplicate run.
The analysis compares paired `J_lambda` and Pareto dominance in timing cost
versus rehandles. This is a development diagnostic, not paper evidence.

```bash
cd /home/ai_diagnosis/HRL_Management
set -o pipefail

MPLCONFIGDIR=/tmp/e207-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E207_e204_immediate_only_ablation_266k_v1/run.sh \
  run-all --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e207-e204-immediate-only-ablation-266k-v1/run.log

MPLCONFIGDIR=/tmp/e207-mpl \
bash experiments/conditioned_vcg/E207_e204_immediate_only_ablation_266k_v1/run.sh analyze
```
