# E243: matched-ranker 5x5 prefilled certification curve

E243 is the expanded frozen certification-mechanism evaluation. It starts
E195's 5x5 yard with 4, 5, 6, or 7 of its eight cells occupied and uses 25
predeclared days at every fill level. Every initial state must have an exact
completion witness before any controller is evaluated.

Four arms use the same simulator and macro executor:

1. frozen VCG with the exact SAFE filter;
2. the same VCG critic over the physically executable frontier;
3. one fixed Dynamic-style ranker over the exact SAFE frontier;
4. the identical Dynamic-style ranker over the physical frontier.

The Dynamic arms instantiate the same class and authenticated ranker digest.
They differ only in the candidate frontier returned by the certification
wrapper. This repairs E237's discrepancy: E237 compared the historical option
scheduler with a separate certified adaptation and therefore was not a strict
filter-only Dynamic pair.

Run the development integration gate and prepare the witnessed panel:

```bash
MPLCONFIGDIR=/tmp/e243-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E243_e195_expanded_prefilled_curve_302k_v1/run.sh \
  gate --device cuda

MPLCONFIGDIR=/tmp/e243-mpl \
bash experiments/conditioned_vcg/E243_e195_expanded_prefilled_curve_302k_v1/run.sh \
  prepare --device cpu
```

Only after both pass, evaluate all 400 paired episodes:

```bash
mkdir -p results/vcg-conditioned-e243-e195-prefilled-certification-curve-302k-v1

MPLCONFIGDIR=/tmp/e243-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E243_e195_expanded_prefilled_curve_302k_v1/run.sh \
  evaluate --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e243-e195-prefilled-certification-curve-302k-v1/run.log
```

This does no training and does not open a protected test panel. The frozen
seed-0 source bundle is distributed separately from Git history and must be
extracted at the repository root before running the gate.
