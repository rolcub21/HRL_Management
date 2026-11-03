# E222 — finalized long-exploration VCG versus matched 8×8 heuristics

E222 is the fresh paper panel for the finalized long-exploration Stage A
models.  It compares E215 seed 0 (episode 300) and E218 seeds 1/2 (episodes
280/220) with matched Dynamic PSLAP and rolling GA.

The panel contains 30 previously unused days: ten each at nominal 20%, 50%,
and 80% occupancy.  Duration streams are paired across workload levels.  All
methods use the same environment, macro executor, observable information, and
objective.  GA has four nuisance replicates per day; Dynamic has one.  Strict
completion is reported before cost.

Before E222 materializes any test instance, a live development-only gate runs
VCG, Dynamic, and GA end to end on an existing heavy development day.  The
panel refuses to prepare until that gate passes.

## Run order

```bash
MPLCONFIGDIR=/tmp/e222-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E222_e215_e218_fresh_matched_8x8_panel_281k_v1/run.sh gate

bash experiments/conditioned_vcg/E222_e215_e218_fresh_matched_8x8_panel_281k_v1/run.sh self-check

# These routes may be launched separately after self-check.
bash experiments/conditioned_vcg/E222_e215_e218_fresh_matched_8x8_panel_281k_v1/run.sh \
  run-method --method e191 --model-seed 0 --device cuda
bash experiments/conditioned_vcg/E222_e215_e218_fresh_matched_8x8_panel_281k_v1/run.sh \
  run-method --method e191 --model-seed 1 --device cuda
bash experiments/conditioned_vcg/E222_e215_e218_fresh_matched_8x8_panel_281k_v1/run.sh \
  run-method --method e191 --model-seed 2 --device cuda
bash experiments/conditioned_vcg/E222_e215_e218_fresh_matched_8x8_panel_281k_v1/run.sh \
  run-method --method dynamic_pslap --device cuda
bash experiments/conditioned_vcg/E222_e215_e218_fresh_matched_8x8_panel_281k_v1/run.sh \
  run-method --method ga --device cuda

bash experiments/conditioned_vcg/E222_e215_e218_fresh_matched_8x8_panel_281k_v1/run.sh analyze
MPLCONFIGDIR=/tmp/e222-mpl .venv/bin/python -m \
  experiments.conditioned_vcg.E222_e215_e218_fresh_matched_8x8_panel_281k_v1.plot
```

The gate and test rows are resumable: existing authenticated rows are reused.
No training occurs in E222.
