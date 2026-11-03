# E203 — E191 matched 8x8 paper test panel

E203 opens the first locked 8x8 test panel after all E191 checkpoint selection
is complete. It contains 30 previously unused EpisodeInstances: ten paired
duration streams at each nominal 20%, 50%, and 80% workload.

Methods:

- the three immutable E202/E191 selected checkpoints, greedy and certified;
- matched Dynamic PSLAP;
- matched rolling GA with four nuisance replicates per instance.

All methods use the exact E191 dynamics, objective, finite horizon, arrivals,
and storage durations. Strict completion is compared before cost. Certified
Dynamic remains excluded as predeclared.

Preflight does not execute a test episode:

```bash
bash experiments/conditioned_vcg/E203_e191_matched_8x8_test_panel_262k_v1/run.sh self-check
```

Opening and running the complete panel:

```bash
MPLCONFIGDIR=/tmp/e203-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E203_e191_matched_8x8_test_panel_262k_v1/run.sh \
  run-all --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e203-e191-matched-8x8-test-panel-262k-v1/run.log
```

The 240 episode coordinates are saved independently and the command is fully
resumable. To split work, use `run-method --method dynamic_pslap`,
`run-method --method ga`, or `run-method --method e191 --model-seed N`.

After completion:

```bash
bash experiments/conditioned_vcg/E203_e191_matched_8x8_test_panel_262k_v1/run.sh analyze
.venv/bin/python -m \
  experiments.conditioned_vcg.E203_e191_matched_8x8_test_panel_262k_v1.plot
```

This writes the compact paper figure `e203-test-panel.png` and the broader
quality/objective/operations dashboard `e203-kpi-dashboard.png`.

