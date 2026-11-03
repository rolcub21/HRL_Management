# E214 — gated on-policy trace Stage B on 8×8

E214 is the final seed-0 repair pilot for the learned future-rehandle term. It
keeps frozen E191, certification, the progress guard, the operational objective,
and all validated acceleration mechanisms unchanged.

The repair replaces E208's one-step TD target with a round-frozen η-return:

```text
G_t = I(c_next) + (1-η) N_round_start(s_next,c_next,α) + η G_next
```

Here `η=0.99`. One frozen head controls all nine episodes in a round, covering
every workload × continuous α-band combination. Only that round's on-policy
episodes train the next head; their data are then discarded. Actual future
counts remain the primary held-out calibration measure.

Before expansion, E214 must pass all predeclared gates:

- held-out MC MAE beats the exact zero head;
- the exact E212 decision-7 replay avoids the old `accept:B3:6:6` choice;
- after forcing the complete E213 prefix, continuation is closer to the stable
  immediate-only outcome than to the 309-rehandle collapse;
- at least one paired λ=20/40 point improves on immediate-only;
- no paired point regresses `J_lambda` by more than 25%;
- every learned and causal episode completes strictly.

Prepare and train four resumable rounds:

```bash
cd /home/ai_diagnosis/HRL_Management
set -o pipefail

MPLCONFIGDIR=/tmp/e214-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E214_on_policy_trace_rehandle_stage_b_8x8_273k_v1/run.sh \
  prepare

MPLCONFIGDIR=/tmp/e214-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E214_on_policy_trace_rehandle_stage_b_8x8_273k_v1/run.sh \
  train-seed --through-round 4 --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e214-on-policy-trace-rehandle-stage-b-8x8-273k-v1/run.log
```

Then run the nine learned endpoint episodes and two causal replays. Existing
E207 immediate-only rows are reused exactly, so they are not rerun:

```bash
MPLCONFIGDIR=/tmp/e214-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E214_on_policy_trace_rehandle_stage_b_8x8_273k_v1/run.sh \
  evaluate --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e214-on-policy-trace-rehandle-stage-b-8x8-273k-v1/eval.log

MPLCONFIGDIR=/tmp/e214-mpl \
bash experiments/conditioned_vcg/E214_on_policy_trace_rehandle_stage_b_8x8_273k_v1/run.sh \
  analyze
```

This is a development-only controller diagnostic. It does not open a sealed
test panel and cannot become paper evidence merely by passing its gates.
