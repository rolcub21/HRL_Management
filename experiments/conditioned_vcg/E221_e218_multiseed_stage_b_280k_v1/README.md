# E221: E218 multiseed Stage B

E221 authenticates the finalized E218 seed-1 and seed-2 development selections
and trains an independent frozen E214/E216 future-rehandle head on each one.
It uses exactly the same 36 Stage-B collection coordinates as E220 seed 0.
Only the frozen base policy, model seed, and deterministic head seed differ.

No `latest.pth` Stage-A trainer checkpoint is required.  E221 authenticates the
E218 root contract, each training contract and summary, and each finalized
`best.pth` before constructing the frozen Stage-A agent.

Run the two seeds independently (or concurrently on separate GPUs):

```bash
MPLCONFIGDIR=/tmp/e221-s1-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E221_e218_multiseed_stage_b_280k_v1/run.sh \
  train-seed --model-seed 1 --through-round 4 --device cuda

MPLCONFIGDIR=/tmp/e221-s2-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E221_e218_multiseed_stage_b_280k_v1/run.sh \
  train-seed --model-seed 2 --through-round 4 --device cuda
```

Each collection episode and round endpoint is durable.  Rerunning the same
command resumes without repeating completed episodes.  Endpoint comparison and
the three-seed Stage-B freeze are intentionally deferred to a versioned paired
development panel after both heads finish.
