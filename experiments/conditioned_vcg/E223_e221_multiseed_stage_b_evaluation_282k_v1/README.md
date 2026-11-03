# E223: E221 multiseed Stage-B evaluation

E223 authenticates the completed round-4 E221 heads for model seeds 1 and 2
and evaluates each independently on the exact three development instances used
by E220 seed 0.  At each workload it compares the learned future-rehandle head
at lambda 10, 20, and 40 with the exact immediate-only controller at lambda 20
and 40.  Stage A and the Stage-B heads are frozen throughout.

The two routes may run concurrently on separate GPUs:

```bash
MPLCONFIGDIR=/tmp/e223-s1-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E223_e221_multiseed_stage_b_evaluation_282k_v1/run.sh \
  evaluate-seed --model-seed 1 --device cuda

MPLCONFIGDIR=/tmp/e223-s2-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E223_e221_multiseed_stage_b_evaluation_282k_v1/run.sh \
  evaluate-seed --model-seed 2 --device cuda
```

After both finish:

```bash
bash experiments/conditioned_vcg/E223_e221_multiseed_stage_b_evaluation_282k_v1/run.sh analyze
```

Every episode row is durable and authenticated, so an interrupted route resumes
without repeating completed episodes.  E223 is development-only and opens no
test panel.

