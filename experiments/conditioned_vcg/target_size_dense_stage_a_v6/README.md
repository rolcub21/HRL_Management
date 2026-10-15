# Stage-A v6: direct graph tensors and a larger reconstruction cache

This exact acceleration keeps the v5 Double-DQN backup, frozen-target score
cache, and native chosen-action gradient path. It changes only immutable graph
input preparation:

- build node tensors directly from `RecoveryState`, without first rebuilding a
  Python `YardGraph` and an edge list that will be discarded;
- reuse a static geometry-feature template and topology for the 10×10 yard;
- retain up to 1,000,000 immutable state tensors instead of 100,000.

Learned embeddings are never cached. Online-network selection is recomputed
after every optimizer step. Frozen-target candidate scores are invalidated at
the existing target-network boundary, so the mathematical TD target is
unchanged.

The larger cache requires substantial host memory: a 10×10 float32 feature
tensor contains 6,000 payload bytes, so one million resident entries have a
lower-bound payload of about 6 GB plus Python and tensor overhead. The cache is
process-local and is not checkpointed.

Do not benchmark this on the same GPU as a running E79 continuation. If a
second GPU is idle, the read-only parity benchmark can be isolated there:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m \
  experiments.conditioned_vcg.target_size_dense_stage_a_v6.benchmark_graph_reconstruction \
  --source-output /home/bukuj/hrl-eval/results/vcg-conditioned-e79-original-base100-full-10x10-144k-v2 \
  --output results/vcg-stage-a-v6-graph-reconstruction-benchmark.json \
  --device cuda
```

## Resumable 10x10 continuation

`resumable_continuation.py` authenticates the episode-28 v5 checkpoint and
runs exactly one episode per invocation.  Every successful episode atomically
replaces `latest.pth`; a failed or interrupted episode leaves the preceding
checkpoint authoritative.  The continuation stops at episode 63, matching
the 8x8 budget of 4,032 exposed jobs (`63 * 64 = 4,032`).

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/path/to/HRL_Management \
  .venv/bin/python \
  experiments/conditioned_vcg/target_size_dense_stage_a_v6/resumable_continuation.py \
  train-next \
  --output results/vcg-e79-v6-direct-legal-overdue-progress-full-10x10-v1 \
  --device cuda
```

`scripts/run_e79_v6_to_63.sh` repeats that one-episode transaction until the
authenticated summary reaches episode 63.  It acquires an output lock so two
overnight loops cannot update the same checkpoint.

## Fixed development evaluation

The terminal evaluator uses the policy that was actually trained: greedy
direct-legal ranking plus the observable overdue-progress filter, without
recoverability certification.  It loads weights into a deployment-only clone
and does not load replay or advance learning state.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/path/to/HRL_Management \
  .venv/bin/python -u \
  experiments/conditioned_vcg/target_size_dense_stage_a_v6/evaluate_fixed_panel.py \
  --checkpoint-output \
    results/vcg-e79-v6-direct-legal-overdue-progress-full-10x10-v1 \
  --output \
    results/vcg-e79-v6-direct-legal-overdue-progress-full-10x10-v1/evaluation/fixed-development-episode63.json \
  --device cuda
```

The seeds `112100--112104` are the prespecified E79 development panel, not a
sealed test panel.  Because certification is disabled, neither successful
completion nor the compatibility certificate carried by a candidate is a
safety guarantee.  See `RESULTS.md` and `MODEL_ARTIFACT.json` for the completed
run, evaluation, and terminal-checkpoint identity.
