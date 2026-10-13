# Stage-A v5: exact graph-input tensor cache

This acceleration retains the v4 Double-DQN backup, frozen-target score cache,
and native gradient path. It adds a bounded CPU cache of immutable graph input
tensors keyed by authenticated `RecoveryState` object identity. Learned graph
embeddings are never cached.

The cache stores node features and interns static topology. Batches are
assembled directly as tensors, bypassing repeated Python graph construction,
padding, and nested-sequence conversion. Entries are not checkpointed and are
reconstructed after every process start. The 100,000-entry LRU bound limits
host memory and records all hits, misses, and evictions in the training audit.

Before training, compare v4 and v5 on exactly the same saved replay batch:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m \
  experiments.conditioned_vcg.target_size_dense_stage_a_v5.benchmark_graph_cache \
  --source-output results/vcg-conditioned-e79-original-base100-full-10x10-144k-v2 \
  --output results/vcg-stage-a-v5-graph-cache-benchmark.json --device cuda
```
