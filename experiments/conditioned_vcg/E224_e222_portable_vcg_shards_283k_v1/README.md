# E224: portable E222 VCG compute shards

E224 relocates only the expensive VCG seed-1/2 computations of the already
opened E222 panel.  It does not define, regenerate, or select a new panel.  The
E222 contract hash, instance-manifest hash, behavioral source hashes, selected
checkpoint hashes, and selected episodes are pinned before execution.  Absolute
checkpoint paths are resolved on the compute host only after their contents
authenticate.

Run the cheap gate before each independent route:

```bash
bash experiments/conditioned_vcg/E224_e222_portable_vcg_shards_283k_v1/run.sh \
  gate --model-seed 1
```

Then run seed 1 or 2.  The routes are independent and resumable:

```bash
MPLCONFIGDIR=/tmp/e224-s1-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E224_e222_portable_vcg_shards_283k_v1/run.sh \
  run-seed --model-seed 1 --device cuda
```

After a route completes, transfer only its
`rows/e191/model-seed-<seed>/` directory to the owner of the E222 panel.  The
owner imports those rows under the corresponding E222 directory and runs the
original E222 `analyze` command.  The E224 output's minimal contract is a shard
anchor, not a replacement for the full E222 contract.

