# Prospective VCG-Dense V1.2 selection

This protocol prospectively tests the physical-relocation-aware checkpoint selector
without changing the VCG architecture, dense timing reward, discount, training
budget, or optimizer.

## Frozen design

- Fresh independent training seeds: 3, 4, and 5.
- Exact V1.1 dense-reward, gamma=0.99, 500-episode recipe.
- Twenty immutable model-only snapshots per seed, every 25 episodes.
- Disjoint training EpisodeInstance bases: 53M, 54M, and 55M.
- The original 79000--79019 validation clone is retained only to reproduce
  the same-trajectory frozen V1.1 reference selector.
- Panel A: EpisodeInstance seeds 83000--83029.
- Panel B: EpisodeInstance seeds 83030--83059.
- MAE noninferiority margin: +2.0 steps versus the same-seed frozen V1.1
  lexicographic reference.

The selector evaluates all 20 snapshots on A before creating any B instance.
On A it retains strict/full snapshots within the MAE margin, then ranks them by
lower **physical storage relocations per 100 deliveries**, greater dense return,
lower MAE, and earlier episode. It evaluates all snapshots on B for a complete
audit ledger, traverses only the frozen A order, and accepts the first strict/full
candidate that stays within +2.0 MAE and has strictly positive physical-storage
relocation saving. The authenticated V1.1 reference is the mandatory fallback.

The contention metric schema is frozen as
`physical_storage_relocation_decomposition_v1`. A physical storage relocation is
one completed storage-to-storage move emitted by the simulator. It is the common
operational burden and does not, by itself, claim an obstruction cause. Every run
must satisfy

`physical_storage_relocations = target_bound_obstruction_clearances + standalone_reconfigurations`.

VCG's standalone reconfigurations are further partitioned by whether a SAFE
direct-delivery candidate was available at that decision epoch. The
`directly_deliverable_self_reconfigurations` count is a subset diagnostic, not an
additional term. The legacy `relocations` field is retained as an exact alias of
the physical total; `obstructive_moves` aliases only target-bound obstruction
clearances and is therefore zero for VCG's standalone Reconfigure macros.

Both A and B are development-selection panels. The `+2.0` gate is an
operational tolerance, not by itself a statistical noninferiority claim. The
selected artifacts remain deployment-ineligible until an untouched final panel
is run under a separately frozen protocol.

## Train

Each seed performs 500 training episodes plus the unchanged periodic validation
clones and writes 20 authenticated snapshots. Run the seeds sequentially unless
the machine has enough independent CPU/GPU capacity for concurrent
exact-frontier search:

```bash
bash experiments/vcg_dense_v1_2_prospective/run_seed.sh 3
bash experiments/vcg_dense_v1_2_prospective/run_seed.sh 4
bash experiments/vcg_dense_v1_2_prospective/run_seed.sh 5
```

Append `--resume-existing` after an interruption. A deliberate pause is
allowed only at a snapshot boundary via `--stop-after-episode N`.

## Select

After all three candidate pools finish:

```bash
bash experiments/vcg_dense_v1_2_prospective/run-selector.sh
```

The complete evaluation is 3 model seeds x 20 snapshots x 60 instances =
3,600 policy episodes. Per-run transactional ledgers support exact resume:

```bash
bash experiments/vcg_dense_v1_2_prospective/run-selector.sh --resume-existing
```

Tests never instantiate either A or B.

## Frozen decision after selection

The selector advances only when all chosen arms are strict/full, at least two
of three model seeds select a non-reference checkpoint, and the equal-seed
panel-B physical-storage relocation saving is positive. Otherwise V1.2 checkpoint selection is
declared unsuccessful and the procedure falls back to V1.1.

If it advances, freeze the selected hashes before opening the untouched final
panel. Operational success there requires an equal-seed MAE point cost no
greater than `+2.0` and positive physical-storage relocation saving. The manuscript may use
“noninferior” only if the prespecified one-sided 95% upper confidence bound is
also within `+2.0`.
