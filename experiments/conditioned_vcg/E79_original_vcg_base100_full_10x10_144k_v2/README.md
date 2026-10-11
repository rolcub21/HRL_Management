# E79-v2: numerical-audit repair, same original VCG training

E79-v1 can stop on a long episode because it requires two differently grouped
floating-point sums of the same primitive rewards to agree within an absolute
`1e-9`. The v2 runner changes **only that end-of-episode accounting check**.
Its allowed difference is computed from the recorded primitive reward
magnitudes and episode length; the per-step reward decomposition check remains
unchanged. If the discrepancy exceeds the computed bound, training still stops.
Each completed episode records the bound and observed discrepancy in
`e79_v2_roundoff_audit`. No reward, discount, optimizer, model, certification,
selection rule, seed, or training budget is changed.

The E79-v1 contract includes source hashes. Do not edit its output or resume it
under a changed contract. If seed 0 completed only episodes 1 and 2, migrate
that checkpoint to a separate v2 output, retaining weights, optimizer, replay,
RNG state, validation history, and any selected candidate. The migration does
not run an episode. It accepts only unfinished checkpoints through episode 62;
keep the entire v1 directory as the provenance record.

After pulling this version on the GPU server, from its repository root:

```bash
E79V2=experiments/conditioned_vcg/E79_original_vcg_base100_full_10x10_144k_v2/run.sh
bash "$E79V2" prepare
bash "$E79V2" self-check
bash "$E79V2" migrate-seed --model-seed 0
bash "$E79V2" train-seed --model-seed 0 --through-episode 6
```

For a two-episode checkpoint, the first continuation reaches the first
prespecified validation point and checks that the checkpoint resumes. If the
v1 run completed six or more episodes, instead set `--through-episode` to
the next validation checkpoint above its completed episode. If it succeeds, continue the
original full budget and start seeds 1 and 2 from scratch under v2:

```bash
for seed in 0 1 2; do
  bash "$E79V2" train-seed --model-seed "$seed"
done
bash "$E79V2" analyze
```

The v2 output is
`results/vcg-conditioned-e79-original-base100-full-10x10-144k-v2`.
Its seed-0 `e79-v2-migration.json` records hashes of the preserved v1 and new
checkpoints. All migrated train-history entries retain their original v1
episode provenance. This is still development training, not a paper result.
