# Target-size VCG training on E28 empty-start workloads

This is a separate development experiment, **not** a correction of the frozen
5×5 zero-shot results. It trains the original Stage A graph operational critic
and Stage B conditioned handling head at 8×8/36 jobs and 10×10/64 jobs. The
geometry, three exits, empty initial yard, arrival interval of 10 primitive
steps, and Poisson(80) storage durations match E28. The training/validation,
pilot, and test instance seeds are disjoint from one another and from E28's
previously opened panel. The exact verifier, D12 family proofs, E14 path
cleanup/cache, selector, and guard remain in place.

This is a *target-trained versus frozen-baseline* comparison, not an equal
training-data comparison: the A3C adaptation remains its frozen 5×5-trained
checkpoint. Report it alongside the separate zero-shot experiment, never as a
replacement for it. An improvement cannot be attributed solely to model
architecture because the training distribution changed.

## Preparation and feasibility gate

Run from the HRL_Management root after copying this directory there:

```bash
bash experiments/conditioned_vcg/target_size_vcg_training/run.sh self-check
bash experiments/conditioned_vcg/target_size_vcg_training/run.sh prepare
```

Stage A is 500 training episodes per model seed and yard size, with five fixed
validation instances every 50 episodes. Stage B is eight rounds of 50 fresh
complete episodes per model seed and yard size, with the operational critic
frozen and late-round updates damped to 25% after round 4. That is four Stage A
and four Stage B training runs, potentially very
expensive at 10×10. **Do a 50-episode Stage A feasibility pause first** and
inspect completion, wall time, and validation before committing to full runs:

```bash
bash experiments/conditioned_vcg/target_size_vcg_training/run.sh train-stage-a \
  --scenario empty_8x8_n36 --model-seed 0 --device cpu --stop-after-episode 50
bash experiments/conditioned_vcg/target_size_vcg_training/run.sh train-stage-a \
  --scenario empty_10x10_n64 --model-seed 0 --device cpu --stop-after-episode 50
```

The same command without `--stop-after-episode` resumes to 500. Run it for
model seeds 0 and 1 at both sizes. Stage B starts only from a complete,
validation-eligible Stage A best checkpoint:

```bash
bash experiments/conditioned_vcg/target_size_vcg_training/run.sh train-stage-b \
  --scenario empty_8x8_n36 --model-seed 0 --device cpu --stop-after-round 1
```

Repeat without `--stop-after-round` to resume through round 8, then run all
remaining size/seed combinations. If a Stage B collection episode is not
strict-safe-complete, the run stops rather than fitting incomplete Monte Carlo
targets. Its `failed-collection.json` is diagnostic; changing the horizon or
workload then requires a separately versioned protocol.

## Evaluation

After *all four* Stage B terminal checkpoints exist, first freeze their hashes
before opening the three-new-instance-per-size test panel:

```bash
bash experiments/conditioned_vcg/target_size_vcg_training/run.sh freeze-test
bash experiments/conditioned_vcg/target_size_vcg_training/run.sh run-panel --panel test
bash experiments/conditioned_vcg/target_size_vcg_training/run.sh analyze --panel test
```

`run-panel` is resumable and includes VCG at λ = 0, .05, .10, .20 for two
model seeds, plus Dynamic PSLAP (one run), capacity-aware GA (four optimizer
replications), and A3C adaptation (15 model/stream replications) on each
exactly matched EpisodeInstance. It executes 168 episode rows in total. The
report averages method replications within each instance, then instances;
placement-clock and release-clock timing, pre-storage wait, rehandles, and
steps are conditional on strict completion, while reliability has its own
denominator. The four block timestamps are checked through the exact timing
identity on each completed episode. Failed episodes must not be silently
removed from completion counts. Neural training or evaluation is never started by `prepare` or
`self-check`.

For a single integration pilot after the first Stage A/B pair completes, use:

```bash
bash experiments/conditioned_vcg/target_size_vcg_training/run.sh run-panel --panel pilot
```

This pilot uses one new instance per size, VCG seed 0 at λ=.10, and one
replicate per baseline; it is not paper evidence. Training and test outputs
are under `results/vcg-target-size-training-development-v1/` by default.
