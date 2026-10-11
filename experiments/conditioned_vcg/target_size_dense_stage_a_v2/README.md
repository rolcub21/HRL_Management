# Target-size VCG-Dense Stage A v2

This corrects the v1 target-size launcher. It wraps the deployed
`train_vcg_dense_proper.py` recipe: dense timing reward, gamma 0.99,
episodic VCG agent, epsilon decay over 3,000 macro decisions, D12 relocation
families, and E14 cache/path cleanup. The exact E28 empty-start 8x8/36-job and
10x10/64-job instances replace only the environment distribution.

The old `results/vcg-target-size-training-development-v1/` checkpoints are
incompatible and must not be resumed. Stage B is intentionally not included:
it must first be changed to load episodic Stage-A checkpoints.

After copying this directory into `experiments/conditioned_vcg/`:

```bash
bash experiments/conditioned_vcg/target_size_dense_stage_a_v2/run.sh hardware
bash experiments/conditioned_vcg/target_size_dense_stage_a_v2/run.sh self-check
bash experiments/conditioned_vcg/target_size_dense_stage_a_v2/run.sh prepare
```

Run a two-episode CUDA timing gate. Episode 2 crosses the replay threshold and
therefore measures real TD-update cost; episode 1 alone does not.

```bash
bash experiments/conditioned_vcg/target_size_dense_stage_a_v2/run.sh train-stage-a \
  --scenario empty_8x8_n36 --model-seed 0 --device cuda \
  --stop-after-episode 2
```

The command writes `latest.pth` before episode 1 and after every expensive
episode, so the timing gate is resumable. Do not launch 10x10 or the full
budget until the two-episode timing report is reviewed. To resume the same
authenticated run, repeat the command without `--stop-after-episode`.

CPU execution is rejected for a full run. A <=2-episode CPU diagnostic is
possible only with both `--device cpu --allow-cpu-pilot`; it is not a runtime
estimate for the intended CUDA protocol.

The full protocol remains 500 target-size episodes so it matches the deployed
episode count. Because an 8x8 or 10x10 episode contains far more jobs,
decisions, candidates, and TD backups than a 5x5 episode, this is not a
compute- or sample-matched comparison. A reduced work-matched budget, if used,
must be a separately versioned experiment.
