# E186: fresh 300-episode E78 replication with long exploration

E186 reruns the original E78 VCG-Dense v1.1 Stage-A architecture from scratch
on the same empty-start 8x8/36-job workload and base-80 objective. It does not
load an E78 checkpoint. The scientific changes from E78 are declared in the
contract: 300 episodes per model seed and epsilon 0.90 to 0.05 over 18,500
macro decisions rather than 3,000.

E78 averaged 84.33 macro decisions per training episode across its three
seeds. The declared 18,500-decision decay therefore reaches the floor around
episode 220 (approximately episode 213--227 at the individual old seed
rates). The schedule remains decision-based, as in E78; episode 220 is a
calibration target, not a forced boundary.

Everything else stays matched to E78: objective, gamma, scenario, three model
seeds, per-episode resumable checkpoints, certification, optimizer, network
and the validated Stage-A-v4 TD-backup acceleration. Evaluation is every 20
episodes rather than every six, reducing development runs from 750 to 225
across three seeds without changing training decisions. No test panel opens.

Preparation and the self-check run no episodes:

```bash
bash experiments/conditioned_vcg/E186_e78_long_exploration_245k_v1/run.sh prepare
bash experiments/conditioned_vcg/E186_e78_long_exploration_245k_v1/run.sh self-check
```

Start with seed 0. The command is resumable and can be stopped safely after
any completed episode:

```bash
MPLCONFIGDIR=/tmp/e186-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E186_e78_long_exploration_245k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 300
```

Seeds 1 and 2 use the same command with the seed changed. Analyze at any time:

```bash
bash experiments/conditioned_vcg/E186_e78_long_exploration_245k_v1/run.sh analyze
```

The output is `results/vcg-conditioned-e186-e78-long-exploration-245k-v1/`.
The default endpoint is 300, but `--through-episode N` supports staged runs.
