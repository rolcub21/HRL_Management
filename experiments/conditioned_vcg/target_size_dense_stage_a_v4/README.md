# Target-size VCG-Dense Stage A v4

V4 is the practical target-trained protocol. It retains the deployed dense
objective, gamma, episodic architecture, exact certified frontier, optimizer,
replay sampling, update cadence, selector, guard, D12, and E14.

Two implementation-only accelerations are authenticated: repeated immutable
states are encoded once in no-gradient online continuation scoring, and
Q-target candidate scores are cached only while Q-target is frozen. The cache
is invalidated after every 200-step target synchronization and is not stored
in checkpoints. Parity tests covered eight updates and the synchronization
boundary.

Training exposure is work-matched to historical 5x5 Stage A (500 episodes x 8
jobs = 4,000 jobs):

- 8x8/36 jobs: 112 episodes = 4,032 jobs;
- 10x10/64 jobs: 63 episodes = 4,032 jobs.

This replaces the misleading 500-large-episode budget. Run a fresh two-episode
8x8 timing gate:

```bash
bash experiments/conditioned_vcg/target_size_dense_stage_a_v4/run.sh self-check
bash experiments/conditioned_vcg/target_size_dense_stage_a_v4/run.sh prepare
bash experiments/conditioned_vcg/target_size_dense_stage_a_v4/run.sh train-stage-a \
  --scenario empty_8x8_n36 --model-seed 0 --device cuda \
  --stop-after-episode 2
```

Do not reuse v2/v3 checkpoints. Stage B remains blocked until Stage A's timing
and learning curves justify proceeding.
