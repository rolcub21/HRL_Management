# E118: shorter exploration and modest earliness penalty, 8×8

Two matched development pilots train Release-Clock ET Cost VCG from scratch
with model seed 0. `schedule_only` retains the E115 objective
`Jref = 2T + E + 0.1L + 10R`. `early125` uses
`J125 = 2T + 1.25E + 0.1L + 10R`.

Both train for 120 episodes. Epsilon falls linearly from 0.90 to 0.05 over
8,000 macro decisions, then stays at 0.05. The learning rate is 5e-5 before
decision 8,000 and 1e-5 thereafter. At the observed roughly 115 decisions per
episode, the floor would arrive near episode 70, leaving about 50 episodes for
refinement. Actual decision counts can change with behavior; the report records
the observed floor episode and number of episodes ending at the floor.

The 8×8 yard, 36 jobs, empty start, 30-step release spacing, Poisson-80
durations, release-fixed due dates, ±20 tolerance, gamma=1, graph model,
shared-action four-head Double-DQN backup, certification, recovery guard, and
frozen-target conservative margin follow E115. The earliness weight applies
only outside the tolerance. Each arm's replay scalar, candidate merit,
checkpoint metadata, and objective-derived component-loss weights use that
arm's objective consistently.

The arms start from identical initial network parameters and use the same
ordered training-instance seeds (the E115 seed base 179000000), and the same
five E115/E116 development-validation seeds. Their trajectories and replay
contents will diverge as their actions diverge. There are no source checkpoints,
warm starts, behavior-return auxiliaries, or E117 fitted weights. Separate
contract hashes reject cross-arm or E115 checkpoint resumes. E115's source and
results remain unchanged; scoped configuration uses its verified trainer and
serializer. Some inherited metadata fields are named `e115_contract_sha256`;
their value here is the **E118 arm contract**, and the checkpoint also records
the explicit E118 arm and parent contract.

Every realized trajectory is scored under **both fixed objectives**. Thus a
lower raw training return cannot be mistaken for a policy difference caused
merely by using another coefficient. Greedy validation runs every six training
episodes, with no validation learning. The primary comparison is the final
five checkpoints **96, 102, 108, 114, 120**. We average across those checkpoints
within each instance first, then compare the two arms on the same five
instances. The report includes earliness, tardiness, release and storage MAE,
waiting, window coverage, work, steps, rehandles, completion, and checkpoint
variation. Repeated checkpoints do not turn five instances into 25 independent
test cases. Best-candidate artifacts are descriptive and are not the selection
rule for this experiment.

This is a two-arm, one-training-seed development comparison. The first arm
changes exploration and the learning-rate schedule together relative to E115;
it does not separate their effects. Comparing the two new arms tests the
earliness coefficient under that same shorter recipe. Fresh instances and
additional training seeds would follow a successful pilot.

## Requirements and checks

Use the existing repository `.venv` and a CUDA-enabled PyTorch installation.
The launcher sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts. No E115
checkpoint is needed. Self-check runs on CPU: one primitive WAIT per arm and two
synthetic optimizer updates; it runs no complete yard episodes.

```bash
cd /home/ai_diagnosis/HRL_Management
E118=experiments/conditioned_vcg/E118_short_exploration_earliness_8x8_182k_v1/run.sh
bash "$E118" self-check
```

## Launch both pilots sequentially

```bash
bash "$E118" train-all --through-episode 120 --device cuda &&
bash "$E118" analyze
```

`schedule_only` runs first, then `early125`. Training checkpoints after every
episode and validates every six. Repeating the command resumes each arm from
its own authenticated checkpoint. A completed arm is reused. The run logs
identify the E118 arm and show `J` (its training cost), `Jref`, release MAE,
earliness, tardiness, work, waiting, strict completion, epsilon, and learning
rate. The inherited banner is replaced with the actual E118 cost formula and
schedule, and inherited `DenseR` log labels are displayed as `Return`.

For separate launches or a bounded first segment:

```bash
bash "$E118" train-arm --arm schedule_only --through-episode 36 --device cuda
bash "$E118" train-arm --arm early125 --through-episode 36 --device cuda
bash "$E118" analyze
```

Then use `train-all --through-episode 120` to continue both. Partial reports
show the validation trajectory and keep the final-five comparison pending.

```bash
bash "$E118" status
```

Outputs live under
`results/vcg-conditioned-e118-short-exploration-earliness-8x8-182k-v1/`, with
one training directory per arm, `e118-report.json`, and `e118-table.md`.
`--output PATH` can select a different empty output root. This launcher prepares
training; no long training is started by the self-check or analysis commands.
