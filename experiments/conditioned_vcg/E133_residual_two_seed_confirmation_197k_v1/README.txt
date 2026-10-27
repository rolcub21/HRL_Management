E133: TWO ADDITIONAL MATCHED TRAINING SEEDS
=========================================

Run seeds 1 and 2 from scratch, both total_q and residual_q, to episode 60.
This adds FOUR training jobs (240 episodes total), not two resumed models.
Existing E131 seed-0 models/results remain read-only discovery evidence.
No learner, objective, reward scale, output link, guard, grouping, optimizer,
target cadence, batch size, nominal-cost calculation or exploration change.
The network/learning implementation is imported directly from frozen E131.

Pairing and information separation
----------------------------------
Within each model seed: same initial parameter tensors and sampler RNG for
both arms, same training-instance stream, same budget, same five greedy
development instances. Across model seeds: distinct initial weights and
training streams. Training seed = 187000000 + model_seed*10000 + episode - 1.
Seed 1 uses 187010000..187010059; seed 2 uses 187020000..187020059.
Both streams are disjoint from seed 0 and validation seeds 187800..187804.
No checkpoint weights, replay, optimizer or counters migrate from seed 0.

Rules unchanged
---------------
Each arm trains for at most 60 episodes with validation every six episodes.
Continuing beyond episode 36 requires the inherited completion/finite-TD
sanity gate. Failures save the endpoint and stop; the launcher does not
silently advance to the next job. The fixed comparison uses checkpoints
48, 54 and 60 only, never best-checkpoint selection. The inherited screening
rule is >=4 J improvement, completion preserved, and no increase in early/late
violation counts or mean excess magnitudes. No independent test panel opens.

Primary confirmation averages the seed-1/2 late-window metrics with equal
weight per training seed and applies that same rule. Each seed's result is
also reported separately. The three-seed mean including selected pilot seed 0
is descriptive, not independent confirmation. The validation panel is reused;
two training seeds and five development instances do not become 30 independent
instances by counting checkpoints. No reward/epsilon/budget retuning is allowed
on disappointing results. Baseline superiority is a separate question.

Preparation
-----------
    cd /home/ai_diagnosis/HRL_Management
    E133=experiments/conditioned_vcg/E133_residual_two_seed_confirmation_197k_v1/run.sh
    bash "$E133" prepare

Preparation authenticates frozen E131 source, preflight and optimizer smoke,
checks paired fresh weights and different cross-seed initializations on CPU,
and writes a new E133 contract. It performs NO training or fresh evaluation.
No E131 seed-0 checkpoint or summary is modified. Protected pilot summaries,
preflight, smoke and contract hashes are checked every invocation.

Recommended launch: all four jobs sequentially, fail-fast
--------------------------------------------------------
In ONE tmux session:
    bash "$E133" train-seeds --through-episode 60 --device cuda

Order:
    seed 1 total_q -> seed 1 residual_q -> seed 2 total_q -> seed 2 residual_q.
Every completed arm is saved before moving to the next. The episode-36 gate
is checked automatically before episode 37; no separate command is necessary.
Any error terminates the launcher. Re-running the same command resumes saved
arms, skips already completed endpoints, then continues the remaining jobs.
It never runs jobs concurrently. Avoid other heavy GPU jobs during this run.
Do not start a second launcher against the same output while one is active.

Optional explicit checkpoints/endpoints
---------------------------------------
    bash "$E133" train --model-seed 1 --arm total_q --through-episode 36 --device cuda
    bash "$E133" train --model-seed 1 --arm total_q --through-episode 60 --device cuda

The --model-seed argument is mandatory for train. Only 1 and 2 are allowed;
seed 0 cannot be accidentally resumed or overwritten. Checkpoint authentication
requires protocol, source contract, model seed AND arm identity. Optimizer,
replay with nominal K, sampler state, counters and global RNG are restored.

Monitoring and outputs
-----------------------
    bash "$E133" analyze

Default output:
    results/vcg-conditioned-e133-residual-two-seed-confirmation-197k-v1
Per arm/seed:
    total_q/seed-1/{latest.pth,summary.json}
    residual_q/seed-1/{latest.pth,summary.json}
    total_q/seed-2/{latest.pth,summary.json}
    residual_q/seed-2/{latest.pth,summary.json}
Aggregate: e133-report.json and e133-table.md. Partial reports explicitly show
pending jobs and withhold confirmation decisions until both new pairs finish.
CUDA/cuBLAS and single-thread CPU settings match E131's launcher. The inner
reused episode line still says E123; the E133 header and checkpoint paths give
the correct training seed and arm. No code was pushed and no long run is
launched by preparation or tests; the user starts training with the command above.
