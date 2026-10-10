# E79: independent 10×10 original-VCG reward-shaped training

E79 trains the **original VCG-Dense v1.1 Stage-A architecture** from scratch
on the empty-start 10×10/64-job workload. It uses three independent model
seeds (0, 1, 2) and **63 episodes per seed**: 4,032 training jobs per seed,
matching the approximately 4,000-job exposure used for E78's 8×8 run.
There is no E70/E78 checkpoint dependency and no Stage B in this protocol.

## Frozen reward and comparison

The delivery reward is `100 − 1.5|e| − 0.5(|e|−20)+`, with the unchanged
primitive-step cost, reward scale, optimizer, certification, hierarchy,
ε schedule (`0.90 → 0.05` over 3,000 decisions), and primitive-step discount
`γ = 0.99`. The base **100** is declared before training by scaling E78's
8×8 base **80** by the yard-width ratio `10/8`; it is a size-scaling
hypothesis, not an empirically optimized or guaranteed-positive reward.
Individual delivery reward can still become negative: it crosses zero at
`|e| = 55` beyond the 20-step window. With `γ = 0.99`, the base change can
change the policy preference for earlier delivery; it is not claimed to be
policy-invariant shaping.

Only strictly complete validation episodes receive a base-40-equivalent
*undiscounted* return (`base-100 return − 64×60`) in the report. This
re-expression facilitates comparison of reported trajectories, not training
targets. Timing, rehandles, and steps remain physical metrics.

Every six episodes, the trainer evaluates on the same five fixed development
instances; there is also a terminal evaluation at episode 63. The prespecified
stability window comprises episodes **36, 42, 48, 54, 60, and 63**. Analysis
reports each seed's full validation history, terminal checkpoint, and complete
late window. The trainer's best checkpoint is descriptive only; there is no
quality-based checkpoint selection or sealed test evaluation.

## Requirements on the other server

- Use the same pushed Git commit and branch, including E79 and the previously
  pushed target-size training dependencies. The code's source hashes are part
  of the frozen contract. A different code revision must use a new output
  version; do not edit protocol files after `prepare`.
- Run from a checkout with a repository-root `.venv` satisfying
  `requirements.txt` and a CUDA-enabled PyTorch installation.
- Confirm `torch.cuda.is_available()` via the `hardware` command below.
  CUDA accelerates neural-network work, but certification and episode
  simulation remain substantially CPU-bound. Plan for a long run and adequate
  disk space for per-episode checkpoints. Prefer one seed at a time on one GPU.
- No 8×8 checkpoint, E70 results, or 10×10 training results need to be copied
  from the first server. E79 starts all three seeds afresh.

From the repository root on that server, after pulling the E79 commit:

```bash
bash experiments/conditioned_vcg/E79_original_vcg_base100_full_10x10_144k_v1/run.sh hardware
bash experiments/conditioned_vcg/E79_original_vcg_base100_full_10x10_144k_v1/run.sh prepare
bash experiments/conditioned_vcg/E79_original_vcg_base100_full_10x10_144k_v1/run.sh self-check
```

These commands run no training episodes. Start with a two-episode integration
check on seed 0, then resume all three seeds to the frozen 63-episode budget:

```bash
bash experiments/conditioned_vcg/E79_original_vcg_base100_full_10x10_144k_v1/run.sh train-seed --model-seed 0 --through-episode 2

for seed in 0 1 2; do
  bash experiments/conditioned_vcg/E79_original_vcg_base100_full_10x10_144k_v1/run.sh train-seed --model-seed "$seed"
done

bash experiments/conditioned_vcg/E79_original_vcg_base100_full_10x10_144k_v1/run.sh analyze
```

Each training command resumes from its own `latest.pth` if interrupted;
`--through-episode N` allows a bounded continuation. Seed 0's two pilot
episodes count toward its 63, not in addition. The report is
`results/vcg-conditioned-e79-original-base100-full-10x10-144k-v1/e79-report.json`.
`analyze` can be run at any time and returns `partial` until all seeds finish.

The inherited dense trainer may print a hard-coded `objective=40...` banner.
The E79 `contract.json`, training summary `objective_spec`, and checkpoints
record the actual base-100 reward; do not interpret that banner as the
objective in force.

This is a development-scale Stage-A replication, not evidence that base 100
outperforms base 40, and not a complete VCG operating-profile comparison.
Stage B and a fresh evaluation panel should follow only after the Stage-A
trajectories have been inspected.
