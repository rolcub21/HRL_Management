# E108: E107 low-epsilon continuation on 8x8

E108 resumes the complete, resumable E107 episode-224 checkpoint. It does not
restart training and does not load the replay-free selected checkpoint.

The agent weights, target network, optimizer, replay, RNG state, decision count,
queue-context representation, certification, guard, rewards, workload, and
validation panel are unchanged. Epsilon is already at 0.05 and remains there.
The only learning-protocol change is the maximum episode, from 224 to 320.

This is a one-seed development continuation. It does not authorize Stage B, an
independent test, deployment, or a paper claim. In particular, one favorable
checkpoint is not evidence of convergence. The analysis reports a
three-validation stability window at each decision checkpoint.

## Prepare and authenticate the continuation

These commands run no episodes:

```bash
cd /home/ai_diagnosis/HRL_Management

bash experiments/conditioned_vcg/E108_e107_low_epsilon_continuation_8x8_172k_v1/run.sh prepare
bash experiments/conditioned_vcg/E108_e107_low_epsilon_continuation_8x8_172k_v1/run.sh self-check
bash experiments/conditioned_vcg/E108_e107_low_epsilon_continuation_8x8_172k_v1/run.sh seed
```

## First persistence check: episode 240

Episode 240 is deliberately the first stop: it is only 16 additional
low-epsilon episodes and aligns exactly with the six-episode validation cadence.

```bash
bash experiments/conditioned_vcg/E108_e107_low_epsilon_continuation_8x8_172k_v1/run.sh \
  train-seed --through-episode 240 --device cuda

bash experiments/conditioned_vcg/E108_e107_low_epsilon_continuation_8x8_172k_v1/run.sh analyze
```

If the episode-224 improvement persists, continue to the prespecified longer
checkpoints:

```bash
for episode in 288 320; do
  bash experiments/conditioned_vcg/E108_e107_low_epsilon_continuation_8x8_172k_v1/run.sh \
    train-seed --through-episode "$episode" --device cuda
  bash experiments/conditioned_vcg/E108_e107_low_epsilon_continuation_8x8_172k_v1/run.sh analyze
done
```

