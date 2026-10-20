# E107: queue-observed long-horizon Stage A on 8x8

E107 is a fresh Stage-A training experiment. It does **not** resume E105 or
E106 because the critic input and replay schema have changed.

The exact certified candidate frontier, recovery guard, graph encoder,
base-80 storage-clock delivery objective, and E105 accrued outside-wait
penalty are retained. The critic additionally receives eight bounded facts
visible at the current decision boundary:

- waiting-job fraction;
- oldest and current-inbound wait ages;
- over-cap fraction and accumulated excess-wait pressure;
- accepted and delivered fractions; and
- elapsed episode fraction.

The context never reads future release times or unreleased durations. It is
stored with every replay record, so a replayed target cannot silently use the
live episode's context.

The training maximum is 224 episodes. Epsilon decays linearly from 0.90 to
0.05 over 18,000 macro decisions rather than E105's 3,000 decisions. It stays
at 0.05 afterwards. Because episode lengths are behavioral outcomes, the
protocol authenticates the schedule in decisions rather than claiming a
particular episode at which the floor will be reached.

This is a one-seed development experiment on a reused validation panel. It
does not authorize Stage B, an independent test, deployment, or a paper claim.

## Prepare and verify (no episode run)

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E107_queue_context_stage_a_8x8_171k_v1/run.sh prepare
bash experiments/conditioned_vcg/E107_queue_context_stage_a_8x8_171k_v1/run.sh self-check
```

## Train

The first useful stop is episode 36:

```bash
bash experiments/conditioned_vcg/E107_queue_context_stage_a_8x8_171k_v1/run.sh \
  train-seed --through-episode 36 --device cuda
bash experiments/conditioned_vcg/E107_queue_context_stage_a_8x8_171k_v1/run.sh analyze
```

Resume the same command at the prespecified checkpoints:

```bash
for episode in 60 112 160 224; do
  bash experiments/conditioned_vcg/E107_queue_context_stage_a_8x8_171k_v1/run.sh \
    train-seed --through-episode "$episode" --device cuda
  bash experiments/conditioned_vcg/E107_queue_context_stage_a_8x8_171k_v1/run.sh analyze
done
```

Stopping at a checkpoint is for inspection and recoverability. A dip at one
checkpoint is not itself a stopping rule, and the best row on this reused
development panel is not test evidence.
