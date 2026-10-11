# VCG 1.1 conditioned handling: seed-0 development screen

This is the first development test of the general conditioned-handling
controller. It is not a confirmation experiment and it does not authorize a
multi-seed claim.

The controller freezes the authenticated VCG 1.1 operational critic and uses

```text
Q_N(s,c,lambda) = 1[c is Reconfigure] + softplus(f_N(h(s,c), lambda))
M(s,c,lambda)   = Q_op^V1.1(s,c) - lambda Q_N(s,c,lambda).
```

The learned target excludes the current action's known handling:

```text
Y_future(t) = sum of physical rehandles from decisions t+1,...,T-1.
```

Consequently the immediate Reconfigure cost cannot be forgotten or counted
twice. At lambda zero the implementation directly calls the original VCG 1.1
selector.

## Training

Seed 0 uses four policy-evaluation/refitting rounds. Each round freezes the
current policy for 50 complete episodes, samples one continuous lambda in
`(0, .2)` per episode by stratified uniform sampling, computes exact
complete-episode Monte Carlo future-handling labels, and then fits only the
conditioned future-handling network. `Q_op` is never updated and no teacher
policy is queried.

The fixed terminal model is the end of round 4. There is no checkpoint
selection. A stopped run resumes only from a completed round boundary.

## Evaluation

The opened `85000..85011` EpisodeInstances are reused. The reporting grid is

```text
0, .025, .0375, .05, .075, .10, .125, .175, .20
```

The four values `.0375`, `.075`, `.125`, and `.175` are interpolation
coordinates absent from the old nested sweep. The old nested controller is
also evaluated there so the comparison is paired at every coordinate.

Only three pre-training rollouts are used: lambda zero and the two positive
sentinels `.025` and `.20`. They must reproduce the authenticated VCG 1.1 and
old nested behavior exactly. Post-training evaluation uses 145 new rollouts:
96 conditioned positive-lambda rows, one lambda-zero sentinel, and 48 old
nested interpolation rows. The 12 authenticated lambda-zero rows are reused
only after the terminal sentinel passes.

## Run

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/vcg_v11_conditioned_handling_seed0_85k/run.sh run-seed0
```

The command is sequential. To resume after interruption, issue the same
command; training resumes from `latest.pth` when it exists. To run stages
separately, use `prepare`, `parity`, `train`, `evaluate`, and `analyze`.

For CPU execution:

```bash
DEVICE=cpu bash experiments/vcg_v11_conditioned_handling_seed0_85k/run.sh run-seed0
```

The final diagnostic is written to
`results/vcg-v1-1-conditioned-handling-seed0-85k-development/conditioned-handling-report.json`.
