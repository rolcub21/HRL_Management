# Conditioned-handling VCG: seed-0 convergence extension

This is a short training-only continuation of the completed four-round seed-0
development experiment. The parent terminal checkpoint remains immutable.
The extension does not reopen or evaluate on the 85k panel.

## Fixed work

The extension adds exactly two rounds:

```text
2 rounds x (50 collection episodes + 9 fixed probes) = 118 simulator episodes
```

It continues the same algorithm and hyperparameters: the operational VCG 1.1
critic stays frozen, lambda is sampled continuously by a stratified schedule,
future handling uses complete-episode Monte Carlo labels, and the handling
network is fitted only after each complete collection round. The terminal
checkpoint is always the end of the second extension round; there is no
checkpoint selection.

## Predeclared stability test

Both transitions—parent round 4 to extension round 1, and extension round 1 to
extension round 2—must satisfy all of the following:

- at most one of the nine fixed probe action sequences changes;
- aggregate physical rehandles across the nine probes is unchanged;
- all probes remain strict-safe-complete;
- held-episode future-handling MAE is at most `0.35`;
- absolute held-episode prediction bias is at most `0.10`.

Return and timing MAE are recorded but are not used to select or stop a
checkpoint. If either transition fails, seeds 1 and 2 are not launched; the
fitting procedure should be reconsidered instead of adding more rounds
indefinitely.

## Run

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/vcg_v11_conditioned_handling_convergence_extension/run.sh run
```

The command resumes from the end of a completed extension round after an
interruption. Its final assessment is written to:

```text
results/vcg-v1-1-conditioned-handling-seed0-convergence-extension/
  convergence-extension-summary.json
```
