# Conditioned-handling VCG: damped convergence experiment

This training-only seed-0 experiment tests a conservative update for the
policy-conditioned future-handling model. It forks from the immutable original
round-4 terminal checkpoint, not from the unsuccessful round-6 convergence
diagnostic.

After fitting a provisional future-handling network on current-policy Monte
Carlo trajectories, the deployed parameters are updated by

```text
theta_next = 0.75 theta_previous + 0.25 theta_provisional.
```

The fixed coefficient `rho=0.25` applies in every round. VCG 1.1 operational
values remain frozen, the immediate Reconfigure cost remains structural, and
the 85k evaluation panel is not reopened.

## Fixed work

```text
4 rounds x (50 collection episodes + 9 fixed probes) = 236 simulator episodes
```

The first two training/probe streams match the earlier full-update convergence
extension, isolating the effect of damping. The experiment always ends after
four rounds and performs no checkpoint selection.

Convergence is assessed on the final two round transitions. Each must have:

- at most one changed action-sequence digest among nine fixed probes;
- unchanged aggregate probe rehandles;
- strict-safe completion for every probe;
- deployed held-episode handling MAE at most `0.35`;
- deployed absolute handling bias at most `0.10`.

If this test passes, the next step is a frozen diagnostic evaluation. If it
fails, neither the 85k evaluation nor seeds 1 and 2 should be run.

## Run

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/vcg_v11_conditioned_handling_damped_convergence/run.sh run
```

The run resumes from completed round boundaries. The final assessment is:

```text
results/vcg-v1-1-conditioned-handling-seed0-damped-convergence/
  damped-convergence-summary.json
```
