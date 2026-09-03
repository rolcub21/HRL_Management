# Two-phase conditioned-handling VCG: model seeds 1 and 2

This experiment replicates the finalized seed-0 training recipe for VCG 1.1
model seeds 1 and 2. Each model uses its own authenticated frozen operational
checkpoint and its own independently fitted nested handling head.

## Fixed recipe per model seed

```text
Phase 1: 4 rounds x (50 collection episodes + 9 probes), full fitted update
Phase 2: 4 rounds x (50 collection episodes + 9 probes), rho=.25 update
Total:   400 collection episodes + 72 probes = 472 simulator episodes
```

The two seeds use the same EpisodeInstance seeds, lambda schedules, validation
splits, and fitting seeds as the finalized seed-0 recipe. Pairing these streams
isolates model-seed stability. The streams remain disjoint from evaluation
panels.

The operational VCG 1.1 critic is frozen throughout. Future handling is fitted
from complete-episode Monte Carlo targets, and the current Reconfigure cost
remains structural. No evaluation panel is opened and no checkpoint is
selected: each terminal is the fixed end of round 8.

Each seed advances to evaluation only if its final two damped transitions pass
the established probe-stability and calibration rule. Evaluation is prepared
only if both seeds pass.

## Run

Both seeds sequentially:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/vcg_v11_conditioned_handling_two_phase_seeds12/run.sh run-all
```

Or separately:

```bash
bash experiments/vcg_v11_conditioned_handling_two_phase_seeds12/run.sh run-seed1
bash experiments/vcg_v11_conditioned_handling_two_phase_seeds12/run.sh run-seed2
bash experiments/vcg_v11_conditioned_handling_two_phase_seeds12/run.sh analyze
```

Runs resume from completed round boundaries. Running seed 1 and seed 2 in
separate terminals is possible, but sequential execution is recommended when
they share one GPU.

The combined training-only decision is written to:

```text
results/vcg-v1-1-conditioned-handling-seeds12-two-phase-development/
  two-phase-seeds12-training-report.json
```
