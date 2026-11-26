# PointMaze isolated R12 factorial runbook

This screen is separate from the running AntMaze R49 evaluation. It neither
changes nor reruns frozen PointMaze R11. It uses two exposed R11 corner goals,
three episodes per goal, and the same four observation-memory x passage arms.

Do not start PointMaze work merely because this code is prepared. First inspect
the AntMaze R49 result. If the Ant result justifies the cross-domain check, run
these commands one at a time.

## Preparation

```bash
E600_DEVICE=cpu projects/tfsn_consequence_value_iteration/pointmaze/run_factorial_r12.sh audit-code
```

```bash
E600_DEVICE=cuda projects/tfsn_consequence_value_iteration/pointmaze/run_factorial_r12.sh train-passage
```

Training uses only existing reward-free PointMaze transitions, prints every
epoch, stops on the shared validation rule, refuses duplicate trainers, and
requires at least 5 GiB free.

## Preflight

```bash
E600_DEVICE=cpu projects/tfsn_consequence_value_iteration/pointmaze/run_factorial_r12.sh preflight
```

The preflight verifies the frozen R11 model, graph and bridge hashes; the
data-derived candidate support and radius bounds; the shared model/loss/trainer
contract; the isolated manifest hash; and fatal posterior exhaustion.

## Development evaluation

```bash
E600_DEVICE=cpu projects/tfsn_consequence_value_iteration/pointmaze/run_factorial_r12.sh evaluate-2x2
```

This is 4 arms x 2 goals x 3 episodes = 24 episodes, with at most 20 commands
per episode. Progress prints after every command. Results are written only
under `results/pointmaze/latent_detection_passage_factorial_r12/`.
