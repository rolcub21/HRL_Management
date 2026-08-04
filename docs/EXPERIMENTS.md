# Experiments and evaluation

## Terminology

- `lambda` / `arrival_rate`: rate of the exponential arrival process; the mean
  inter-arrival interval is `1 / lambda`.
- `mu` / `proc_mean`: mean of the Poisson distribution used for required
  storage duration.
- training seed: seed used while fitting an HRL or GA-assisted HRL checkpoint.
- evaluation seed: independently controls a common evaluation episode.
- success: all 40 blocks were delivered before the step limit.
- delivery error: mean recorded delivery-time error for an episode. Existing
  files can contain signed values; some plots use its absolute interpretation.

## Entry-point matrix

| Entry point | Method | Purpose | Status in this copy |
|---|---|---|---|
| `example/run_small_rooms_example_hrl.py` | HRL | train learned-selector sensitivity runs | code and output convention present |
| `example/run_small_rooms_example_ga.py` | GA-assisted HRL | train with GA storage assignment | assignment file missing |
| `example/run_small_rooms_example_pslap.py` | PSLaP | heuristic sensitivity grid | self-contained in principle |
| `common_evaluate.py` | all three | one seed-controlled evaluation row | usable when required artifacts exist |
| `run_learned_common_evaluation.sh` | HRL and GA | batch learned-policy comparison | paths refer to another machine |
| `plot_revision_sensitivity.py` | all three | aggregate sensitivity plots | referenced result directories missing |

## Training experiments as currently configured

### Learned selector HRL

The script uses diagonal parameter pairs:

```text
(lambda=0.2, mu=20)
(lambda=0.5, mu=50)
(lambda=1.0, mu=80)
```

It trains seeds 3 and 4. The active `is_quick_test=True` branch sets 2,500
episodes, 4,000 steps per episode, and a 100-episode plotting window. The
inactive branch sets only 2,000 episodes, so the name "quick test" is currently
misleading.

### GA-assisted HRL

The grid, seeds, episode count, and agent hyperparameters match the learned
selector experiment. It requires:

```text
results/ga_20260311-093907-20 Gen/best_assignment.pkl
```

That file and directory are absent from this repository.

### PSLaP sensitivity

The active configuration evaluates the Cartesian product:

```text
lambda in {0.2, 0.5, 1.0}
mu     in {20, 50, 80}
seed   in {0, 1, 2}
```

For each combination it runs five episodes with a 4,000-step limit. This is
9 settings times 3 seeds times 5 episodes, or 135 episodes in total.

## Common evaluator

`common_evaluate.py` appends one row to a CSV and accepts:

```text
--method {hrl,ga,pslap}
--lambda FLOAT
--mu FLOAT
--eval-seed INT
--train-seed INT          optional metadata, default -1
--checkpoint PATH         required for HRL and GA
--ga-assignment PATH      required for GA
--max-steps INT           default 4000
--output PATH             required
```

Examples, after setting `PYTHONPATH` as shown in the main README:

```bash
# PSLaP does not need a learned checkpoint.
python common_evaluate.py \
  --method pslap --lambda 0.2 --mu 20 --eval-seed 0 \
  --output results/evaluation.csv

# HRL needs a compatible checkpoint.
python common_evaluate.py \
  --method hrl --lambda 0.2 --mu 20 \
  --train-seed 3 --eval-seed 0 \
  --checkpoint /path/to/model_HRL.pth \
  --output results/evaluation.csv

# GA additionally needs the saved assignment.
python common_evaluate.py \
  --method ga --lambda 0.2 --mu 20 \
  --train-seed 3 --eval-seed 0 \
  --checkpoint /path/to/model_HRL.pth \
  --ga-assignment /path/to/best_assignment.pkl \
  --output results/evaluation.csv
```

The evaluator forces greedy selection for learned policies. It reconstructs
the networks, loads manager and worker weights, and reconstructs the learned
selector architecture from its checkpoint when evaluating HRL.

## Output schemas

### Common-evaluation CSV

```text
method, lambda, mu, train_seed, eval_seed, checkpoint_id,
return, delivery_error, success, steps, decision_seconds
```

### Training NPZ

```text
episode_returns
episode_avg_error
episode_success
manager_losses
worker_losses
```

### Training JSON

The JSON log records the same logical arrays in a human-readable form.

### Checkpoint

Every current checkpoint contains:

```text
manager_state_dict
worker_state_dict
manager_opt_state
worker_opt_state
epsilon
step_count
```

Learned-selector checkpoints also contain keys prefixed with `selector_`.

## Existing evidence in this copy

- `common_eval_20260623-142857.csv`: 15 PSLaP rows across three diagonal
  settings and five evaluation seeds.
- `common_eval_20260623-143142.csv`: one HRL evaluation row.
- `evaluation_smoke.csv` and `evaluation_smoke_gpu.csv`: duplicate HRL smoke
  evaluations on two execution environments.
- `ga_smoke.csv`: one successful GA-assisted evaluation.
- `hrl_old_arch_smoke.csv`: one unsuccessful old-architecture HRL evaluation.
- `example/models/`: 85 historical checkpoints dated from July 2025 through
  February 2026.
- `runs/hrl/`: TensorBoard event files for several runs.
- root-level NPZ files: legacy reward arrays whose provenance is only partially
  encoded in their key names.

## Reproducibility cautions

Do not compare result files solely by filename. A defensible comparison should
record method, code revision, environment parameters, training seed, evaluation
seed, step limit, checkpoint identity, GA assignment identity, dependency
versions, and whether delivery error is treated as signed or absolute.

The existing training scripts do not store all of that metadata in one manifest.
Adding an experiment manifest is part of the proposed reorganization.

The workspace's default `python3` interpreter currently cannot import PyTorch,
so none of the learned-policy entry points should be assumed runnable until a
project environment has been created or activated.
