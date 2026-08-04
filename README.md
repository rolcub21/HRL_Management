# Shipyard HRL Evaluation

Research code for a stochastic shipyard block-storage problem. The repository
compares three ways of choosing and executing storage operations:

1. hierarchical reinforcement learning (HRL) with a learned storage selector;
2. HRL with a genetic-algorithm (GA) storage assignment; and
3. the non-learning PSLaP heuristic.

The code was previously described as `HRL_Management`. This README documents
the repository as it exists today. No claims are made here that all historical
artifacts can be reproduced from this copy alone.

## Start here

- [Architecture](docs/ARCHITECTURE.md) explains the environment, options,
  learning agent, and method variants.
- [Experiments](docs/EXPERIMENTS.md) lists the runnable entry points, current
  parameter grids, outputs, and known external inputs.
- [Current layout](docs/CURRENT_LAYOUT.md) maps files to their roles and
  distinguishes source from generated artifacts.
- [Reorganization plan](docs/REORGANIZATION_PLAN.md) proposes a clean target
  layout without pretending that the current imports already support it.

## Problem summary

The environment represents a grid-based storage yard containing 40 blocks.
Blocks arrive after exponentially distributed inter-arrival times controlled
by `lambda` (`arrival_rate`). Each block requires a Poisson-distributed storage
duration controlled by `mu` (`proc_mean`). An agent must receive, store,
retrieve, and deliver every block.

Primitive actions are:

| ID | Action |
|---:|---|
| 0 | up |
| 1 | down |
| 2 | left |
| 3 | right |
| 4 | pick up |
| 5 | put down |
| 6 | wait |

The episode terminates when all 40 blocks have been delivered. Training and
evaluation scripts also impose a maximum number of steps.

## Methods

| Method | Storage decision | Execution |
|---|---|---|
| HRL | learned `StorageSelectOption` | manager/worker DQN |
| GA | fixed assignment loaded by `GAStorageSelectOption` | manager/worker DQN |
| PSLaP | heuristic policy | deterministic planning logic |

The HRL agent maintains separate manager and worker Q-networks. The manager
selects temporally extended options; the worker selects primitive actions.

## Installation

The last recorded working environment used Python 3.10.4, PyTorch 2.5.1,
NumPy 1.24.3, pandas 2.2.2, and Matplotlib 3.9.1. Install a PyTorch build suited
to the local CPU or CUDA environment first, then install the remaining direct
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch
python -m pip install -r requirements.txt
```

The default `python3` available in this workspace does not currently have
PyTorch installed; activate a prepared environment before running an entry
point.

The repository is not currently packaged. Its imports rely on both the project
root and `example/` being on `PYTHONPATH`:

```bash
cd /home/bukuj/hrl-eval
export PYTHONPATH="$PWD:$PWD/example"
```

## Entry points

Common, seed-controlled evaluation:

```bash
python common_evaluate.py \
  --method pslap \
  --lambda 0.2 \
  --mu 20 \
  --eval-seed 0 \
  --output results/evaluation.csv
```

Training and sensitivity scripts:

```bash
python example/run_small_rooms_example_hrl.py
python example/run_small_rooms_example_ga.py
python example/run_small_rooms_example_pslap.py
```

Read [Experiments](docs/EXPERIMENTS.md) before launching them. The training
scripts contain hard-coded grids and can run for a long time. The GA script
also requires an assignment file which is not present in this copy.

## Output conventions

New training scripts create a timestamped directory below `results/` with:

```text
results/<experiment-id>/
├── datalogs/   compressed NumPy logs
├── logs/       JSON logs
├── models/     PyTorch checkpoints
├── plots/      PNG and EPS figures
└── tb/         TensorBoard events
```

Historical outputs predate that convention and remain scattered in the root,
`example/models/`, and `runs/`. They are inventoried in
[Current layout](docs/CURRENT_LAYOUT.md).

## Current limitations

- There is no stable Python package or automated test suite.
- Several imports depend on the current working directory and `PYTHONPATH`.
- The batch evaluation shell script contains paths from another machine.
- The GA experiment requires a missing `best_assignment.pkl` artifact.
- The sensitivity plotting script refers to `results/` directories absent from
  this checkout.
- Historical checkpoints may use different network architectures.
- The environment state docstring mentions two timing features that the current
  return value does not include; see the architecture notes.

These limitations are documented before structural changes are attempted so
that future refactoring can preserve behavior intentionally.
