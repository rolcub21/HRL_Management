# Shipyard HRL Evaluation

Research code for a stochastic shipyard block-storage problem. The repository
contains legacy manager/worker HRL, GA-assisted HRL, explicit PSLAP/GA
baselines, and the current viability-constrained graph (VCG) SMDP controller.

The code was previously described as `HRL_Management`. This README documents
the repository as it exists today. No claims are made here that all historical
artifacts can be reproduced from this copy alone.

## Start here

- [Architecture](docs/ARCHITECTURE.md) explains the environment, options,
  learning agent, and method variants.
- [Experiments](docs/EXPERIMENTS.md) lists the runnable entry points, current
  parameter grids, outputs, and known external inputs.
- [Unified VCG](docs/UNIFIED_VCG.md) gives the current paper-facing variants,
  nested handling architecture, confirmed lambda frontier, and matched results.
- [Fully learned hierarchy](docs/FULLY_LEARNED_HIERARCHY.md) documents the
  parameterized mode/cell controller, phased training, and strict evaluation.
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
| Dynamic PSLAP | repaired online heuristic | immutable yard planning and deterministic execution |
| Rolling PSLAP GA | arrived-block rolling search | shared deterministic PSLAP execution |
| Offline PSLAP GA | full-schedule search | shared deterministic execution; offline reference only |
| Legacy adapted PSLAP | frozen pre-audit heuristic | internal ablation only |
| Duration-aware REG hybrid | frozen REG-v5 assignment | deterministic one-step timing lookahead with strict atomic macros |
| Fully learned hierarchy v2 | learned temporal mode, retrieval target, and exact cell | strict reserved atomic macros with one common mode-regularized SMDP target |
| Unified VCG | frozen VCG 1.1 operational controller plus detached handling predictor | exact-safe VCG selection with an optional inference-time handling cost |

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

Run maintained entry points from the project root. The active option and common
evaluation paths use package-qualified imports; some legacy modules still need
both the project root and `example/` on `PYTHONPATH`:

```bash
cd /path/to/HRL_Management
export PYTHONPATH="$PWD:$PWD/example"
```

## Entry points

Common, seed-controlled evaluation:

```bash
python common_evaluate.py \
  --method dynamic_pslap \
  --lambda 0.2 \
  --mu 20 \
  --eval-seed 0 \
  --output results/evaluation.csv
```

Storage-assignment isolation with the neutral Track A executor:

```bash
python track_a_evaluate.py \
  --method dynamic_pslap \
  --lambda 0.5 --mu 50 --episode-seed 100 --ga-seed 3100 \
  --save-instance results/track-a-seed100.json \
  --audit-output results/track-a-dynamic-seed100-audit.json \
  --output results/track-a.csv
```

Track A supplies every method the same physically feasible candidate tuple and
uses a fixed retrieval, relocation, routing, and primitive-execution stack.
Invalid proposals fail the strict result. A shared nearest-reachable fallback
may continue the operational episode, but every later outcome is flagged as
fallback-contaminated.

`reg_selector` is supported only with a newly trained selector feature-version
3 checkpoint; version 2 exposed hidden durations for unarrived blocks and is
rejected from the online Track A table.

`reg_selector_v4` is the Track A-native learned selector. It encodes every
observable block as a permutation-invariant set and scores each admissible cell
with one shared candidate network. Its training and evaluation both use the
neutral executor and exact shared mask.

`reg_selector_v5` retains that architecture but learns non-overlapping
assignment-to-assignment SMDP transitions with a variable-duration Double-DQN
target. A frozen v5 checkpoint can be exercised inside the complete Track B
controller with `track_b_evaluate.py`; this is an integration/transfer test, not
a Track A baseline row.

`compare_track_b.py` performs the complete-system comparison. It evaluates the
learned gated controller and the deterministic baseline executors on the exact
same immutable `EpisodeInstance` for every seed. Online nearest-free, dynamic
PSLAP, and rolling GA are primary Track B comparators; full-schedule GA is
reported separately as an information-advantaged offline reference.

The primary Track B controller is `atomic_inbound_scheduler` (v5). Its learned
decision epochs expose 42 stable macro outputs: one mandatory atomic
Accept--Pickup--REG-Assign--Store workflow, one event-driven Defer, and one
atomic Retrieve--Deliver job for each of the 40 manifest blocks. Primitive
controls remain serialized only for checkpoint/base-class compatibility and are
masked from action selection, replay continuation, and learning. Forced
singleton decisions do not advance epsilon; discretionary exploration assigns
equal mass to the Retrieve group and Defer before choosing a block. The common
SMDP continuation is a masked maximum over admissible macros. The v4 split
inbound/two-mode controller remains available as the `mode_only_scheduler`
ablation, and earlier variants remain explicitly selectable.

`track_b_urgency_evaluate.py` provides the first repaired scheduling candidate:
frozen REG-v5 still chooses storage cells, while a deterministic scheduler
retrieves executable jobs whose canonical complete-plan slack is nonpositive,
then accepts inbound inventory, uses an executable retrieval to release
capacity when needed, and otherwise defers to the next observed event. Its
retrieval initiation contract is strict v2: a stored-yard canonical plan must
exist and its first complete live pickup/putdown leg must be executable. A
blocked due job with no state-changing recovery is an explicit method failure,
not a repeated WAIT. The learned checkpoints retain retrieval v1, so this is a
repaired hybrid candidate—not a one-factor scheduling ablation against v5.

The separately versioned `reg_v5_duration_aware_atomic` candidate extends that
rule only at positive-slack inbound epochs. It previews the current frozen
REG-v5 placement, estimates the complete AcceptStore duration, projects the
current head retrieval into the post-accept yard, and compares predicted
absolute delivery errors. With current and projected slacks `s0` and `s1`, it
retrieves when `|-s0| <= |-s1| + m`. Due retrieval, capacity release, and event
deferral retain the urgency-first contracts. The preview is noncommitting:
AcceptStore-v1 makes its real assignment after pickup, so the preview cell and
duration are audited estimates rather than promises. Calibration over margins
`{0, 2, 4}` selected `m=2`; final claims require the untouched holdout protocol
in [Experiments](docs/EXPERIMENTS.md).

`compare_track_b_assignment_sources.py` is the strict component ablation for
that scheduler. It keeps the duration-aware macro policy, margin, executor,
relocation rule, defer contract, and exact `EpisodeInstance` fixed while using
REG-v5, nearest-free, or dynamic PSLAP for both preview and post-pickup
assignment. All sources share one candidate mask and invalid proposals fail
without fallback. The paired output therefore isolates the assignment
source's coupled contribution within this scheduler; congestion claims still
require the separately planned narrow-gate setting.

`compare_track_b_contention.py` implements that follow-up as a paired ordinary
versus two-cell-egress experiment crossed with the same three assignment
sources. It matches stochastic work across geometries with a geometry-free
`schedule_id`, keeps the complete scheduler/executor contract fixed, audits
relocations and replans, and reports paired difference-in-differences. Here
"contention" means inventory-induced egress obstruction in the single-agent
yard. The frozen REG checkpoint's constrained result is also a zero-shot
geometry-generalization result. See [Experiments](docs/EXPERIMENTS.md) for the
frozen confirmation command and interpretation limits.

The opt-in relational experiment keeps that repaired macro layer and frozen
REG-v5 selector but replaces the fixed 42-output scheduling head with one shared
action-conditional scorer. `train_relational_track_b.py` learns a residual over
the urgency baseline from variable feasible `AcceptStore`, strict retrieval, and
Defer candidates; `track_b_relational_evaluate.py` is its only checkpoint
loader. Its opt-in `kind_safe_residual_map` policy preserves the baseline macro
kind while allowing learned retrieval reordering; existing `residual_map` and
`baseline` behavior remains unchanged. Its architecture, action interface,
network keys, and checkpoint schema are distinct from atomic v5, so the
historical path remains reproducible. This first version isolates scheduling:
direct `Store(block, cell)` candidates and post-decision search are
intentionally deferred to separately versioned work.

The evaluator materializes the stochastic schedule and yard geometry as an
immutable `EpisodeInstance`. Every result records its content-derived
`instance_id`, so rows are considered matched only when those IDs agree. Use
`--save-instance instances/seed-0.json` once and
`--instance instances/seed-0.json` for subsequent methods when an exact,
portable replay artifact is desired.

Maintained Track A and Track B outputs also report inbound storage flow time,
`F_i = S_i - A_i`, from exogenous arrival to the first successful storage
PUTDOWN. This is total arrival-to-completed-storage response time, not queue
wait, required storage duration, or delivery timing error. Unfinished arrived
blocks are right-censored without imputing completion times; future arrivals
are counted separately. Primary mean, median, p90, p95, and maximum flow-time
statistics are null unless the complete manifest has been stored, while counts,
completion rates, raw completed times, and censor records remain auditable.

Training and sensitivity scripts:

```bash
PYTHONHASHSEED=0 python train_reg_selector.py --episodes 20 \
  --output-dir results/reg-selector-v3-seed0-20ep
PYTHONHASHSEED=0 python train_reg_selector_v4.py --episodes 100 \
  --output-dir results/reg-selector-v4-seed0-100ep
PYTHONHASHSEED=0 python train_reg_selector_v5.py --episodes 100 \
  --output-dir results/reg-selector-v5-seed0-100ep
python -m example.run_small_rooms_example_hrl
python -m example.run_small_rooms_example_ga
python -m example.run_small_rooms_example_pslap
```

`train_reg_selector.py` preserves the v3 flat selector trained inside the full
HRL controller. `train_reg_selector_v4.py` trains v4 directly inside Track A.
The historical sensitivity launchers retain their larger hard-coded grids.

`--method pslap` remains a compatibility alias for `dynamic_pslap`. Use
`--method legacy_adapted_pslap` only for a labeled internal ablation.
`--method pslap_ga_2009_offline` runs the full-schedule Park--Seo-inspired GA
reference; it is information-advantaged and not a deployable online peer.
`--method pslap_ga_2009_rolling` reoptimizes using only arrived blocks.
`--method pslap_ga_2009` remains a compatibility alias for the offline method.
The evolutionary operators and environment mapping are documented reproduction
choices, not an official author implementation. The existing `--method ga`
remains a different GA-assisted HRL method.

A small matched-seed geometry comparison is available through
`compare_pslap_baselines.py`; use `--exit-width 3` to activate obstruction and
relocation behavior that may remain absent under the default gate.

Read [Experiments](docs/EXPERIMENTS.md) before launching them. The training
scripts contain hard-coded grids and can run for a long time. The GA script
also requires an assignment file which is not present in this copy.

The maintained test suite uses pytest:

```bash
python -m pytest -q
```

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

- There is no installable Python package; legacy modules still mix import
  styles.
- The automated suite covers option/selector/checkpoint contracts, not
  statistical learning performance.
- The batch evaluation shell script contains paths from another machine.
- The GA experiment requires a missing `best_assignment.pkl` artifact.
- The sensitivity plotting script refers to `results/` directories absent from
  this checkout.
- Historical checkpoints may use different network architectures and lack the
  controller-index metadata required for verified evaluation.
- The environment state docstring mentions two timing features that the current
  return value does not include; see the architecture notes.

These limitations are documented before structural changes are attempted so
that future refactoring can preserve behavior intentionally.
