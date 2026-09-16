# Viability-Constrained Graph SMDP

## Status and purpose

This document defines the controller contract that follows the compact-yard
stress audit. It is a new primary method, not an imitation of dynamic PSLAP,
nearest-free, REG-v5, or another assignment policy. Baseline policies may be
used only as external comparisons.

The central separation is:

1. **Viability:** preserve the existence of a legal plan that completes the
   work already admitted to the yard.
2. **Liveness:** make progress rather than defer or cycle forever.
3. **Performance:** optimize timing, travel, throughput, and relocations inside
   the viable action set.

Safety is therefore not represented by a weighted reward penalty.

## Canonical transition system

The viability model is defined at atomic macro-decision boundaries. No block
is carried and no assignment is half committed at such a boundary. The
physical transfer action is

```text
Move(block, destination)
```

and is legal only when:

- the empty agent has a clear route to the selected block, treating every
  other live block as an obstacle;
- after lifting the selected block, the carried block has a clear route to the
  destination;
- a storage destination is an unoccupied, unreserved storage cell; or
- a delivery destination is a real exit.

The resulting parameterized control classes are:

- `Accept(block, cell)`: transfer the unique admitted inbound block to a cell;
- `Deliver(block, exit)`: transfer a stored block to an exit;
- `Reconfigure(block, cell)`: transfer a stored block to another storage cell;
- `Defer(horizon)`: advance time without changing the yard configuration.

Relocation preserves the original storage clock. Delivery may be early or
late; timing quality belongs to the task objective, not geometric viability.

This strict macro model is the authority for the new controller. The legacy
primitive environment currently allows an empty agent to cross inventory,
which is not the rule used by the deployed strict macros. Results must name the
action model under which a certificate was produced.

## Recovery state and information boundary

A recovery state contains:

- traversable, storage, exit, pickup, and waiting cells;
- agent position;
- occupied storage cells and their block identities;
- fixed live obstacles and reserved cells;
- optional remaining primitive-step budget.

The first certificate covers **current admitted-work recoverability under
closed admission**. It must not inspect future arrival times or processing
durations that are hidden from the online policy. Pickup and waiting cells can
be reserved as structural worst-case obstacles while external work remains.

This is narrower than complete-episode success. A finite evaluation horizon
must be passed explicitly before the certificate can imply completion within
that horizon.

## Recoverability and safe controls

Let `Goal` be the set of configurations with no currently admitted stored
blocks. For a macro budget `h`, define the backward-reachable sets

\[
K_0=\mathrm{Goal},
\]

\[
K_{h+1}=K_h\cup\{s:\exists c\in C_{\mathrm{phys}}(s),
F(s,c)\in K_h\}.
\]

The recoverability rank is

\[
\rho(s)=\min\{h:s\in K_h\}.
\]

The implementation distinguishes this minimum rank from an arbitrary valid
witness length. Breadth-first search supplies exact `rho`; the faster
goal-directed search supplies a sound feasibility witness but does not label
its length as an exact rank.

A control is viable when its entire macro is legal and its post-decision state
remains in the recoverable set:

\[
C_K(s)=\{c\in C_{\mathrm{phys}}(s):F(s,c)\in K\}.
\]

Temporary obstruction is allowed when a legal reconfiguration sequence can
remove it. Requiring every block to retain an immediate clear exit is therefore
not the definition of safety.

Exact search returns one of three results:

- `SAFE`: a complete executable witness was found;
- `UNSAFE`: exhaustive search proved that no witness exists within the stated
  physical or time horizon;
- `UNKNOWN`: a computational depth or node budget was exhausted.

`UNKNOWN` must never be reported as certified safe.

## Constrained mode-level SMDP

Let `C_g^K(s) = C_g(s) intersect C_K(s)`. The within-mode value is evaluated
only over that set:

\[
U_g^Q(s)=\tau_g\log\sum_{c\in C_g^K(s)}
\bar\mu_g(c\mid s)\exp(Q_g(s,c)/\tau_g).
\]

An empty viable candidate set makes that mode unavailable. The common SMDP
continuation value is then formed over the remaining modes. With an exact,
state-derived viable action set, this preserves the existing non-expansion and
contraction arguments while restricting optimality to recoverability-preserving
policies.

Viability alone does not prevent endless deferral or relocation cycles. Near a
viability boundary, the controller must select a recovery action that reduces
`rho`, or retain and execute a certified recovery witness. Away from the
boundary, the task return supplies the normal progress incentive.

## Graph representation and learned safety critic

The scalable representation is a variable-size cell/block graph evaluated
after applying each candidate action counterfactually:

- traversable cells are nodes connected by physical adjacency;
- exit, pickup, waiting, and storage roles are static node features;
- occupancy, reservation, agent position, remaining time, and urgency are
  dynamic features;
- candidate placement, delivery, or reconfiguration is represented in the
  post-decision graph.

The learned controller has separate heads:

```text
Q_task(state, action)       operational return
L_recovery(state, action)  reachability/recoverability
```

Selection is lexicographic or constrained: maximize task value only among
actions whose conservative recovery estimate passes the required threshold.
Training labels for `L_recovery` come from transition-model search and
counterfactual simulator states, not from baseline action choices.

Until a learned safety head has a separately validated error bound, it may
order exact checks but cannot replace the verifier in claims of certified
recoverability. Pruning is a separately labeled restricted-frontier ablation.

## Required audit fields

Every experiment using this controller records:

- transition/action-model version;
- information and arrival-disturbance contract;
- exact-search depth, node, and primitive-step budgets;
- counts of `SAFE`, `UNSAFE`, and `UNKNOWN` candidates;
- selected candidate certificate and witness length;
- epochs with no viable acceptance;
- recovery-rank changes and reconfiguration count;
- verifier time and cache statistics;
- whether an exact verifier or only a learned estimate was authoritative.

The primary method must record `baseline_viability_teacher = false`.

## Implementation stages

1. Exact compact-yard `RecoveryState` search and witness validation.
2. Explicit `Reconfigure(block, cell)` execution matching the search model.
3. Viability-constrained candidate construction and SMDP backup.
4. Counterfactual graph encoder and learned recovery head.
5. Dynamics-generated curriculum over layouts, loads, and near-boundary states.
6. Calibration stress tests, followed only then by the sealed holdout.

## Implemented controller contract

The runnable `v1` controller now uses:

- one shared graph encoder over both the current state and each certified
  counterfactual successor;
- the current embedding, successor embedding, their difference, and neutral
  action parameters as the input to one task-Q head;
- `Accept`, `Recover` (`Deliver` or `Reconfigure`), and `Defer` modes;
- a cardinality-normalized log-mean-exp within every live mode and over the
  live modes;
- one common duration-aware SMDP target for every mode; and
- regularized Double-Q evaluation, with the online network defining the soft
  target policy and the target network evaluating it.

Finite-temperature log-mean-exp is the training operator. Deployment is its
deterministic MAP realization: choose the mode with maximum regularized mode
value, then the maximum-Q candidate in that mode. It is therefore an off-policy
regularized value-learning method; the finite-temperature value is not claimed
to be exactly the value of the deterministic deployment policy.

The liveness state is explicit. After a bounded number of nonprogress recovery
choices, the controller retains and executes a complete dynamics-search
witness. Its active flag, nonprogress counter, remaining witness length, and
forced-frontier flag are part of the learned action record. The same restricted
next frontier is stored in replay, so deployment and Bellman continuation do
not silently use different action sets.

## Current calibration evidence

Exhaustive enumeration of the 5x5 compact post-transfer kernel produced 1,280
states (occupancy zero through eight): 1,261 `SAFE`, 19 `UNSAFE`, and zero
`UNKNOWN` under a 20,000-node goal-directed search budget. Unsafe states first
appear at occupancy five. At occupancy eight, four agent positions are safe
and five are unsafe despite identical storage occupancy, demonstrating that
recoverability depends on route topology and agent position rather than a
cell-count reserve alone.

The small end-to-end calibration smoke completed all three training and two
fresh evaluation episodes with strict success. The denser eight-block smoke
also completed, used five reconfigurations, and activated the retained-witness
guard twice (six forced witness steps) without a mismatch. These runs establish
interface and learning-loop execution, not comparative performance.

## Dynamics-only viability-critic calibration

`train_viability_critic.py` supplies the missing amortized-critic pipeline.
It procedurally generates compact connected layouts, samples load-boundary
states and reservations, and labels the complete serialized recovery state
with dynamics search. Goal-directed search supplies sound feasibility and
witness-step targets. A deterministic subset of safe states is rechecked with
breadth-first search; only those certificates supervise the exact recovery-rank
head. `UNKNOWN` records remain in the dataset audit but are absent from every
supervised loss.

The four partitions are disjoint by static layout:

- `train` fits independently initialized, independently bootstrapped critics;
- `validation` selects the epoch using a frozen false-safe-first rule;
- `calibration` selects the ensemble-LCB screening threshold;
- `test` independently checks the frozen threshold and reports a one-sided
  binomial false-safe upper confidence bound.

Run the planned seed-zero fit with:

```bash
cd /home/ai_diagnosis/HRL_Management

PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  nice -n 10 .venv/bin/python -u train_viability_critic.py \
  --output-dir results/viability-critic-v1-seed0 \
  --seed 0 --data-seed 81000 --split-seed 82000 \
  --grid-sizes 4x4 4x5 5x4 5x5 5x6 6x5 \
  --layouts-per-size 6 --states-per-layout 512 \
  --max-wall-fraction 0.20 \
  --search-max-nodes 100000 \
  --rank-label-fraction 0.10 --rank-search-max-nodes 100000 \
  --reservation-probability 0.20 --max-reservations 2 \
  --timing-scale 100 \
  --ensemble-size 5 \
  --graph-hidden-dim 64 --graph-embedding-dim 64 \
  --message-passing-steps 4 --head-hidden-dim 64 \
  --epochs 60 --steps-per-epoch 100 \
  --batch-size 128 --eval-batch-size 256 --eval-every 5 \
  --learning-rate 3e-4 --weight-decay 1e-5 \
  --false-safe-weight 10 --unsafe-sampling-fraction 0.50 \
  --lcb-scale 2 --calibration-confidence 0.95 \
  --max-false-safe-upper-bound 0.10 \
  --min-calibration-safe-accepted 10 \
  --device cuda
```

The process prints each completed layout and every training epoch. It writes
the authenticated JSONL dataset and split manifest under `data/`, a resumable
`latest.pth`, a deployment-only `best.pth`, `training-history.json`,
`calibration.json`, and `training-summary.json`. Resume an interrupted fit by
repeating the same configuration with:

```bash
--resume results/viability-critic-v1-seed0/latest.pth
```

Data can instead be generated and inspected first with `--generate-only`.
Restart the same output directory from those authenticated data using
`--reuse-data` and the otherwise identical training arguments.

Even a confirmed calibration result is empirical prioritization evidence. The
checkpoint explicitly records `critic_certificate_authority=false` and
`exact_verifier_authoritative=true`; no finite-data confidence interval turns
the ensemble LCB into a formal recoverability certificate.

The completed seed-zero artifact selected threshold `0.366068`. Its frozen
threshold was independently confirmed on the held-out test layouts with zero
false-safe predictions among 56 exact `UNSAFE` examples and safe coverage
`0.999`. The deployment artifact is
`results/viability-critic-v1-seed0/best.pth`, with pinned SHA-256
`19dfe16d637d6bdb06d6fbbc8aeb6c9605f5ba75dfa4996adf1f577fb0ef4e71`.
These are calibration results, not certificate authority.

## Exact-full critic prioritization

The critic is integrated under
`calibrated_lcb_full_permutation_v1`. At each strict boundary it scores only
uncached post-decision states, orders threshold-passing states first and then
by descending LCB, and sends every state through exact recoverability search.
The implementation then restores the original canonical candidate order
before graph-Q evaluation. Therefore:

- every executable candidate is still backed by an exact `SAFE` certificate;
- `UNSAFE` and `UNKNOWN` remain fail-closed even when the critic ranks them
  first;
- the regularized complete frontier and selected control are unchanged; and
- cold-cache exact-verifier calls cannot decrease under this contract.

The last point is structural. Early stopping or permanently dropping a
below-threshold state would define a restricted-frontier ablation, not the
same cardinality-normalized Bellman operator. Any call-saving version must be
reported separately. A promising semantics-preserving direction is to use the
critic to guide node expansion *inside* exact witness search while retaining
exhaustive proof requirements for `UNSAFE`.

Run the hierarchy with the pinned critic ordering using:

```bash
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u train_viability_graph_smdp.py \
  --output-dir results/vcg-smdp-priority-calibration \
  --episodes 3 --eval-episodes 2 \
  --grid-rows 5 --grid-cols 5 --number-blocks 2 \
  --search-order goal_directed --search-max-nodes 20000 \
  --viability-critic-checkpoint \
    results/viability-critic-v1-seed0/best.pth \
  --viability-critic-expected-sha256 \
    19dfe16d637d6bdb06d6fbbc8aeb6c9605f5ba75dfa4996adf1f577fb0ef4e71 \
  --viability-critic-device cuda \
  --device cuda
```

The paired benchmark reloads the same frozen controller separately for
`exact_full` and `critic_ordered_full`, uses the same saved
`EpisodeInstance`, gives each arm an empty private certificate cache, and
fails if frontiers, actions, trajectories, outcomes, cache hits, or exact
misses differ:

```bash
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u benchmark_viability_critic_priority.py \
  --controller-checkpoint \
    results/vcg-smdp-contention-smoke-v2/checkpoint.pth \
  --critic-checkpoint results/viability-critic-v1-seed0/best.pth \
  --critic-expected-sha256 \
    19dfe16d637d6bdb06d6fbbc8aeb6c9605f5ba75dfa4996adf1f577fb0ef4e71 \
  --output-dir results/viability-priority-paired \
  --seeds 74000 74001 74002 \
  --max-steps 4000 \
  --device cuda
```

A deliberately truncated, single-thread CPU semantic smoke on contention seed
74000 passed all equivalence invariants and made 72 exact cache misses in both
arms. Total frontier time was `0.0529 s` for canonical exact search and
`0.1408 s` for critic ordering, of which `0.0814 s` was critic inference. An
oversized default CPU thread pool made this small-network inference much
slower; CPU runs should set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. This smoke
validates integration semantics only. The GPU paired run above is the
appropriate primary systems measurement; it still should not be expected to
reduce the number of exact checks.

## Matched complete-system baseline diagnostic

`compare_viability_graph_baselines.py` compares canonical `exact_full` VCG
against complete online baseline systems on the same saved
`EpisodeInstance`. Each baseline uses the source-neutral duration-aware
reserved-cell scheduler with either nearest-free, dynamic PSLAP, or the
online 2009-style rolling GA assignment source. The critic is disabled for
VCG because it is behavior-equivalent and slower. Offline GA and regime-shifted
REG-v5 are excluded from the primary group.

The first three-instance plumbing run used the 5x5, eight-block, arrival-10,
processing-80 contention regime:

| Complete system | Return | Strict success | Steps | MAE | Tardiness | Within window | Relocations |
|---|---:|---:|---:|---:|---:|---:|---:|
| VCG exact-full | 128.67 | 1.000 | 187.0 | 66.625 | 0.375 | 0.125 | 0 |
| Duration-aware + nearest-free | 164.82 | 1.000 | 168.7 | 45.292 | 0.000 | 0.250 | 0 |
| Duration-aware + dynamic PSLAP | 225.88 | 1.000 | 168.0 | 19.208 | 1.333 | 0.625 | 2 |
| Duration-aware + rolling GA | 153.56 | 0.667 | 136.3 | 14.375* | 0.188* | 0.688* | 2 |

`*` Rolling-GA timing statistics cover its two completed episodes. On seed
74001 it failed under the strict no-fallback protocol because it could not
construct a unique assignment for all arrived pending jobs.

This table is deliberately non-claim-bearing. The available VCG contention
checkpoint has one training episode, 25 transitions, 18 gradient steps, zero
target-network updates, and saved epsilon `0.816`. Its large early-delivery
bias diagnoses missing task learning rather than a viability failure. A formal
comparison requires independently trained checkpoints selected on disjoint
validation instances, followed by the predeclared 50-instance paired in-regime
test panel (77000--77049) across all three learned training seeds.

Reproduce the plumbing comparison with:

```bash
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u compare_viability_graph_baselines.py \
  --controller-checkpoint \
    results/vcg-smdp-contention-smoke-v2/checkpoint.pth \
  --methods \
    vcg_smdp \
    duration_aware_nearest_free \
    duration_aware_dynamic_pslap \
    duration_aware_pslap_ga_2009_rolling \
  --seeds 74000 74001 74002 \
  --instances-dir results/viability-priority-paired/instances \
  --max-steps 2000 \
  --device cuda \
  --output-dir results/vcg-baseline-plumbing-smoke-3seed
```

## Proper contention training protocol

`train_viability_graph_smdp_proper.py` is the claim-preparation trainer. It is
separate from the calibration smoke runner and adds the lifecycle guarantees
that the one-episode checkpoint lacked:

- 500 unique training `EpisodeInstance`s per model seed;
- a fixed, disjoint 20-instance checkpoint-selection panel every 25 episodes;
- a frozen greedy validation clone with no replay, learning, exploration, or
  mutation of the training RNG and clocks;
- a fresh exact-certificate cache for every training and validation episode;
- atomic `latest.pth` (optimizer, replay, RNG, and clocks) and deployment-only
  `best.pth`, with authenticated resume;
- exact-full verification with no critic, teacher, baseline query, or fallback;
  and
- explicit refusal to open the in-regime 77000--77049 or stress 69000--69009
  test panels.

The three planned training replicas use model seeds 0, 1, and 2 and disjoint
training namespaces beginning at 30,000,000, 31,000,000, and 32,000,000.
Every replica uses validation seeds 75000--75019; those instances are
development data and cannot be reused for final effect estimates.

For seed 0, the full command is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 \
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u train_viability_graph_smdp_proper.py \
  --output-dir results/vcg-contention-v1-seed0-500ep \
  --model-seed 0 \
  --total-episodes 500 \
  --device cuda
```

The defaults freeze the 5x5/eight-block/arrival-10/processing-80 regime, the
64/64 three-pass graph network, batch 128, replay 20,000, one update per macro,
target synchronization every 200 gradient steps, and decision-based epsilon
0.90 to 0.05 over 10,000 macro decisions. To resume an interrupted run, repeat
the identical command and add:

```bash
--resume results/vcg-contention-v1-seed0-500ep/latest.pth
```

Checkpoint selection is lexicographic: strict success, completion, normalized
return, lower MAE, lower relocation rate, then the earlier episode. A selected
checkpoint is deployment-eligible only if all validation episodes finish with
100% strict success and all blocks delivered. The trainer never authorizes or
runs a formal test; all three independently selected model-seed checkpoints
must be frozen first.

Run the compact smoke with:

```bash
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u train_viability_graph_smdp.py \
  --output-dir results/vcg-smdp-calibration-smoke-v1 \
  --episodes 3 --eval-episodes 2 \
  --grid-rows 5 --grid-cols 5 --number-blocks 2 \
  --search-order goal_directed --search-max-nodes 20000 \
  --device cpu
```

## Scope that remains

The exact online mask certifies current admitted work under closed admission;
it is not a complete-episode certificate and does not reveal future schedules.
The graph viability critic is an amortized estimate trained from dynamics
labels. Until its false-safe error is separately bounded, it may prioritize
the order of exact checks but cannot replace the exact verifier in certified
claims. Larger-layout training and contention holdouts therefore come after
verifier/critic calibration, not before it.

## Runnable calibration smoke

The baseline-free integration can be exercised end to end with:

```bash
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u train_viability_graph_smdp.py \
  --output-dir results/vcg-smdp-calibration-smoke \
  --device auto
```

The compact defaults perform one training and one greedy evaluation episode,
write `checkpoint.pth` and `results.json`, and make at least one replay update.
The runner executes only exactly certified `ExplicitAccept`, `DirectDeliver`,
`Reconfigure`, and `ExplicitDefer` options. An empty safe frontier, an
`UNKNOWN` certificate, or a retained-witness mismatch is a method failure;
there is no baseline fallback.

This runner is calibration-only and refuses the declared `stress_v1` sealed
holdout seed namespace. Its certificate scope is current admitted work under
closed admission. It does **not** claim a complete-episode guarantee for
future arrivals or an external evaluation horizon. The learned viability
critic is not certificate-authoritative in this integration; exact dynamics
search remains the hard mask.
