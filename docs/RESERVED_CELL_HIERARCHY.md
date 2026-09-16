# Reserved-cell hybrid hierarchy

## Status and scope

This document defines the production Track-B controller that combines the
duration-aware deterministic scheduler with a frozen REG-v5 spatial selector.
Its defining contract is that an accepted inbound macro executes the exact
storage cell proposed at the scheduler decision epoch. The previous
post-pickup recomputation path remains available only as an internal ablation.

The method is a **full hybrid hierarchy**, but it is not a jointly trained or
fully learned hierarchical reinforcement-learning system. Its hierarchy is:

```text
duration-aware deterministic temporal manager
├── AcceptStore(block, cell)
│   ├── frozen REG-v5 supplies cell
│   └── ReservedAcceptStoreOption-v2 executes that exact cell
├── StrictRetrieveDeliver(block)
│   └── strict atomic retrieval executor with neutral relocation
└── DeferUntilEvent
    └── bounded event-driven defer executor
```

The temporal manager determines which temporally extended macro should run.
REG-v5 determines the spatial parameter of an inbound macro. Deterministic
atomic executors then realize the selected macro safely. These are distinct
levels of control with distinct responsibilities, even though only the spatial
level contains a learned model.

## Why the reserved-cell contract is necessary

In the legacy v1 path, the duration-aware scheduler previews a storage cell at
the current decision epoch and uses it to estimate the duration and
counterfactual effect of accepting the inbound block. `AcceptStoreOption-v1`
then navigates to the block, picks it up, and calls the assignment source again.
That second decision is made several environment steps later. Time-dependent
features, observed queue state, and stored-block timing state can change during
the pickup leg, so the second cell need not be the cell evaluated by the
scheduler.

The v2 path removes this semantic gap. The scheduler evaluates a parameterized
macro `AcceptStore(block, cell)`, binds the proposal token if it selects that
macro, and the executor later commits the token without calling the assignment
source again. Consequently, the spatial counterfactual used by temporal
lookahead and the spatial action that is executed are the same.

This decision timing also matches REG-v5's training support. Track A invokes
the selector before navigation and pickup, then executes the selected placement.
The selector canonicalizes the current block's carrying-status coordinates, but
a real pickup leg still advances time, ages stored blocks, and may expose new
arrivals or promote a waiting block. Post-pickup recomputation therefore changes
meaningful model inputs and can create a queue configuration absent from Track-A
assignment epochs. V2 scores once at the native pre-pickup epoch; v1 is the
phase-shifted path.

Binding and physical mutation are intentionally distinguished:

- The **hierarchical action is bound** at the scheduler decision epoch.
- `block.storage_location` remains unset during the navigation-to-pickup leg.
- Immediately after pickup, the executor revalidates and commits the bound
  token, setting `block.storage_location` without rescoring or substitution.

This preserves an atomic pickup-and-store executor while giving the manager a
well-defined action parameter at selection time.

## Formal SMDP interpretation

Let `s` be an observable Track-B decision state and let `b_in(s)` denote the
currently admissible inbound block, when one exists. The controller has three
families of temporally extended decisions:

\[
\mathcal{M}(s)=
\mathcal{R}(s)\;\cup\;\mathcal{D}(s)\;\cup\;\mathcal{S}(s),
\]

where:

- \(\mathcal{R}(s)\) contains executable
  `StrictRetrieveDeliver(block)` macros;
- \(\mathcal{D}(s)\) contains the event-driven defer macro when admissible;
- \(\mathcal{S}(s)\) contains a parameterized inbound action
  `AcceptStore(b_in(s), c)`.

For an inbound block, the shared safety mask supplies the variable candidate
set \(F(s,b)\). Frozen REG-v5 supplies one parameter:

\[
c_\theta(s,b)=
\arg\max_{c\in F(s,b)} Q_{\text{REG}}(s,b,c).
\]

The deterministic temporal manager evaluates the resulting macro

\[
m_{\text{store}}(s)=
\operatorname{AcceptStore}(b_{\text{in}}(s),c_\theta(s,b_{\text{in}}(s)))
\]

against admissible retrieval or defer macros. If it selects acceptance, the
pair `(block, cell)` is bound and cannot be recomputed. If it selects retrieval,
the unbound proposal is discarded and has no environmental effect.

Each selected macro runs for a variable duration \(k\), receives its cumulative
environment reward during execution, and returns control at the next scheduler
decision epoch. It is therefore an SMDP action. The policy can be factored as a
deterministic temporal gate over a learned spatial proposal:

\[
c=c_\theta(s,b),\qquad
m=\pi_H(s;c),\qquad
\pi_E(\cdot\mid s,m,c)
\]

where \(\pi_H\) is the deterministic duration-aware manager and \(\pi_E\) is
the relevant deterministic atomic executor. This is a parameterized hierarchy,
not a flat comparison between separately calibrated primitive and option
critics.

### Duration-aware temporal decision

Already-due executable retrievals have strict precedence and bypass REG
proposal generation. Otherwise, when an inbound block and an executable
retrieval coexist, the manager compares the absolute delivery error predicted
for retrieving the head job now with the error predicted after executing the
exact proposed `AcceptStore(block, cell)` macro. With margin `m`, retrieval is
selected when

```text
abs(error_retrieve_now) <= abs(error_after_accept) + m
```

so equality favors retrieval. The post-accept yard projection places the
inbound block at the bound cell and ages existing stored blocks by the estimated
pickup-and-storage duration. If no retrieval is executable, v2 still generates
and binds a proposal before accepting; there is no proposal-free accept branch.

## Components and learned status

### Frozen REG-v5 spatial subpolicy

REG-v5 is a cardinality-invariant candidate scorer trained previously on the
Track-A assignment problem. In this Track-B deployment:

- its checkpoint is loaded once and its deployment digest is recorded;
- learning and exploration are disabled;
- it sees only the current online/arrived information regime;
- it scores candidates from the shared candidate mask;
- one side-effect-free greedy proposal is evaluated per proposal epoch;
- an accepted proposal is committed without a second network call.

REG-v5 is not updated from Track-B returns in this method. The hierarchy is
therefore not trained end-to-end.

### Deterministic temporal manager

The manager is a hand-specified duration-aware policy. It implements due-first
retrieval, one-step absolute-timing lookahead, capacity-release retrieval, and
event-driven defer. Its lookahead margin is fixed by the evaluation contract.
It is not a learned manager or critic.

### Reserved inbound executor

`ReservedAcceptStoreOption-v2` navigates to the named inbound block, picks it
up, commits the already-bound proposal, navigates to the reserved cell, and
puts the block down. Assignment is a zero-duration controller operation between
the pickup and first storage movement; it does not introduce an artificial
`WAIT` transition.

### Strict retrieval and defer branches

There is one named `StrictRetrieveDeliverOption-v2` per manifest block. Its
initiation contract requires a canonical retrieval plan with a complete live
first leg. Retrieval may use the fixed neutral relocation rule
`nearest_feasible_path_then_cell_v1`; failures and relocation counts are
audited. `StrategicDeferOption-v1` waits only until an observable event or its
configured bound. Neither branch calls the storage selector.

## Proposal token and provenance

`StorageAssignmentPreview` is the immutable proposal token. It contains:

| Field | Meaning |
|---|---|
| `contract` | Source-specific read-only preview contract. |
| `proposal_id` | Stable 20-hex-character identifier for this exact proposal. |
| `instance_id` | Exact geometry-dependent `EpisodeInstance` identity. |
| `assignment_source` | Source name, such as `reg_selector_v5`. |
| `assignment_source_version` | Exact source implementation/model version. |
| `block_label` | Inbound block parameter. |
| `time_step` | Scheduler epoch at which the proposal was produced. |
| `source_cell` | Block location from which placement is planned. |
| `candidate_count` | Number of cells in the shared mask. |
| `candidate_mask_id` | 16-hex-character digest of the ordered candidate mask. |
| `chosen_cell` | Spatial parameter supplied to `AcceptStore`. |

The proposal identifier hashes the instance ID, source identity and version,
block label, time step, source cell, candidate-mask ID, and chosen cell. It
therefore joins the same decision across scheduler, selector, store-event, and
option-outcome audits without relying only on block order.

`AcceptStoreDurationEstimate` carries the proposal token together with the
estimated pickup steps, storage steps, total macro duration, preview duration,
candidate provenance, and proposed cell. The scheduler binds this complete
estimate, ensuring that its temporal counterfactual and the executor's spatial
parameter share one provenance object.

The following records expose the lifecycle:

- scheduler decision: `preview_proposal_id`, proposed cell, mask ID, and
  estimated durations;
- selector decision: proposal ID, selection and commit time steps, proposal
  and live mask IDs, chosen cell, and commitment contract;
- store event: phase `reserved_chosen` with proposal ID and cell;
- inbound option outcome: proposal ID, reserved cell, mask ID, proposal epoch,
  and execution-match flag;
- episode audits: ordered bound and committed proposal-ID lists and reservation
  counters.

## Lifecycle

1. At a scheduler epoch, strict due retrieval is checked first. A selected due
   retrieval produces no storage proposal or reservation.
2. For an admissible inbound comparison, the assignment source makes one
   side-effect-free proposal under the current shared mask.
3. The option builds a duration estimate for that exact cell. The scheduler's
   projected post-placement retrieval uses the same cell.
4. If retrieval is selected, the proposal is discarded. It is never committed
   or counted as a selector assignment.
5. If acceptance is selected, `bind_estimate` validates the instance, source,
   block, current epoch, source cell, mask, cell, and proposal digest. The
   estimate becomes the option's sole active reservation.
6. The reserved option must start at the bound epoch. It revalidates the token
   before executing the pickup leg.
7. After successful pickup, `commit_preview` validates the live shared mask and
   token provenance without invoking the source. It writes exactly the bound
   cell to `block.storage_location`.
8. The option plans and executes the storage leg, puts the block down, records
   whether the executed cell matches the reservation, and clears the active
   reservation.
9. Environment reset also clears reservations, committed-token state, and all
   episode reservation counters. Tokens from a previous instance cannot be
   reused.

## Strict failure semantics

The production path has no assignment fallback and never silently substitutes
another cell.

- A missing, malformed, outside-mask, or source-invalid proposal makes the
  scheduler explicitly infeasible.
- A token with the wrong instance, source, source version, block, epoch, source
  cell, candidate count, mask digest, chosen cell, or proposal digest is
  rejected.
- A proposal may be bound only once and committed only once.
- If the macro does not start at its bound epoch, it terminates with
  `reserved_assignment_not_started_at_bound_epoch` before pickup.
- If live validation changes before pickup, the option fails explicitly before
  carrying inventory.
- If the proposal becomes invalid after pickup, commitment fails; the source is
  not called again and no alternative cell is selected.
- A missing safe path to the bound cell is a macro runtime failure, not a
  reason to rescore.
- Retrieval and defer retain their own strict completion-or-explicit-failure
  contracts.

An episode is a strict success only if it terminates normally with no method
failure, invalid assignment, fallback, inbound failure, retrieval failure, or
illegal drop, and the reservation-integrity checks all pass.

## Versioned contracts

### Production duration-aware hierarchy

| Contract | Identifier |
|---|---|
| Assignment commitment CLI value | `decision_epoch_reserved` |
| Method | `reg_v5_reserved_cell_hierarchical` |
| Scheduler architecture | `deterministic_duration_aware_reserved_cell_hierarchy_v3` |
| Policy realization | `deterministic_bound_cell_duration_lookahead_v3` |
| Controller action interface | `interleaved_atomic_scheduler_reserved_cell_v4` |
| Accept controller ID | `option:AcceptStore:v2` |
| Accept option version | `accept_store_reserved_cell_v2` |
| Commitment contract | `decision_epoch_proposal_bound_once_v2` |
| Reservation contract | `bound_cell_parameterized_accept_store_v1` |
| Decision-epoch contract | `due_then_exact_bound_cell_lookahead_then_capacity_then_event_v3` |
| Accept-duration estimate | `exact_bound_cell_current_epoch_path_with_handling_v3` |
| Source-neutral method | `reserved_cell_assignment_source_ablation` |
| Source-neutral policy | `deterministic_bound_assignment_duration_lookahead_v3` |
| Strict retrieval option | `retrieve_deliver_option_v3` |
| Strict retrieval executor | `named_atomic_retrieval_executor_v3` |
| Defer option | `strategic_defer_until_event_v1` |
| Relocation selector | `nearest_feasible_path_then_cell_v1` |

### Due-only reserved-cell control

| Contract | Identifier |
|---|---|
| Method | `reserved_cell_urgency_first_atomic` |
| Scheduler architecture | `deterministic_urgency_first_reserved_cell_v2` |
| Policy realization | `deterministic_due_first_bound_cell_v2` |
| Decision-epoch contract | `due_retrieval_then_bound_inbound_then_capacity_then_event_v2` |

The due-only control uses the same frozen selector, proposal token,
`AcceptStore-v2`, strict retrieval executor, defer option, relocation rule, and
environment contract. Only the temporal scheduling rule changes.

### Legacy recomputation ablation

The compatibility value `post_pickup_recompute` retains:

- controller interface `interleaved_atomic_scheduler_v3`;
- controller ID `option:AcceptStore:v1`;
- option version `accept_store_option_v1`;
- commitment contract `post_pickup_recompute_v1`.

This version previews for lookahead but recomputes after pickup. It is retained
to quantify the effect of binding, reproduce older experiments, and diagnose
preview/commit drift. It is not the production hierarchical method, and its
identifier must not be exchanged with v2 in checkpoint or protocol metadata.

## Structural invariants

For every complete strict-success episode under the reserved contract:

1. Each accepted inbound macro has exactly one bound proposal.
2. Each bound proposal has exactly one committed selector decision.
3. Each committed proposal produces exactly one inbound option outcome.
4. Bound proposal IDs are unique and their ordered list equals the ordered
   committed proposal-ID list.
5. The committed proposal-ID list equals the valid reserved selector-decision
   list.
6. Reservation bound count, commit count, execution-match count, inbound
   success count, and completed inbound assignment count are equal.
7. Reservation invalidation count is zero.
8. Every successful executed storage cell equals the corresponding bound cell.
9. The source is evaluated once for a bound spatial decision and is not called
   after pickup.
10. A retrieval-selected proposal is never committed.
11. A due-retrieval decision invokes neither assignment preview nor binding.
12. No reservation or committed-token state survives environment reset.
13. Invalid assignments, fallbacks, illegal drops, and silent cell
    substitutions are all zero.

The evaluator reports `reservation_integrity` and includes it in strict-method
success. The scheduler, selector, and atomic-option records must be joined by
proposal ID before performance metrics are interpreted.

## Fresh paired evaluation protocol

### Fixed inputs

The reserved-cell method is evaluated with a fresh seed namespace. Results from
the legacy recomputation runs are not reused as reserved-cell confirmation.

- Development/integrity seeds: `47000`, `47001`, `47002`.
- Untouched confirmation seeds: `48000` through `48009`.
- REG checkpoint:
  `results/reg-selector-v5-seed0-local-500ep/best.pth`.
- Arrival and processing regime: `lambda=0.5`, `mu=50`.
- Yard: 10 by 10.
- Ordinary condition: the environment's ordinary default bottom gate.
- Constrained condition: right-aligned bottom egress of width 2.
- Maximum episode steps: 4000.
- Maximum defer steps: 10.
- Duration-aware lookahead margin: 2 steps.
- Delivery target window: 20 steps.
- Bootstrap samples: 10,000.
- Assignment commitment: `decision_epoch_reserved`.
- Information regime: online arrived-only.
- Candidate contract: `shared_candidate_mask_v1`.
- Assignment failure contract: invalid or missing proposal is an explicit
  failure, with no fallback.

Within a geometry, sources consume the same exact `EpisodeInstance` and pair by
`instance_id`. Across geometries, ordinary and constrained instances share the
same arrivals and processing durations and pair by `schedule_id`, which
deliberately excludes geometry.

### Assignment-source factor

The fixed reserved-cell duration-aware hierarchy is run with:

- `reg_selector_v5` as the learned spatial source and reference;
- `nearest_free` as a simple deterministic spatial source;
- `dynamic_pslap` as the primary deterministic PSLAP comparator.

Each source controls both the spatial proposal used by lookahead and the bound
cell executed after acceptance. This is therefore a comparison of spatial
sources embedded in the same temporal hierarchy. It does not isolate a cell
choice from all downstream scheduler interactions.

The unified development run is:

```bash
cd /home/ai_diagnosis/HRL_Management

PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u compare_track_b_hierarchy.py \
  --selector-checkpoint results/reg-selector-v5-seed0-local-500ep/best.pth \
  --seeds 47000 47001 47002 \
  --lambda 0.5 --mu 50 \
  --grid-rows 10 --grid-cols 10 \
  --max-steps 4000 --max-defer-steps 10 \
  --lookahead-margin-steps 2 --target-window 20 \
  --constrained-exit-width 2 \
  --bootstrap-samples 10000 --device cpu \
  --output-dir results/track-b-hierarchy-dev-47000-47002
```

Development seeds are used to verify execution, provenance, reservation
integrity, geometry pairing, and output completeness. They must not be used to
retune the checkpoint, gate width, lookahead margin, reward, source contracts,
or confirmation analysis.

After the development integrity gate passes, the untouched confirmation is:

```bash
cd /home/ai_diagnosis/HRL_Management

PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u compare_track_b_hierarchy.py \
  --selector-checkpoint results/reg-selector-v5-seed0-local-500ep/best.pth \
  --seeds 48000 48001 48002 48003 48004 \
    48005 48006 48007 48008 48009 \
  --lambda 0.5 --mu 50 \
  --grid-rows 10 --grid-cols 10 \
  --max-steps 4000 --max-defer-steps 10 \
  --lookahead-margin-steps 2 --target-window 20 \
  --constrained-exit-width 2 \
  --bootstrap-samples 10000 --device cpu \
  --output-dir results/track-b-hierarchy-confirmation-48000-48009
```

Primary outcomes are strict success, return, absolute timing error, target
window rate, tardiness, steps, storage-flow time, retrieval relocations,
retrieval replans, and obstructive moves. The primary interaction is

```text
(REG - dynamic PSLAP advantage under constrained egress)
- (REG - dynamic PSLAP advantage under ordinary egress)
```

paired by `schedule_id`. Nearest-free is secondary. Reservation-integrity and
relocation-activation summaries are manipulation checks, not optional
diagnostics.

### Due-only temporal control

A matched due-only control is required to identify the contribution of the
duration-aware lookahead. It uses frozen REG-v5 and the exact same reserved-cell
spatial and execution contracts, but replaces
`ReservedCellDurationAwareAtomicScheduler` with
`ReservedCellUrgencyFirstAtomicScheduler` (`scheduler_variant=due_only`).

The unified runner executes this control on the same saved development and
confirmation instances in both geometries and under both commitment contracts.
It pairs duration-aware and due-only outcomes within geometry by `instance_id`
and across geometries by `schedule_id`. The source is REG-v5 in these cells;
only the temporal rule changes within a commitment condition.

The control reports the same performance and integrity fields as the
duration-aware method, plus scheduler reason counts. Reserved due-only cells
still require one bound proposal, one commit, and one exact execution for every
accepted inbound macro.

## Completed results

The development and confirmation protocols above completed without retuning.
All intervals below are schedule-paired 10,000-sample bootstrap 95% intervals;
positive advantages favor the named left method.

### Integrity first

| Check | Development | Confirmation |
|---|---:|---:|
| Complete protocol cells | 48/48 | 160/160 |
| Strict successes | 48/48 | 160/160 |
| Reserved runs passing integrity | 24/24 | 80/80 |
| Bound = committed = execution-match | 960 | 3,200 |
| Reservation invalidations | 0 | 0 |
| Invalid assignments / fallbacks / illegal drops | 0 / 0 / 0 | 0 / 0 / 0 |
| Method failures | 0 | 0 |

All within-geometry runs used identical `instance_id` values, while paired
ordinary and constrained runs shared `schedule_id` and had distinct geometry-
dependent instance identities. The frozen selector deployment digest was
`6bb345a317b5b7350251c96722e267fbc530a8882d379b81655c7752408be793`.

The semantic intervention was active. In confirmation duration-aware REG runs,
legacy preview and executed cells matched only 96/371 times (25.9%) in ordinary
geometry and 116/366 times (31.7%) in constrained geometry. V2 matched 400/400
accepted assignments in each geometry. The live candidate mask stayed valid;
V2 removed post-pickup rescoring rather than repairing invalid cells.

### Production reserved hierarchy

Confirmation cell means are:

| Geometry | Source | Return | Absolute error | Within window | Steps | Mean storage flow | Relocations/episode |
|---|---|---:|---:|---:|---:|---:|---:|
| Ordinary | REG-v5 | 1461.25 | 3.94 | 1.000 | 1135.6 | 501.87 | 0.0 |
| Ordinary | Dynamic PSLAP | 1446.19 | 4.41 | 1.000 | 991.2 | 434.07 | 0.0 |
| Ordinary | Nearest-free | 857.62 | 15.73 | 0.745 | 940.9 | 401.31 | 0.0 |
| Constrained | REG-v5 | 1470.31 | 3.78 | 1.000 | 1146.6 | 510.31 | 0.0 |
| Constrained | Dynamic PSLAP | 1351.90 | 6.56 | 0.928 | 972.2 | 426.45 | 1.4 |
| Constrained | Nearest-free | 857.62 | 15.73 | 0.745 | 940.9 | 401.31 | 0.0 |

Under constrained egress, reserved REG-v5 versus reserved dynamic PSLAP had:

- return advantage `+118.40`, CI `[95.51, 139.90]`, wins 10/10;
- absolute-error reduction `2.785` steps, CI `[2.290, 3.313]`, wins 10/10;
- within-target-window advantage `+0.0725`, CI `[0.0525, 0.0950]`, wins 10/10;
- retrieval-relocation reduction `1.4` per episode, CI `[0.8, 2.0]`, with
  eight wins and two ties;
- no clear tardiness reduction (`-0.233`, CI `[-0.578, 0.085]`). Dynamic
  PSLAP's larger errors were often early rather than tardy;
- `174.4` more episode steps and `83.85` more arrival-to-storage flow steps.

In ordinary geometry the REG return advantage over dynamic PSLAP was `+15.05`,
CI `[-9.43, 39.68]`, so there is no clear return separation. REG still reduced
absolute error by `0.468`, CI `[0.067, 0.855]`, but used 144.4 more steps.

The primary cross-geometry difference-in-differences was strongly positive:

```text
(REG - dynamic under constrained egress)
- (REG - dynamic under ordinary egress)
= +103.35 return, 95% CI [69.12, 142.00], wins 10/10.
```

The corresponding absolute-error interaction was `2.318` steps, CI
`[1.655, 3.125]`, and the relocation interaction was `1.4`, CI `[0.8, 2.0]`.
This supports a specific learned-source advantage: REG's layout policy is
robust to a narrow egress and avoids the retrieval relocations incurred by
dynamic PSLAP. It does not establish a general throughput advantage.

### Temporal manager and commitment ablations

The duration-aware manager was decisively useful. With the production reserved
contract, duration-aware minus due-only return was:

- `+175.16`, CI `[133.95, 211.24]` in ordinary geometry;
- `+156.84`, CI `[120.64, 193.70]` in constrained geometry.

Both contrasts won all 10 schedules and reduced absolute error by 2.99 and 2.70
steps, respectively. The full result is therefore attributable to the
combination of deterministic temporal scheduling and learned spatial placement,
not REG-v5 alone.

Exact reservation did not itself increase duration-aware return. Reserved minus
legacy recomputation was `-25.90`, CI `[-49.64, -4.17]` in ordinary geometry,
and `-7.57`, CI `[-29.89, 13.83]` under constrained egress. It also added 46.1
and 65.8 steps, respectively. In the due-only constrained control, however,
reservation improved return by `+65.18`, CI `[11.25, 119.74]`, and reduced
absolute error by `1.43`, CI `[0.41, 2.42]`.

This distinction matters. V2 is adopted because it defines and executes the
same parameterized SMDP action, aligns with REG-v5's pre-pickup training epoch,
and has perfect provenance—not because semantic consistency automatically
improves every return. The legacy result shows that adaptive post-pickup
rescoring can sometimes improve outcomes, but that behavior is a different
two-stage controller. If retained as a production idea, it should be formalized
as `Acquire(block)` followed by a new `Store(cell)` decision and trained under
that interface, not described as one pre-bound `AcceptStore(block, cell)` macro.

### Artifacts

Development:

- `results/track-b-hierarchy-dev-47000-47002/hierarchy-results.csv`
- `results/track-b-hierarchy-dev-47000-47002/hierarchy-full.json`
- `results/track-b-hierarchy-dev-47000-47002/hierarchy-summary.json`
- `results/track-b-hierarchy-dev-47000-47002/hierarchy-audit.json`
- `results/track-b-hierarchy-dev-47000-47002/instances/`

Confirmation:

- `results/track-b-hierarchy-confirmation-48000-48009/hierarchy-results.csv`
- `results/track-b-hierarchy-confirmation-48000-48009/hierarchy-full.json`
- `results/track-b-hierarchy-confirmation-48000-48009/hierarchy-summary.json`
- `results/track-b-hierarchy-confirmation-48000-48009/hierarchy-audit.json`
- `results/track-b-hierarchy-confirmation-48000-48009/instances/`
