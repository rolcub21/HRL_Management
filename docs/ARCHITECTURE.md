# Architecture

## Runtime flow

```text
SmallRoomsEnv
    |
    +-- primitive options: movement, pickup, putdown, wait
    |
    +-- high-level options: pickup, store, retrieve/deliver, select storage
            |
            +-- DQNAgent manager chooses a high-level option
            +-- DQNAgent worker chooses primitive actions
            +-- selected option supplies initiation/policy/termination rules
```

The PSLAP-family baselines bypass the learned manager/worker decision process.
`dynamic_pslap` uses repaired yard-planning logic; `legacy_adapted_pslap`
preserves the pre-audit controller for internal ablation. The GA variant still
trains the manager and worker networks, but its storage selector reads a fixed
assignment.

## Core abstractions

### Environment

`environment.py` defines `BaseEnvironment`, the abstract interface used by the
option framework. `example/small_rooms_env.py` supplies the actual shipyard
environment as `SmallRoomsEnv`.

The default yard is a 10 by 10 grid with walls around the boundary, a pickup
position near the entrance, the configured delivery gate, and all eligible free
cells treated as storage positions. There are 40 blocks per episode.

At reset:

- each block receives a positive Poisson-distributed required storage duration
  with mean `proc_mean` (`mu`);
- block inter-arrival gaps are exponentially distributed with mean
  `1 / arrival_rate` (`1 / lambda`);
- the first block is available immediately; and
- storage locations begin unassigned unless `choose_storage=True`.

For common evaluation, these exogenous inputs are sampled before controller
construction and stored in a frozen `EpisodeInstance`. It contains the complete
arrival schedule, required storage durations, rate parameters, and exact yard
geometry. `SmallRoomsEnv.reset(instance=...)` validates and replays that object
without drawing a new schedule. Its `instance_id` is a stable hash of the
canonical serialized content and is the unit used to pair method results.

The terminal condition is that every block has been delivered.

### State representation

`SmallRoomsEnv.get_current_state()` currently returns:

```text
(agent_position, block_features)
```

Each block contributes:

```text
(
  status_one_hot,
  block_position,
  distance_agent_to_block,
  distance_block_to_storage,
  distance_block_to_exit,
  remaining_storage_time,
  path_is_clear,
  path_length,
)
```

Blocks that have not entered the yard use sentinel values such as `(-1, -1)`
and `-1`. `example/helper/tools.py` flattens this nested state before it is sent
to a neural network.

Known mismatch: the method's docstring also describes `t_next` and `n_wait`,
but the implementation does not return those two values. Documentation and
model dimensions should follow the implementation until this is resolved as a
deliberate code change.

### Reward and metrics

The implemented reward includes:

- a `-0.09` cost for every environment step;
- `+5` when a block is stored at its assigned location; and
- a delivery reward with base `10` and a timing bonus of up to `30`, decreasing
  with the absolute delivery-time error until the bonus reaches zero at an
  error magnitude of 20.

The active timing reward is symmetric: deviations `-e` and `+e` receive the
same delivery reward. A signed mean therefore measures early/late bias, not
timing accuracy. Evaluation reports signed deviation, absolute error,
earliness, tardiness, the rate within the `+/-20` target window, tardy rate,
90th-percentile tardiness/absolute error, and within-episode variability.

Inbound responsiveness is reported separately as storage flow time. For block
`i`,

```text
F_i = S_i - A_i,
```

where `A_i` is its exogenous `arrival_step` and `S_i` is the environment step
of its first successful storage PUTDOWN (`stored_time_step`). Thus `F_i`
includes queueing, scheduler deferral, travel to pickup, pickup, loaded travel,
and putdown. It is not pure queue wait, required storage duration, delivery
error, delivery lead time, or episode makespan. Relocation preserves the first
storage timestamp and cannot restart this clock.

At observation end `T`, an arrived but unstored block is right-censored with
age `T-A_i`; a block with `A_i>T` is counted as not yet arrived and has not
entered the flow-time risk set. Censor ages are never substituted for completed
flow times. The manifest count is partitioned into completed, right-censored,
and not-yet-arrived counts, with completion rates reported against both the
full manifest and arrived inventory. Mean, median, p90, p95, and maximum flow
time are defined only when every manifest block has completed storage;
otherwise these primary statistics are `NaN` in memory and `null` in JSON.
Completed raw times, censor ages, and manifest-aligned status records remain in
the audit.

The legacy agent can add another `-0.5` training-only WAIT penalty. Corrected
foundation, scheduler-v4, and atomic-scheduler-v5 training set this penalty to
zero, so replay contains the environment reward actually reported at
evaluation. Several old reward alternatives remain commented in the source;
they are not part of the primary method.

The legacy training field `episode_avg_error` is a signed deviation and must not
be interpreted as an error norm. Primary evaluation claims should use the
disaggregated timing fields above. Training also logs total episode return,
all-blocks-delivered success, manager loss, and worker loss.

### Options

`option.py` defines `BaseOption`; `primitive_option.py` adapts primitive actions
to that interface. Shipyard options live under `example/Options/`:

| File | Role |
|---|---|
| `pickupOption.py` | pick up an inbound block |
| `PickupRipeOption.py` | collect a block whose storage time is complete |
| `storeOption.py` | carry a block to its storage location |
| `DeliverOption.py` | carry a retrieved block to an exit |
| `AcceptStoreOption.py` | atomically accept, assign, and store one inbound block |
| `RetrieveDeliverOption.py` | atomically retrieve and deliver one named block |
| `StrategicDeferOption.py` | defer until an observed event, deadline, or cap |
| `waitOption.py` | wait when no useful operation is available |
| `selector.py` | learned storage-cell selector |
| `GAStorageSelectOption.py` | selector backed by a saved GA assignment |
| `random_selector.py` | random/experimental selector variant |

The mixed filename capitalization is historical and matters on Linux.

The maintained option contracts include the following safeguards:

- pickup targets only the inbound pickup cell;
- Store immediately puts down a block when it starts at its assigned cell and
  refuses an occupied assignment;
- retrieval and delivery ignore occupied or unreachable exits and can replan;
- GA assignments must contain one valid, unique cell per block; and
- per-option internal state is reset at initiation/termination boundaries.

The learned selector represents the carried block first, followed by the most
urgent active stored blocks. Time and global features are normalized, delivered
inventory is excluded from active congestion summaries, and per-episode usage
state is reset with the environment. Each storage assignment is trained as a
delayed contextual decision. The default selector return is the complete
environment reward stream from the storage-location decision until delivery:

```text
G_sel(t) = sum_{i=0}^{k-1} gamma^i r_env(t+i+1).
```

This stream includes the placement reward, intervening step rewards, and the
delivery step's base reward and timing bonus exactly once. The delivery record
is terminal for selector learning (`done=1`), so the state encoded at delivery
is retained only for diagnostics and is never treated as another admissible
storage decision or used for Q-value bootstrapping. Episode truncation likewise
closes each outstanding assignment with its discounted failure outcome.

`StorageSelectOption.RETURN_EXPLICIT_TERMINAL` retains the equivalent split
form for internal audits: the environment's delivery component is removed from
the accumulated stream and added once at the same discount on closure. The
default `RETURN_FULL_ENVIRONMENT` form is the manuscript-facing definition.

## Learning agent

`options_agent.py` contains the current `DQNAgent` and two Q-networks:

- the manager network scores non-primitive options;
- the worker network scores primitive options.

The agent uses epsilon-greedy selection, a prioritized manager replay buffer, a
worker replay buffer, target networks, soft target updates, gradient clipping,
and separate Adam optimizers. Default values exist in `DQNAgent`, while the
training entry points override several of them.

Manager-option and primitive-action output indices are canonicalized and saved
as checkpoint metadata. Evaluation verifies those lists before loading weights.
Legacy checkpoints without metadata are rejected by default because their head
indices cannot be reconstructed safely.

### Versioned controller semantics

The historical `DQNAgent` still compares manager and worker outputs while
training them with restricted within-head continuations. It is retained for
reproduction and is not the primary composite-SMDP claim.

`gated_agent.py` versions the audited alternatives. V1 uses a shared
regularized continuation; v2 uses hard maxima within heads and a regularized
two-mode gate; v3 corrects observation, normalization, reward, and truncation
contracts; and scheduler-v4 introduces manifest-indexed retrieval jobs while
retaining split inbound and primitive controls. These remain internal
ablations.

The primary Track B controller is `atomic_inbound_scheduler_v5`. It rejects a
primitive/option comparison at scheduling epochs and instead defines a single
macro-action SMDP. Its 42 manager outputs are one forced `AcceptStore`, one
`DeferUntilEvent`, and 40 named `RetrieveDeliver` jobs. Primitives are never
admissible at a v5 decision or replay continuation. Its target is therefore

```text
y_M = R^(k) + gamma^k (1-d) max_{c in C_macro(s')} Q_M^-(s', c).
```

Inbound acceptance is structurally mandatory when feasible. Frozen REG-v5 is
called exactly once between pickup and the first storage action, with no
environment-time assignment step. Only genuine Retrieve-versus-Defer epochs
advance epsilon. The inactive worker network is currently kept in checkpoints
for base-class compatibility; it receives no transitions or updates.

The repaired no-training candidate is
`deterministic_urgency_first_atomic_v1`. It retains frozen REG-v5 assignment
but replaces the learned 42-way dispatch with the following online rule:

1. retrieve the executable job with the lowest nonpositive slack;
2. otherwise accept a feasible inbound block;
3. if inbound acceptance is infeasible, retrieve the closest-to-due
   executable job to release capacity;
4. otherwise defer to an observed arrival/deadline event.

Ties use relocation count, canonical complete-plan duration, and manifest
index. The ETA is explicitly the stored-yard canonical estimate. Strict
retrieval v2 separately validates that the first full live leg is executable;
it does not claim that the canonical ETA is a live-route duration. If a due job
exists but none has a strict live start and no inbound transition can change
the state, the method fails explicitly. Since current learned checkpoints use
retrieval v1, learned-vs-repaired differences combine scheduling and retrieval
admission changes. The v1 learned interface remains untouched for checkpoint
reproduction.

The separately versioned `deterministic_duration_aware_atomic_v2` scheduler
keeps the urgency-first action interface, strict retrieval-start v2, frozen
REG-v5 selector, and downstream executors. It changes only the positive-slack
choice between accepting the currently observable inbound block and retrieving
the head executable job. Executable jobs retain the deterministic ordering
`(slack, relocation_count, canonical_duration, manifest_index)`; lookahead is
applied only to the first job in that ordering, not to a multi-job rollout.

Let `R` be that job's signed time remaining, `E0` its complete canonical
retrieval ETA from the current state, and

```text
s0    = R - E0
e_now = -s0.
```

AcceptStore-v1 supplies a read-only current-epoch REG preview and nominal full
placement duration `D`. The scheduler ages the existing yard by `D`, inserts
the previewed inbound block, moves the projected agent to the previewed cell,
and replans the same head job. If the projected plan has canonical ETA `E1`
and slack `s1`, then

```text
e_accept = -s1 = D + E1 - R.
```

For configured margin `m`, the decision rule is

```text
retrieve now iff abs(e_now) <= abs(e_accept) + m.
```

Equality therefore favors retrieval, and a positive margin deliberately gives
retrieval limited protection against preview error. An unavailable projected
plan has infinite accept-first cost. Already-due executable jobs still bypass
lookahead and are retrieved; infeasible inbound placement still invokes the
capacity-release retrieval rule; states without inbound work still use the
same event-driven Defer; and blocked due work retains strict explicit failure.

This projection is intentionally not described as exact. The preview observes
the current decision epoch, but AcceptStore-v1 calls REG again after navigating
to and picking up the block. Time, queue features, and consequently the chosen
cell can change. Its nominal storage leg and projected retrieval use the
stored-yard canonical planner; actual macro execution additionally enforces
the live route and strict live-first-leg contracts. The counterfactual can
therefore be optimistic when other live inventory changes route feasibility,
especially under layouts or arrival mechanics unlike the calibrated default.
The audit therefore records preview/executed-cell agreement,
estimated/actual placement duration, preview failures, decision costs, and the
shared arrival-to-completed-storage flow records. A reserved-assignment
AcceptStore would be a new option contract and would require a matching
due-only control before it could support a one-factor claim.

The experimental `relational_residual_macro_scheduler_v1` is a parallel Track B
path rather than an atomic-v5 replacement. It constructs the currently
executable macro set and evaluates each candidate with the same function:

```text
score(s, a) = urgency_baseline(s, a)
              + learned_residual(s, a)
              - uncertainty_penalty * predicted_risk(s, a)
```

A Deep-Set encoder pools the observable block set, a shared candidate encoder
represents `AcceptStore`, strict `RetrieveDeliver(block)`, or Defer, and a short
history summary records recent macro types, durations, returns, and failures.
The residual and risk heads are zero-initialized, so an untrained controller is
exactly the deterministic urgency baseline. A configurable margin records and
limits learned overrides. Training uses variable-candidate Double-DQN SMDP
targets; action identities are inputs rather than dedicated output neurons.

The opt-in `kind_safe_residual_map` deployment policy adds a category-preserving
safety gate without changing the checkpoint or network. It accepts learned
reordering only when the learned and baseline candidates have the same macro
kind. With the current candidate set, this lets the network reorder retrieval
blocks while keeping the baseline decision between `AcceptStore`, retrieval,
and Defer. Rejected cross-kind proposals are audited separately as
`cross_kind_override_rejected`; the unrestricted `residual_map` and exact
`baseline` realizations are unchanged.

Version 1 deliberately freezes REG-v5 inside `AcceptStore`, permitting a clean
scheduler comparison. It does not yet expand storage cells into separate
`Store(block, cell)` candidates or simulate exact post-decision states. Those
are follow-on architectures and must receive new checkpoint identifiers.

Current sensitivity-script overrides are:

| Parameter | Value |
|---|---:|
| batch size | 128 |
| discount `gamma` | 0.99 |
| manager learning rate | 0.00005 |
| worker learning rate | 0.00003 |
| soft-update `tau` | 0.001 |
| update interval | 100 steps |
| gradient clip | 5.0 |

Checkpoints are written every 500 episodes and once at the end. They contain
manager/worker network and optimizer states, epsilon, and global step count.
Learned-selector checkpoints additionally contain selector network/optimizer
state, epsilon, and call count. New checkpoints also store a schema version,
the ordered manager/primitive controller identifiers, and the selector feature
version, return definition, and event-return discount.

## Alternative methods

### Genetic algorithm

`example/genetic_algorithm.py` and `GA/` contain the historical
reward-optimizing assignment search used by the GA-assisted HRL method. That
path stores and retrieves blocks sequentially during fitness evaluation, so it
does not reproduce overlapping-inventory PSLAP obstruction minimization.

`PSLAP/ga_optimizer.py` instead evaluates the known finite schedule with
overlapping inventory and lexicographically minimizes infeasible events,
obstructive moves, and then route length. `PSLAP/ga_policy.py` applies its fixed
assignment through the same online retrieval dispatcher as `dynamic_pslap`.
The implementation is labeled a Park--Seo-inspired reproduction because no
official source repository or complete public specification of every GA
operator setting was located; the chosen operators are recorded in `GAConfig`.

### PSLAP-family baselines

`PSLAP/dynamic_policy.py` and `PSLAP/dynamic_yard.py` implement the repaired
online heuristic. Planning uses a detached snapshot containing the exact walls,
storage cells, and delivery gate from the environment. A relocation preserves
the original storage clock and is recorded as an obstructive move rather than a
second placement.

Storage assignment and retrieval scheduling are separate components. All
current and future online PSLAP-family assignment strategies use the shared
travel-time-aware dispatcher in `PSLAP/retrieval_dispatch.py`. It counts the
agent-to-block path, pickup, block-to-exit path, putdown, and any planned
relocations. Retrieval becomes dispatchable when that complete action horizon
reaches the block's remaining storage time. This keeps the dispatcher fixed
when storage assigners are compared.

`PSLAP/legacy.py` exposes the historical `PSLAPPolicy.py`/`yard_logic.py`
controller unchanged under `legacy_adapted_pslap`. It is retained only for
internal ablation because its forced-obstruction branch is not reliable.

`PSLAP/baselines.py` owns the stable method names. The unqualified `pslap` name
is a compatibility alias for `dynamic_pslap`. The GA reproduction is split by
information regime: `pslap_ga_2009_offline` optimizes the complete episode
schedule before execution, while `pslap_ga_2009_rolling` reoptimizes at each
inbound epoch over only blocks whose arrival time has been reached. The old
`pslap_ga_2009` name aliases the offline reference. `PSLAP/run_pslap.py`
supplies common execution and reports returns, timing, obstructive moves,
assignment fallbacks, illegal drops, and success.

### Neutral Track A assignment layer

`PSLAP/track_a.py` evaluates storage assignment independently of method-specific
retrieval or execution. At each inbound decision it constructs one ordered
physical feasibility mask using storage membership, vacancy, and reachability:

```text
assignment_source(yard, inbound_block, pickup_cell, valid_candidates)
    -> proposed_cell
```

Every source receives the exact same candidate tuple. The proposal is recorded
before validation. A valid cell is executed unchanged. A missing, malformed,
non-storage, occupied, unreachable, or out-of-mask proposal is a strict method
failure. For the separate operational-continuation result, the executor may use
one shared shortest-reachable fallback; the executed fallback cell is stored
separately and all later transitions and deliveries are marked contaminated.
An epoch with no feasible candidate is recorded as downstream system
infeasibility and is not charged as an invalid method proposal.

The downstream stack is fixed across assignment sources: travel-time-aware
retrieval scheduling, shortest-path/lexicographic relocation placement,
deterministic routing, and actual primitive environment transitions. The
existing complete-system policies retain their original relocation behavior;
the neutral rule is confined to Track A.

Track A exposes `reg_selector`, `reg_selector_v4`, `nearest_free`, `dynamic_pslap`,
`pslap_ga_2009_rolling`, and `pslap_ga_2009_offline`. Selector feature version 3
removes hidden future-duration information from unarrived-block features and
ordering. `reg_selector` accepts only a version-3 checkpoint; version-2 and
unverified checkpoints are rejected and must be retrained before entering the
online assignment table.

### REG-v4 Track A-native selector

`PSLAP/reg_selector_v4.py` preserves v3 as an internal flat-network ablation
and defines a separate feature-version 4 architecture. V4 encodes all
observable blocks with a shared block encoder and permutation-invariant mean
and maximum pooling. A second shared encoder scores any number of admissible
candidate cells. Thus block ordering and candidate-set cardinality do not alter
the score of an otherwise identical decision.

V4 receives only the exact Track A candidate tuple. Unarrived blocks expose
neither required duration nor a preassigned cell. Its event begins at the
assignment decision, accumulates the actual neutral-executor environment reward
once per primitive step, and closes without bootstrap at that block's delivery
or episode truncation. Training uses normalized raw event returns, Huber loss,
gradient clipping, replay, and assignment-count-based exploration. Periodic
validation uses fixed seeds disjoint from the generated training-instance seed
namespace; the best checkpoint is selected by strict success and then return.

## Known architectural risks

- Legacy source paths still mix package-qualified and bare module names.
- Legacy controllers that jointly select the two heads retain restricted or
  calibration-sensitive continuation semantics; they are ablations, not v5.
- Experimental DQN implementations also exist in `example/dqn.py` and
  `example/dqn_agent.py`; `options_agent.py` is the current HRL implementation.
- Some code comments describe earlier or proposed behavior rather than active
  behavior.
- Checkpoints produced before schema version 3 cannot verify all controller,
  selector-feature, and selector-return semantics. They require the explicit
  legacy override and must remain labeled non-comparable.
