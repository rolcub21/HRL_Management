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

PSLaP bypasses the learned manager/worker decision process and operates through
its heuristic policy and yard-planning logic. The GA variant still trains the
manager and worker networks, but its storage selector reads a fixed assignment.

## Core abstractions

### Environment

`environment.py` defines `BaseEnvironment`, the abstract interface used by the
option framework. `example/small_rooms_env.py` supplies the actual shipyard
environment as `SmallRoomsEnv`.

The default yard is a 10 by 10 grid with walls around the boundary, a pickup
position near the entrance, a seven-cell delivery gate, and all eligible free
cells treated as storage positions. There are 40 blocks per episode.

At reset:

- each block receives a positive Poisson-distributed required storage duration
  with mean `proc_mean` (`mu`);
- block inter-arrival gaps are exponentially distributed with mean
  `1 / arrival_rate` (`1 / lambda`);
- the first block is available immediately; and
- storage locations begin unassigned unless `choose_storage=True`.

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

The agent adds another `-0.5` penalty while executing `WaitOption`. Several old
reward alternatives and an unused potential-based shaping calculation remain
commented in the source; they are not part of current behavior.

Primary logged metrics are total episode return, mean delivery-time error,
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
| `waitOption.py` | wait when no useful operation is available |
| `selector.py` | learned storage-cell selector |
| `GAStorageSelectOption.py` | selector backed by a saved GA assignment |
| `random_selector.py` | random/experimental selector variant |

The mixed filename capitalization is historical and matters on Linux.

## Learning agent

`options_agent.py` contains the current `DQNAgent` and two Q-networks:

- the manager network scores non-primitive options;
- the worker network scores primitive options.

The agent uses epsilon-greedy selection, a prioritized manager replay buffer, a
worker replay buffer, target networks, soft target updates, gradient clipping,
and separate Adam optimizers. Default values exist in `DQNAgent`, while the
training entry points override several of them.

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
state, epsilon, and call count.

## Alternative methods

### Genetic algorithm

`example/genetic_algorithm.py` contains an assignment-search implementation.
`GA/` contains supporting yard and sub-algorithm logic. The evaluated GA method
loads the resulting assignment with `GAStorageSelectOption`; it is therefore a
GA-assisted HRL method rather than a fully independent controller.

### PSLaP

`PSLAP/PSLAPPolicy.py` and `PSLAP/yard_logic.py` implement the heuristic policy
and yard model. `PSLAP/run_pslap.py` exposes the experiment function used by
both the sensitivity runner and common evaluator.

## Known architectural risks

- Source imports mix package-qualified and bare module names.
- Option ordering partly relies on sets; manager options are sorted in the
  constructor but refreshed unsorted during training.
- Experimental DQN implementations also exist in `example/dqn.py` and
  `example/dqn_agent.py`; `options_agent.py` is the current HRL implementation.
- Some code comments describe earlier or proposed behavior rather than active
  behavior.
- Checkpoint compatibility is not versioned.
