"""Exact compact-yard recovery under a strict decision-boundary model.

This module is deliberately policy independent.  It does not query a learned
selector, PSLAP rule, urgency rule, or any other behavior policy.  It searches
the physical configuration graph induced by two strict macros:

* direct delivery of one approachable stored block to a real exit; and
* relocation of one approachable stored block to a reachable empty cell.

The certificate has a narrow contract.  Admission is closed, no future arrival
schedule is assumed, the agent carries at most one block, and fixed obstacles
remain fixed for the entire recovery horizon.  It is therefore a certificate
of *current-state strict-macro recoverability*, not full-episode viability.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from heapq import heappop, heappush
from typing import Iterable, Optional

from PSLAP.dynamic_yard import BlockView, Cell, YardSnapshot


STRICT_MACRO_ACTION_MODEL = (
    "strict_decision_boundary_empty_hand_approach_one_carried_block_"
    "direct_delivery_or_single_relocation_v1"
)
CLOSED_ADMISSION_CONTRACT = (
    "closed_admission_no_future_schedule_current_fixed_obstacles_and_"
    "storage_reservations_v1"
)
RECOVERY_SEARCH_ORDERS = ("breadth_first", "goal_directed")


class ViabilityStatus(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"


class RecoveryActionKind(str, Enum):
    DELIVERY = "DELIVERY"
    RELOCATION = "RELOCATION"


@dataclass(frozen=True)
class RecoveryAction:
    """One executable strict macro and its deterministic shortest paths."""

    kind: RecoveryActionKind
    block_label: str
    source: Cell
    destination: Cell
    approach_path: tuple[Cell, ...]
    transport_path: tuple[Cell, ...]

    @property
    def steps(self) -> int:
        """Primitive movement, pickup, and putdown steps for this macro."""

        return (
            len(self.approach_path)
            - 1
            + 1
            + len(self.transport_path)
            - 1
            + 1
        )


@dataclass(frozen=True)
class RecoveryState:
    """Authoritative immutable state for strict-macro recovery analysis.

    ``fixed_obstacles`` are physically impassable for the complete horizon.
    ``reserved_cells`` remain traversable but cannot be relocation targets,
    matching the repository's storage-reservation semantics.
    """

    rows: int
    cols: int
    traversable: frozenset[Cell]
    storage_cells: frozenset[Cell]
    exits: tuple[Cell, ...]
    blocks: tuple[BlockView, ...]
    agent_position: Cell
    fixed_obstacles: frozenset[Cell] = frozenset()
    reserved_cells: frozenset[Cell] = frozenset()
    pickup_cells: frozenset[Cell] = frozenset()
    wait_cells: frozenset[Cell] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "traversable", frozenset(self.traversable))
        object.__setattr__(self, "storage_cells", frozenset(self.storage_cells))
        object.__setattr__(self, "exits", tuple(sorted(set(self.exits))))
        object.__setattr__(
            self,
            "blocks",
            tuple(sorted(self.blocks, key=lambda item: (item.label, item.position))),
        )
        object.__setattr__(self, "fixed_obstacles", frozenset(self.fixed_obstacles))
        object.__setattr__(self, "reserved_cells", frozenset(self.reserved_cells))
        object.__setattr__(self, "pickup_cells", frozenset(self.pickup_cells))
        object.__setattr__(self, "wait_cells", frozenset(self.wait_cells))
        self._validate()

    @classmethod
    def from_yard_snapshot(
        cls,
        yard: YardSnapshot,
        agent_position: Cell,
        *,
        fixed_obstacles: Iterable[Cell] = (),
        reserved_cells: Iterable[Cell] = (),
        pickup_cells: Iterable[Cell] = (),
        wait_cells: Iterable[Cell] = (),
    ) -> "RecoveryState":
        """Adapt a read-only yard snapshot into the complete recovery state."""

        return cls(
            rows=yard.rows,
            cols=yard.cols,
            traversable=yard.traversable,
            storage_cells=yard.storage_cells,
            exits=yard.exits,
            blocks=yard.blocks,
            agent_position=agent_position,
            fixed_obstacles=frozenset(fixed_obstacles),
            reserved_cells=frozenset(reserved_cells),
            pickup_cells=frozenset(pickup_cells),
            wait_cells=frozenset(wait_cells),
        )

    def to_yard_snapshot(self) -> YardSnapshot:
        """Return the geometry/block view expected by existing read-only tools."""

        return YardSnapshot(
            rows=self.rows,
            cols=self.cols,
            traversable=self.traversable,
            storage_cells=self.storage_cells,
            exits=self.exits,
            blocks=self.blocks,
        )

    def occupancy(self) -> dict[Cell, BlockView]:
        return {block.position: block for block in self.blocks}

    def block(self, label: str) -> Optional[BlockView]:
        return next((item for item in self.blocks if item.label == label), None)

    def _validate(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("recovery geometry dimensions must be positive")
        if self.agent_position not in self.traversable:
            raise ValueError("agent_position must be traversable")
        if self.agent_position in self.fixed_obstacles:
            raise ValueError("agent_position cannot be a fixed obstacle")
        if not self.storage_cells <= self.traversable:
            raise ValueError("every storage cell must be traversable")
        if not frozenset(self.exits) <= self.traversable:
            raise ValueError("every exit must be traversable")
        if not self.fixed_obstacles <= self.traversable:
            raise ValueError("every fixed obstacle must be traversable")
        if not self.reserved_cells <= self.storage_cells:
            raise ValueError("every reserved cell must be a storage cell")
        if not self.pickup_cells <= self.traversable:
            raise ValueError("every pickup cell must be traversable")
        if not self.wait_cells <= self.traversable:
            raise ValueError("every wait cell must be traversable")
        if self.fixed_obstacles.intersection(self.reserved_cells):
            raise ValueError("fixed obstacles and reservations must be disjoint")

        labels = [block.label for block in self.blocks]
        positions = [block.position for block in self.blocks]
        if len(labels) != len(set(labels)):
            raise ValueError("stored block labels must be unique")
        if len(positions) != len(set(positions)):
            raise ValueError("stored block positions must be unique")
        if any(position not in self.storage_cells for position in positions):
            raise ValueError("every stored block must occupy a storage cell")
        if self.fixed_obstacles.intersection(positions):
            raise ValueError("fixed obstacles cannot overlap stored blocks")
        if self.reserved_cells.intersection(positions):
            raise ValueError("reserved cells cannot overlap stored blocks")


@dataclass(frozen=True)
class RecoverabilityCertificate:
    """Auditable result for the stated strict-macro recovery contract."""

    status: ViabilityStatus
    witness: tuple[RecoveryAction, ...]
    witness_primitive_steps: Optional[int]
    explored_nodes: int
    generated_states: int
    max_depth_reached: int
    max_primitive_steps_reached: int
    frontier_states: int
    exhaustive: bool
    reason: str
    fixed_obstacles: frozenset[Cell]
    reserved_cells: frozenset[Cell]
    primitive_step_budget: Optional[int]
    search_order: str
    contract: str = CLOSED_ADMISSION_CONTRACT
    action_model: str = STRICT_MACRO_ACTION_MODEL

    @property
    def is_safe(self) -> bool:
        return self.status is ViabilityStatus.SAFE

    @property
    def is_unsafe(self) -> bool:
        return self.status is ViabilityStatus.UNSAFE

    @property
    def is_unknown(self) -> bool:
        return self.status is ViabilityStatus.UNKNOWN

    @property
    def witness_macro_count(self) -> Optional[int]:
        """Number of macros in a valid witness, when one was found."""

        return len(self.witness) if self.is_safe else None

    @property
    def recovery_rank_is_exact(self) -> bool:
        """Whether witness length equals the minimum macro recovery rank.

        Breadth-first search expands the unit-cost macro graph by depth, so
        its first complete witness realizes ``rho``.  Goal-directed search is
        a faster feasibility search and makes no shortest-witness claim.
        """

        return self.is_safe and (
            not self.witness or self.search_order == "breadth_first"
        )

    @property
    def exact_recovery_rank(self) -> Optional[int]:
        return (
            self.witness_macro_count
            if self.recovery_rank_is_exact
            else None
        )


ConfigurationKey = tuple[
    tuple[Cell, ...],
    tuple[Cell, ...],
    Cell,
    tuple[tuple[str, Cell], ...],
]
NodeKey = tuple[ConfigurationKey, int, int]


def _neighbors(cell: Cell) -> tuple[Cell, ...]:
    row, col = cell
    return tuple(
        sorted(
            (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            )
        )
    )


def _shortest_clear_path(
    state: RecoveryState,
    start: Cell,
    goal: Cell,
    *,
    ignore_label: Optional[str] = None,
) -> Optional[tuple[Cell, ...]]:
    if (
        start not in state.traversable
        or goal not in state.traversable
        or start in state.fixed_obstacles
        or goal in state.fixed_obstacles
    ):
        return None

    occupied = {
        cell
        for cell, block in state.occupancy().items()
        if block.label != ignore_label and cell != start
    }
    blocked = occupied | state.fixed_obstacles
    queue = deque([start])
    parent: dict[Cell, Optional[Cell]] = {start: None}

    while queue:
        cell = queue.popleft()
        if cell == goal:
            path = []
            cursor: Optional[Cell] = goal
            while cursor is not None:
                path.append(cursor)
                cursor = parent[cursor]
            return tuple(reversed(path))
        for neighbor in _neighbors(cell):
            if (
                neighbor in state.traversable
                and neighbor not in blocked
                and neighbor not in parent
            ):
                parent[neighbor] = cell
                queue.append(neighbor)
    return None


def _action_sort_key(action: RecoveryAction):
    return (
        0 if action.kind is RecoveryActionKind.DELIVERY else 1,
        action.block_label,
        action.destination,
        action.approach_path,
        action.transport_path,
    )


def legal_recovery_actions(state: RecoveryState) -> tuple[RecoveryAction, ...]:
    """Enumerate all executable strict macros at this decision boundary."""

    if not isinstance(state, RecoveryState):
        raise TypeError("legal_recovery_actions requires a RecoveryState")
    occupied = frozenset(state.occupancy())
    empty_storage = tuple(
        sorted(
            state.storage_cells
            - occupied
            - state.fixed_obstacles
            - state.reserved_cells
        )
    )
    actions = []

    for block in state.blocks:
        # Empty-hand approach treats every stored block as an obstacle except
        # the selected pickup target.  The agent may share the start cell with
        # a block immediately after a previous PUTDOWN and can move away.
        approach = _shortest_clear_path(
            state,
            state.agent_position,
            block.position,
            ignore_label=block.label,
        )
        if approach is None:
            continue

        # Pickup removes the selected block from occupancy while it is carried.
        for exit_cell in state.exits:
            transport = _shortest_clear_path(
                state,
                block.position,
                exit_cell,
                ignore_label=block.label,
            )
            if transport is not None:
                actions.append(
                    RecoveryAction(
                        RecoveryActionKind.DELIVERY,
                        block.label,
                        block.position,
                        exit_cell,
                        approach,
                        transport,
                    )
                )

        for destination in empty_storage:
            transport = _shortest_clear_path(
                state,
                block.position,
                destination,
                ignore_label=block.label,
            )
            if transport is not None:
                actions.append(
                    RecoveryAction(
                        RecoveryActionKind.RELOCATION,
                        block.label,
                        block.position,
                        destination,
                        approach,
                        transport,
                    )
                )

    return tuple(sorted(actions, key=_action_sort_key))


def _apply_legal_action(
    state: RecoveryState,
    action: RecoveryAction,
) -> RecoveryState:
    block = state.block(action.block_label)
    if block is None:
        raise ValueError(f"unknown stored block: {action.block_label}")
    if action.kind is RecoveryActionKind.DELIVERY:
        blocks = tuple(item for item in state.blocks if item.label != block.label)
    elif action.kind is RecoveryActionKind.RELOCATION:
        blocks = tuple(
            replace(item, position=action.destination)
            if item.label == block.label
            else item
            for item in state.blocks
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown recovery action kind: {action.kind}")
    return replace(state, blocks=blocks, agent_position=action.destination)


def apply_recovery_action(
    state: RecoveryState,
    action: RecoveryAction,
) -> RecoveryState:
    """Revalidate and apply one strict recovery macro."""

    if action not in legal_recovery_actions(state):
        raise ValueError("recovery action is not legal in the supplied state")
    return _apply_legal_action(state, action)


def _configuration_key(state: RecoveryState) -> ConfigurationKey:
    return (
        tuple(sorted(state.fixed_obstacles)),
        tuple(sorted(state.reserved_cells)),
        state.agent_position,
        tuple(sorted((block.label, block.position) for block in state.blocks)),
    )


def _node_key(
    state: RecoveryState,
    depth: int,
    primitive_steps: int,
    *,
    resource_sensitive: bool,
) -> NodeKey:
    # Without a primitive horizon, the first BFS visit has minimum macro depth
    # and dominates every later visit to the same physical configuration.
    return (
        _configuration_key(state),
        depth if resource_sensitive else 0,
        primitive_steps if resource_sensitive else 0,
    )


def _reconstruct_witness(
    terminal_key: NodeKey,
    parents: dict[NodeKey, tuple[Optional[NodeKey], Optional[RecoveryAction]]],
) -> tuple[RecoveryAction, ...]:
    actions = []
    cursor = terminal_key
    while True:
        parent, action = parents[cursor]
        if parent is None:
            return tuple(reversed(actions))
        if action is None:  # pragma: no cover
            raise RuntimeError("missing action in recovery witness")
        actions.append(action)
        cursor = parent


def _certificate(
    state: RecoveryState,
    *,
    status: ViabilityStatus,
    witness: tuple[RecoveryAction, ...] = (),
    explored_nodes: int,
    generated_states: int,
    max_depth_reached: int,
    max_primitive_steps_reached: int,
    frontier_states: int,
    exhaustive: bool,
    reason: str,
    primitive_step_budget: Optional[int],
    search_order: str,
) -> RecoverabilityCertificate:
    return RecoverabilityCertificate(
        status=status,
        witness=witness,
        witness_primitive_steps=(
            sum(action.steps for action in witness) if status is ViabilityStatus.SAFE else None
        ),
        explored_nodes=explored_nodes,
        generated_states=generated_states,
        max_depth_reached=max_depth_reached,
        max_primitive_steps_reached=max_primitive_steps_reached,
        frontier_states=frontier_states,
        exhaustive=exhaustive,
        reason=reason,
        fixed_obstacles=state.fixed_obstacles,
        reserved_cells=state.reserved_cells,
        primitive_step_budget=primitive_step_budget,
        search_order=search_order,
    )


def analyze_recoverability(
    state: RecoveryState,
    *,
    max_depth: Optional[int] = None,
    max_nodes: Optional[int] = 100_000,
    max_primitive_steps: Optional[int] = None,
    search_order: str = "breadth_first",
) -> RecoverabilityCertificate:
    """Return a ``SAFE``, ``UNSAFE``, or ``UNKNOWN`` recovery certificate.

    ``max_nodes`` and ``max_depth`` bound proof computation; hitting either
    returns ``UNKNOWN``.  ``max_primitive_steps`` instead models a real
    remaining execution horizon.  Exhausting all actions within that horizon
    proves ``UNSAFE`` *within the supplied primitive-step budget*.
    """

    if not isinstance(state, RecoveryState):
        raise TypeError("analyze_recoverability requires a RecoveryState")
    if search_order not in RECOVERY_SEARCH_ORDERS:
        raise ValueError(
            f"search_order must be one of {RECOVERY_SEARCH_ORDERS!r}"
        )
    for name, value, allow_zero in (
        ("max_depth", max_depth, True),
        ("max_nodes", max_nodes, False),
        ("max_primitive_steps", max_primitive_steps, True),
    ):
        if value is None:
            continue
        minimum = 0 if allow_zero else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer or None")

    # A non-BFS order may discover the same physical configuration first by a
    # longer route.  Keep resource consumption in the visited key whenever a
    # real depth/step bound is active so that such a discovery cannot suppress
    # a later feasible route within the supplied budget.
    resource_sensitive = (
        max_depth is not None or max_primitive_steps is not None
    )
    initial_key = _node_key(
        state, 0, 0, resource_sensitive=resource_sensitive
    )
    parents = {initial_key: (None, None)}
    states = {initial_key: (state, 0, 0)}

    if not state.blocks:
        return _certificate(
            state,
            status=ViabilityStatus.SAFE,
            explored_nodes=1,
            generated_states=1,
            max_depth_reached=0,
            max_primitive_steps_reached=0,
            frontier_states=0,
            exhaustive=True,
            reason=(
                "safe:closed_admission_strict_macro_recovery;"
                "no_future_schedule;current_fixed_obstacles_and_"
                "storage_reservations;empty_yard"
            ),
            primitive_step_budget=max_primitive_steps,
            search_order=search_order,
        )

    serial = 0
    if search_order == "breadth_first":
        frontier = deque([initial_key])
    else:
        frontier = [(len(state.blocks), 0, 0, serial, initial_key)]

    def pop_frontier():
        if search_order == "breadth_first":
            return frontier.popleft()
        return heappop(frontier)[-1]

    def push_frontier(key, successor, depth, primitive_steps):
        nonlocal serial
        if search_order == "breadth_first":
            frontier.append(key)
            return
        serial += 1
        heappush(
            frontier,
            (
                len(successor.blocks),
                depth,
                primitive_steps,
                serial,
                key,
            ),
        )
    explored_nodes = 0
    max_depth_reached = 0
    max_steps_reached = 0
    depth_cutoffs: set[NodeKey] = set()

    while frontier:
        if max_nodes is not None and explored_nodes >= max_nodes:
            return _certificate(
                state,
                status=ViabilityStatus.UNKNOWN,
                explored_nodes=explored_nodes,
                generated_states=len(states),
                max_depth_reached=max_depth_reached,
                max_primitive_steps_reached=max_steps_reached,
                frontier_states=len(frontier),
                exhaustive=False,
                reason=(
                    "unknown:closed_admission_strict_macro_recovery;"
                    "no_future_schedule;current_fixed_obstacles_and_"
                    "storage_reservations;"
                    "node_budget_exhausted"
                ),
                primitive_step_budget=max_primitive_steps,
                search_order=search_order,
            )

        key = pop_frontier()
        current, depth, used_steps = states[key]
        explored_nodes += 1
        max_depth_reached = max(max_depth_reached, depth)
        max_steps_reached = max(max_steps_reached, used_steps)
        actions = legal_recovery_actions(current)

        if max_depth is not None and depth >= max_depth:
            for action in actions:
                next_steps = used_steps + action.steps
                if (
                    max_primitive_steps is not None
                    and next_steps > max_primitive_steps
                ):
                    continue
                successor = _apply_legal_action(current, action)
                successor_key = _node_key(
                    successor,
                    depth + 1,
                    next_steps,
                    resource_sensitive=resource_sensitive,
                )
                if successor_key not in states:
                    depth_cutoffs.add(successor_key)
            continue

        for action in actions:
            next_steps = used_steps + action.steps
            if (
                max_primitive_steps is not None
                and next_steps > max_primitive_steps
            ):
                continue
            successor = _apply_legal_action(current, action)
            successor_depth = depth + 1
            successor_key = _node_key(
                successor,
                successor_depth,
                next_steps,
                resource_sensitive=resource_sensitive,
            )
            if successor_key in states:
                continue
            states[successor_key] = (successor, successor_depth, next_steps)
            parents[successor_key] = (key, action)
            max_depth_reached = max(max_depth_reached, successor_depth)
            max_steps_reached = max(max_steps_reached, next_steps)

            if not successor.blocks:
                witness = _reconstruct_witness(successor_key, parents)
                return _certificate(
                    state,
                    status=ViabilityStatus.SAFE,
                    witness=witness,
                    explored_nodes=explored_nodes,
                    generated_states=len(states),
                    max_depth_reached=max_depth_reached,
                    max_primitive_steps_reached=max_steps_reached,
                    frontier_states=len(frontier),
                    exhaustive=False,
                    reason=(
                        "safe:closed_admission_strict_macro_recovery;"
                        "no_future_schedule;current_fixed_obstacles_and_"
                        "storage_reservations;"
                        "deterministic_witness_found"
                    ),
                    primitive_step_budget=max_primitive_steps,
                    search_order=search_order,
                )
            push_frontier(
                successor_key,
                successor,
                successor_depth,
                next_steps,
            )

    if depth_cutoffs:
        return _certificate(
            state,
            status=ViabilityStatus.UNKNOWN,
            explored_nodes=explored_nodes,
            generated_states=len(states),
            max_depth_reached=max_depth_reached,
            max_primitive_steps_reached=max_steps_reached,
            frontier_states=len(depth_cutoffs),
            exhaustive=False,
            reason=(
                "unknown:closed_admission_strict_macro_recovery;"
                "no_future_schedule;current_fixed_obstacles_and_"
                "storage_reservations;"
                "depth_budget_exhausted"
            ),
            primitive_step_budget=max_primitive_steps,
            search_order=search_order,
        )

    budget_suffix = (
        ";within_primitive_step_budget"
        if max_primitive_steps is not None
        else ""
    )
    return _certificate(
        state,
        status=ViabilityStatus.UNSAFE,
        explored_nodes=explored_nodes,
        generated_states=len(states),
        max_depth_reached=max_depth_reached,
        max_primitive_steps_reached=max_steps_reached,
        frontier_states=0,
        exhaustive=True,
        reason=(
            "unsafe:closed_admission_strict_macro_recovery;"
            "no_future_schedule;current_fixed_obstacles_and_"
            "storage_reservations;"
            "reachable_state_space_exhausted"
            + budget_suffix
        ),
        primitive_step_budget=max_primitive_steps,
        search_order=search_order,
    )


__all__ = [
    "CLOSED_ADMISSION_CONTRACT",
    "STRICT_MACRO_ACTION_MODEL",
    "RECOVERY_SEARCH_ORDERS",
    "RecoverabilityCertificate",
    "RecoveryAction",
    "RecoveryActionKind",
    "RecoveryState",
    "ViabilityStatus",
    "analyze_recoverability",
    "apply_recovery_action",
    "legal_recovery_actions",
]
