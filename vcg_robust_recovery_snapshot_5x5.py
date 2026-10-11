"""On-demand robust recovery certificates for frozen 5x5 snapshots.

The solver consumes only an immutable :class:`RecoveryState` and a primitive
completion budget.  Admission is closed and no future EpisodeInstance schedule
is available to the query.  Every strict recovery macro is realized under the
finite product envelope

    extra duration in {0, 1}
      x
    nominal stop or a clear one-cell post-macro stop.

The returned recursive object is a bounded robust-completion winning set, not
the full dynamic-episode viability kernel.  Count-budget exhaustion is
``UNKNOWN`` and therefore fail-closed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from typing import Iterable, Mapping, Optional, Sequence

from PSLAP.dynamic_yard import BlockView, Cell
from PSLAP.viability import (
    RecoveryAction,
    RecoveryActionKind,
    RecoveryState,
    apply_recovery_action,
    legal_recovery_actions,
)


PROTOCOL = "vcg_5x5_frozen_snapshot_robust_recovery_v1"
UNCERTAINTY_CONTRACT = (
    "extra_duration_0_or_1_times_nominal_or_clear_radius_one_post_macro_stop_v1"
)


class SolveStatus(str, Enum):
    WINNING = "WINNING"
    LOSING = "LOSING"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DisturbanceContract:
    delays: tuple[int, ...] = (0, 1)
    include_clear_adjacent_stops: bool = True

    def __post_init__(self) -> None:
        delays = tuple(self.delays)
        if (
            not delays
            or 0 not in delays
            or len(delays) != len(set(delays))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in delays
            )
        ):
            raise ValueError("delays must be unique non-negative integers including zero")
        if not isinstance(self.include_clear_adjacent_stops, bool):
            raise TypeError("include_clear_adjacent_stops must be bool")
        object.__setattr__(self, "delays", tuple(sorted(delays)))


NOMINAL_CONTRACT = DisturbanceContract(
    delays=(0,), include_clear_adjacent_stops=False
)
ROBUST_CONTRACT = DisturbanceContract()


@dataclass(frozen=True)
class ExecutionOutcome:
    disturbance_id: str
    delay_steps: int
    stop_cell: Cell
    stop_extra_steps: int
    primitive_steps: int
    successor: Optional[RecoveryState]
    horizon_safe: bool
    tube_safe: bool
    nominal: bool

    def __post_init__(self) -> None:
        if not self.disturbance_id:
            raise ValueError("disturbance_id must be nonempty")
        for name, value in (
            ("delay_steps", self.delay_steps),
            ("stop_extra_steps", self.stop_extra_steps),
            ("primitive_steps", self.primitive_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"{name} must be a non-negative integer")
        if not isinstance(self.horizon_safe, bool) or not isinstance(
            self.tube_safe, bool
        ):
            raise TypeError("outcome safety flags must be bool")
        if self.horizon_safe != (self.successor is not None):
            raise ValueError("horizon_safe must agree with successor presence")
        if not isinstance(self.nominal, bool):
            raise TypeError("nominal must be bool")


@dataclass(frozen=True)
class ActionCertificate:
    action_key: str
    status: SolveStatus
    outcome_count: int
    outcome_status_counts: tuple[tuple[str, int], ...]
    witness_macro_depth_bound: Optional[int]
    physical_rehandle: int

    @property
    def admitted(self) -> bool:
        return self.status is SolveStatus.WINNING


@dataclass(frozen=True)
class StateCertificate:
    status: SolveStatus
    witness_action_key: Optional[str]
    witness_macro_depth_bound: Optional[int]
    reason: str

    @property
    def admitted(self) -> bool:
        return self.status is SolveStatus.WINNING


@dataclass(frozen=True)
class MethodCertificate:
    method: str
    state: StateCertificate
    actions: tuple[ActionCertificate, ...]
    expanded_nodes: int
    generated_outcomes: int
    cache_hits: int
    cache_entries: int
    cutoff_count: int
    max_expansions: int

    @property
    def admitted_action_keys(self) -> tuple[str, ...]:
        return tuple(action.action_key for action in self.actions if action.admitted)


@dataclass(frozen=True)
class SnapshotCertificate:
    primitive_budget: int
    nominal: MethodCertificate
    one_step: MethodCertificate
    recursive: MethodCertificate
    semantic_digest: str


def recovery_action_key(action: RecoveryAction) -> str:
    prefix = (
        "deliver"
        if action.kind is RecoveryActionKind.DELIVERY
        else "reconfigure"
    )
    return (
        f"{prefix}:{action.block_label}:"
        f"{action.destination[0]}:{action.destination[1]}"
    )


def _action_sort_key(action: RecoveryAction) -> tuple:
    return (
        0 if action.kind is RecoveryActionKind.DELIVERY else 1,
        recovery_action_key(action),
        int(action.steps),
        action.approach_path,
        action.transport_path,
    )


def canonical_actions(state: RecoveryState) -> tuple[RecoveryAction, ...]:
    return tuple(sorted(legal_recovery_actions(state), key=_action_sort_key))


def _neighbors(cell: Cell) -> tuple[Cell, ...]:
    row, column = cell
    return (
        (row - 1, column),
        (row, column - 1),
        (row, column + 1),
        (row + 1, column),
    )


def clear_stop_cells(
    successor: RecoveryState,
    nominal_stop: Cell,
    *,
    include_adjacent: bool,
) -> tuple[Cell, ...]:
    cells = {tuple(nominal_stop)}
    if include_adjacent:
        occupied = frozenset(successor.occupancy())
        excluded = (
            successor.fixed_obstacles
            | successor.reserved_cells
            | successor.pickup_cells
            | successor.wait_cells
            | occupied
        )
        cells.update(
            neighbor
            for neighbor in _neighbors(tuple(nominal_stop))
            if neighbor in successor.traversable and neighbor not in excluded
        )
    return tuple(sorted(cells))


def execution_envelope(
    state: RecoveryState,
    action: RecoveryAction,
    *,
    remaining_primitive_steps: int,
    contract: DisturbanceContract = ROBUST_CONTRACT,
) -> tuple[ExecutionOutcome, ...]:
    """Enumerate the complete declared product envelope for one strict macro."""

    if (
        isinstance(remaining_primitive_steps, bool)
        or not isinstance(remaining_primitive_steps, int)
        or remaining_primitive_steps < 0
    ):
        raise ValueError("remaining_primitive_steps must be non-negative")
    if action not in legal_recovery_actions(state):
        raise ValueError("execution envelope requires a legal recovery action")
    nominal_successor = apply_recovery_action(state, action)
    stops = clear_stop_cells(
        nominal_successor,
        action.destination,
        include_adjacent=contract.include_clear_adjacent_stops,
    )
    outcomes = []
    for delay in contract.delays:
        for stop in stops:
            stop_extra = int(tuple(stop) != tuple(action.destination))
            cost = int(action.steps) + int(delay) + stop_extra
            horizon_safe = cost <= remaining_primitive_steps
            successor = (
                replace(nominal_successor, agent_position=tuple(stop))
                if horizon_safe
                else None
            )
            nominal = delay == 0 and tuple(stop) == tuple(action.destination)
            outcomes.append(
                ExecutionOutcome(
                    disturbance_id=(
                        f"delay:{delay}|stop:{stop[0]}:{stop[1]}"
                    ),
                    delay_steps=int(delay),
                    stop_cell=tuple(stop),
                    stop_extra_steps=stop_extra,
                    primitive_steps=cost,
                    successor=successor,
                    horizon_safe=horizon_safe,
                    tube_safe=True,
                    nominal=nominal,
                )
            )
    identifiers = [outcome.disturbance_id for outcome in outcomes]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("execution envelope produced duplicate outcomes")
    if sum(outcome.nominal for outcome in outcomes) != 1:
        raise RuntimeError("execution envelope must contain exactly one nominal outcome")
    return tuple(outcomes)


class OnDemandRecoverySolver:
    """Count-bounded AND/OR search over a strictly decreasing step budget."""

    def __init__(
        self,
        *,
        contract: DisturbanceContract,
        max_expansions: int,
    ) -> None:
        if (
            isinstance(max_expansions, bool)
            or not isinstance(max_expansions, int)
            or max_expansions < 1
        ):
            raise ValueError("max_expansions must be a positive integer")
        self.contract = contract
        self.max_expansions = max_expansions
        self.expanded_nodes = 0
        self.generated_outcomes = 0
        self.cache_hits = 0
        self.cutoff_count = 0
        self._cache: dict[tuple[RecoveryState, int], StateCertificate] = {}

    @property
    def cache_entries(self) -> int:
        return len(self._cache)

    def _unknown(self, reason: str) -> StateCertificate:
        return StateCertificate(SolveStatus.UNKNOWN, None, None, reason)

    def solve(
        self,
        state: RecoveryState,
        remaining_primitive_steps: int,
    ) -> StateCertificate:
        if not isinstance(state, RecoveryState):
            raise TypeError("robust recovery search requires a RecoveryState")
        if (
            isinstance(remaining_primitive_steps, bool)
            or not isinstance(remaining_primitive_steps, int)
            or remaining_primitive_steps < 0
        ):
            raise ValueError("remaining_primitive_steps must be non-negative")
        key = (state, int(remaining_primitive_steps))
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if not state.blocks:
            result = StateCertificate(
                SolveStatus.WINNING,
                None,
                0,
                "complete_closed_admission_workload",
            )
            self._cache[key] = result
            return result
        if self.expanded_nodes >= self.max_expansions:
            self.cutoff_count += 1
            return self._unknown("deterministic_expansion_budget_exhausted")
        self.expanded_nodes += 1
        actions = canonical_actions(state)
        if not actions:
            result = StateCertificate(
                SolveStatus.LOSING,
                None,
                None,
                "no_legal_strict_recovery_action",
            )
            self._cache[key] = result
            return result

        saw_unknown = False
        for action in actions:
            evaluated = self.evaluate_action(
                state,
                action,
                remaining_primitive_steps,
            )
            if evaluated.status is SolveStatus.WINNING:
                result = StateCertificate(
                    SolveStatus.WINNING,
                    evaluated.action_key,
                    evaluated.witness_macro_depth_bound,
                    "robust_witness_found",
                )
                self._cache[key] = result
                return result
            saw_unknown = saw_unknown or evaluated.status is SolveStatus.UNKNOWN
        if saw_unknown:
            return self._unknown("at_least_one_action_unresolved")
        result = StateCertificate(
            SolveStatus.LOSING,
            None,
            None,
            "all_legal_actions_have_a_losing_realization",
        )
        self._cache[key] = result
        return result

    def evaluate_action(
        self,
        state: RecoveryState,
        action: RecoveryAction,
        remaining_primitive_steps: int,
    ) -> ActionCertificate:
        outcomes = execution_envelope(
            state,
            action,
            remaining_primitive_steps=remaining_primitive_steps,
            contract=self.contract,
        )
        self.generated_outcomes += len(outcomes)
        statuses = []
        depths = []
        for outcome in outcomes:
            if not outcome.tube_safe or not outcome.horizon_safe:
                statuses.append(SolveStatus.LOSING)
                continue
            if outcome.successor is None:  # defensive agreement with flag
                statuses.append(SolveStatus.UNKNOWN)
                continue
            child = self.solve(
                outcome.successor,
                remaining_primitive_steps - outcome.primitive_steps,
            )
            statuses.append(child.status)
            if child.witness_macro_depth_bound is not None:
                depths.append(child.witness_macro_depth_bound)
        if any(status is SolveStatus.LOSING for status in statuses):
            status = SolveStatus.LOSING
            depth = None
        elif any(status is SolveStatus.UNKNOWN for status in statuses):
            status = SolveStatus.UNKNOWN
            depth = None
        else:
            status = SolveStatus.WINNING
            depth = 1 + max(depths, default=0)
        counts = Counter(status.value for status in statuses)
        return ActionCertificate(
            action_key=recovery_action_key(action),
            status=status,
            outcome_count=len(outcomes),
            outcome_status_counts=tuple(sorted(counts.items())),
            witness_macro_depth_bound=depth,
            physical_rehandle=int(
                action.kind is RecoveryActionKind.RELOCATION
            ),
        )


def _state_from_actions(
    actions: Sequence[ActionCertificate],
    *,
    winning_reason: str,
    losing_reason: str,
    unknown_reason: str,
) -> StateCertificate:
    winning = [action for action in actions if action.status is SolveStatus.WINNING]
    if winning:
        chosen = min(winning, key=lambda action: action.action_key)
        return StateCertificate(
            SolveStatus.WINNING,
            chosen.action_key,
            chosen.witness_macro_depth_bound,
            winning_reason,
        )
    if any(action.status is SolveStatus.UNKNOWN for action in actions):
        return StateCertificate(SolveStatus.UNKNOWN, None, None, unknown_reason)
    return StateCertificate(SolveStatus.LOSING, None, None, losing_reason)


def _method_certificate(
    method: str,
    state: StateCertificate,
    actions: Sequence[ActionCertificate],
    solver: OnDemandRecoverySolver,
) -> MethodCertificate:
    return MethodCertificate(
        method=method,
        state=state,
        actions=tuple(actions),
        expanded_nodes=solver.expanded_nodes,
        generated_outcomes=solver.generated_outcomes,
        cache_hits=solver.cache_hits,
        cache_entries=solver.cache_entries,
        cutoff_count=solver.cutoff_count,
        max_expansions=solver.max_expansions,
    )


def certify_snapshot(
    state: RecoveryState,
    *,
    primitive_budget: int,
    max_expansions: int = 50_000,
    robust_contract: DisturbanceContract = ROBUST_CONTRACT,
) -> SnapshotCertificate:
    if not isinstance(state, RecoveryState):
        raise TypeError("snapshot certification requires a RecoveryState")
    if (
        isinstance(primitive_budget, bool)
        or not isinstance(primitive_budget, int)
        or primitive_budget < 0
    ):
        raise ValueError("primitive_budget must be a non-negative integer")
    root_actions = canonical_actions(state)

    nominal_solver = OnDemandRecoverySolver(
        contract=NOMINAL_CONTRACT,
        max_expansions=max_expansions,
    )
    nominal_actions = tuple(
        nominal_solver.evaluate_action(state, action, primitive_budget)
        for action in root_actions
    )
    nominal_state = _state_from_actions(
        nominal_actions,
        winning_reason="at_least_one_nominal_completion_action",
        losing_reason="all_nominal_actions_lose_within_budget",
        unknown_reason="nominal_completion_unresolved",
    )
    nominal = _method_certificate(
        "nominal", nominal_state, nominal_actions, nominal_solver
    )

    one_reference = OnDemandRecoverySolver(
        contract=NOMINAL_CONTRACT,
        max_expansions=max_expansions,
    )
    one_actions = []
    for action in root_actions:
        robust_outcomes = execution_envelope(
            state,
            action,
            remaining_primitive_steps=primitive_budget,
            contract=robust_contract,
        )
        one_reference.generated_outcomes += len(robust_outcomes)
        statuses = []
        depths = []
        for outcome in robust_outcomes:
            if not outcome.tube_safe or not outcome.horizon_safe:
                statuses.append(SolveStatus.LOSING)
                continue
            if outcome.successor is None:
                statuses.append(SolveStatus.UNKNOWN)
                continue
            child = one_reference.solve(
                outcome.successor,
                primitive_budget - outcome.primitive_steps,
            )
            statuses.append(child.status)
            if child.witness_macro_depth_bound is not None:
                depths.append(child.witness_macro_depth_bound)
        if any(item is SolveStatus.LOSING for item in statuses):
            action_status = SolveStatus.LOSING
            action_depth = None
        elif any(item is SolveStatus.UNKNOWN for item in statuses):
            action_status = SolveStatus.UNKNOWN
            action_depth = None
        else:
            action_status = SolveStatus.WINNING
            action_depth = 1 + max(depths, default=0)
        counts = Counter(item.value for item in statuses)
        one_actions.append(
            ActionCertificate(
                action_key=recovery_action_key(action),
                status=action_status,
                outcome_count=len(robust_outcomes),
                outcome_status_counts=tuple(sorted(counts.items())),
                witness_macro_depth_bound=action_depth,
                physical_rehandle=int(
                    action.kind is RecoveryActionKind.RELOCATION
                ),
            )
        )
    one_state = _state_from_actions(
        one_actions,
        winning_reason="all_immediate_realizations_nominally_recoverable",
        losing_reason="every_action_has_a_non_nominally_recoverable_realization",
        unknown_reason="one_step_containment_unresolved",
    )
    one_step = _method_certificate(
        "one_step", one_state, one_actions, one_reference
    )

    recursive_solver = OnDemandRecoverySolver(
        contract=robust_contract,
        max_expansions=max_expansions,
    )
    recursive_actions = tuple(
        recursive_solver.evaluate_action(state, action, primitive_budget)
        for action in root_actions
    )
    recursive_state = _state_from_actions(
        recursive_actions,
        winning_reason="at_least_one_recursively_robust_completion_action",
        losing_reason="all_actions_leave_the_recursive_completion_set",
        unknown_reason="recursive_completion_unresolved",
    )
    recursive = _method_certificate(
        "recursive", recursive_state, recursive_actions, recursive_solver
    )

    semantic = {
        "protocol": PROTOCOL,
        "uncertainty_contract": UNCERTAINTY_CONTRACT,
        "primitive_budget": primitive_budget,
        "state": recovery_state_to_dict(state),
        "methods": {
            item.method: {
                "status": item.state.status.value,
                "witness_action_key": item.state.witness_action_key,
                "witness_macro_depth_bound": (
                    item.state.witness_macro_depth_bound
                ),
                "actions": [
                    {
                        "key": action.action_key,
                        "status": action.status.value,
                        "outcome_count": action.outcome_count,
                        "physical_rehandle": action.physical_rehandle,
                    }
                    for action in item.actions
                ],
            }
            for item in (nominal, one_step, recursive)
        },
    }
    digest = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SnapshotCertificate(
        primitive_budget=primitive_budget,
        nominal=nominal,
        one_step=one_step,
        recursive=recursive,
        semantic_digest=digest,
    )


def recovery_state_to_dict(state: RecoveryState) -> dict:
    return {
        "rows": state.rows,
        "cols": state.cols,
        "traversable": [list(cell) for cell in sorted(state.traversable)],
        "storage_cells": [list(cell) for cell in sorted(state.storage_cells)],
        "exits": [list(cell) for cell in state.exits],
        "blocks": [
            {
                "label": block.label,
                "position": list(block.position),
                "remaining_time": float(block.remaining_time),
            }
            for block in state.blocks
        ],
        "agent_position": list(state.agent_position),
        "fixed_obstacles": [list(cell) for cell in sorted(state.fixed_obstacles)],
        "reserved_cells": [list(cell) for cell in sorted(state.reserved_cells)],
        "pickup_cells": [list(cell) for cell in sorted(state.pickup_cells)],
        "wait_cells": [list(cell) for cell in sorted(state.wait_cells)],
    }


def recovery_state_from_dict(value: Mapping) -> RecoveryState:
    return RecoveryState(
        rows=int(value["rows"]),
        cols=int(value["cols"]),
        traversable=frozenset(tuple(cell) for cell in value["traversable"]),
        storage_cells=frozenset(tuple(cell) for cell in value["storage_cells"]),
        exits=tuple(tuple(cell) for cell in value["exits"]),
        blocks=tuple(
            BlockView(
                str(block["label"]),
                tuple(block["position"]),
                float(block["remaining_time"]),
            )
            for block in value["blocks"]
        ),
        agent_position=tuple(value["agent_position"]),
        fixed_obstacles=frozenset(
            tuple(cell) for cell in value.get("fixed_obstacles", ())
        ),
        reserved_cells=frozenset(
            tuple(cell) for cell in value.get("reserved_cells", ())
        ),
        pickup_cells=frozenset(
            tuple(cell) for cell in value.get("pickup_cells", ())
        ),
        wait_cells=frozenset(
            tuple(cell) for cell in value.get("wait_cells", ())
        ),
    )


def method_to_dict(value: MethodCertificate) -> dict:
    return {
        "method": value.method,
        "state": {
            "status": value.state.status.value,
            "admitted": value.state.admitted,
            "witness_action_key": value.state.witness_action_key,
            "witness_macro_depth_bound": value.state.witness_macro_depth_bound,
            "reason": value.state.reason,
        },
        "admitted_action_keys": list(value.admitted_action_keys),
        "actions": [
            {
                "action_key": action.action_key,
                "status": action.status.value,
                "admitted": action.admitted,
                "outcome_count": action.outcome_count,
                "outcome_status_counts": dict(action.outcome_status_counts),
                "witness_macro_depth_bound": action.witness_macro_depth_bound,
                "physical_rehandle": action.physical_rehandle,
            }
            for action in value.actions
        ],
        "computation": {
            "expanded_nodes": value.expanded_nodes,
            "generated_outcomes": value.generated_outcomes,
            "cache_hits": value.cache_hits,
            "cache_entries": value.cache_entries,
            "cutoff_count": value.cutoff_count,
            "max_expansions": value.max_expansions,
        },
    }


def snapshot_certificate_to_dict(value: SnapshotCertificate) -> dict:
    return {
        "primitive_budget": value.primitive_budget,
        "semantic_digest": value.semantic_digest,
        "nominal": method_to_dict(value.nominal),
        "one_step": method_to_dict(value.one_step),
        "recursive": method_to_dict(value.recursive),
    }


def make_two_block_5x5_fixture() -> RecoveryState:
    """Repository-layout fixture used by focused exact tests."""

    rooms = ("###.#", "#...#", "#...#", "#...#", "#...#")
    traversable = frozenset(
        (row, column)
        for row, line in enumerate(rooms)
        for column, token in enumerate(line)
        if token != "#"
    )
    storage = frozenset(
        {
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 1),
            (3, 2),
            (3, 3),
        }
    )
    pickup = frozenset({(1, 3)})
    wait = frozenset({(0, 3)})
    return RecoveryState(
        rows=5,
        cols=5,
        traversable=traversable,
        storage_cells=storage,
        # A single designated recovery exit keeps this exact fixture small;
        # authenticated panel states retain all three live environment exits.
        exits=((4, 3),),
        blocks=(
            BlockView("A", (1, 1), 20.0),
            BlockView("B", (1, 2), 20.0),
        ),
        agent_position=(1, 1),
        fixed_obstacles=pickup | wait,
        pickup_cells=pickup,
        wait_cells=wait,
    )


__all__ = [
    "ActionCertificate",
    "DisturbanceContract",
    "ExecutionOutcome",
    "MethodCertificate",
    "NOMINAL_CONTRACT",
    "OnDemandRecoverySolver",
    "PROTOCOL",
    "ROBUST_CONTRACT",
    "SnapshotCertificate",
    "SolveStatus",
    "StateCertificate",
    "UNCERTAINTY_CONTRACT",
    "canonical_actions",
    "certify_snapshot",
    "clear_stop_cells",
    "execution_envelope",
    "make_two_block_5x5_fixture",
    "method_to_dict",
    "recovery_action_key",
    "recovery_state_from_dict",
    "recovery_state_to_dict",
    "snapshot_certificate_to_dict",
]
