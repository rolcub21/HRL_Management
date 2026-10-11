"""Exact reduced-yard recursive robust-kernel experiment.

This additive experiment solves two finite, closed-admission games using the
repository's strict recovery macros:

* a resource-augmented 4x4 game with one bounded extra execution step per
  macro; and
* a 1x5 corridor game with bounded terminal-position uncertainty.

It compares nominal containment, one-step containment against the nominal
kernel, and the least fixed-point robust completion kernel.  No network,
checkpoint, training loop, or environment episode is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, replace
import hashlib
from itertools import permutations
import json
import math
import os
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Optional, Sequence

from PSLAP.dynamic_yard import BlockView
from PSLAP.viability import (
    RecoveryAction,
    RecoveryActionKind,
    RecoveryState,
    apply_recovery_action,
    legal_recovery_actions,
)


PROTOCOL = "vcg_recursive_robust_kernel_reduced_yard_v1"
CLAIM_SCOPE = "exact_finite_closed_admission_recovery_game_not_full_episode_v1"
MERIT_CONTRACT = "synthetic_fixed_merit_qop_minus_lambda_qn_for_mechanism_only_v1"
HANDLING_LAMBDAS = (0.0, 0.025, 0.05, 0.1, 0.2)


@dataclass(frozen=True)
class GameOutcome:
    disturbance_id: str
    successor: Hashable
    tube_safe: bool
    primitive_steps: int
    uncertainty_kind: str

    def __post_init__(self) -> None:
        if not self.disturbance_id:
            raise ValueError("disturbance_id must be nonempty")
        if not isinstance(self.tube_safe, bool):
            raise TypeError("tube_safe must be bool")
        if (
            not isinstance(self.primitive_steps, int)
            or isinstance(self.primitive_steps, bool)
            or self.primitive_steps < 0
        ):
            raise TypeError("primitive_steps must be a non-negative integer")
        if not self.uncertainty_kind:
            raise ValueError("uncertainty_kind must be nonempty")


@dataclass(frozen=True)
class GameAction:
    key: str
    kind: str
    nominal_disturbance_id: str
    outcomes: tuple[GameOutcome, ...]
    q_operational: float
    q_rehandles: float
    physical_rehandles: int

    def __post_init__(self) -> None:
        if not self.key or not self.kind:
            raise ValueError("action key and kind must be nonempty")
        if not self.outcomes:
            raise ValueError("every action requires a nonempty outcome set")
        identifiers = [item.disturbance_id for item in self.outcomes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("action outcome disturbance identifiers must be unique")
        if self.nominal_disturbance_id not in set(identifiers):
            raise ValueError("nominal disturbance must belong to the outcome set")
        if any(
            not math.isfinite(float(value))
            for value in (self.q_operational, self.q_rehandles)
        ):
            raise ValueError("action merits must be finite")
        if (
            not isinstance(self.physical_rehandles, int)
            or isinstance(self.physical_rehandles, bool)
            or self.physical_rehandles < 0
        ):
            raise TypeError("physical_rehandles must be a non-negative integer")

    @property
    def nominal_outcome(self) -> GameOutcome:
        return next(
            item
            for item in self.outcomes
            if item.disturbance_id == self.nominal_disturbance_id
        )


@dataclass(frozen=True)
class KernelResult:
    states: frozenset[Hashable]
    ranks: Mapping[Hashable, int]
    layers: tuple[tuple[Hashable, ...], ...]

    def rank(self, state: Hashable) -> Optional[int]:
        return self.ranks.get(state)


@dataclass(frozen=True)
class BudgetedRecoveryNode:
    recovery_state: Optional[RecoveryState]
    remaining_primitive_steps: int
    failure: bool = False

    def __post_init__(self) -> None:
        if self.failure:
            if self.recovery_state is not None:
                raise ValueError("failure node cannot contain a RecoveryState")
            return
        if not isinstance(self.recovery_state, RecoveryState):
            raise TypeError("nonfailure budgeted node requires a RecoveryState")
        if (
            isinstance(self.remaining_primitive_steps, bool)
            or self.remaining_primitive_steps < 0
        ):
            raise ValueError("remaining primitive steps must be non-negative")


@dataclass
class FiniteRobustGame:
    states: frozenset[Hashable]
    terminal_states: frozenset[Hashable]
    unsafe_states: frozenset[Hashable]
    actions: Mapping[Hashable, tuple[GameAction, ...]]
    state_ids: Mapping[Hashable, str]
    uncertainty_contract: str

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("finite game must contain states")
        if not self.terminal_states <= self.states:
            raise ValueError("terminal states must belong to the game")
        if not self.unsafe_states <= self.states:
            raise ValueError("unsafe states must belong to the game")
        if self.terminal_states & self.unsafe_states:
            raise ValueError("terminal and unsafe states must be disjoint")
        if set(self.actions) != set(self.states):
            raise ValueError("actions mapping must cover every state exactly")
        if set(self.state_ids) != set(self.states):
            raise ValueError("state_ids must cover every state exactly")
        identifiers = list(self.state_ids.values())
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("state identifiers must be unique")
        for state, choices in self.actions.items():
            keys = [choice.key for choice in choices]
            if len(keys) != len(set(keys)):
                raise ValueError("action keys must be unique within a state")
            if state in self.terminal_states and choices:
                raise ValueError("terminal states cannot expose actions")
            for choice in choices:
                if any(outcome.successor not in self.states for outcome in choice.outcomes):
                    raise ValueError("every outcome successor must belong to the game")
        if not self.uncertainty_contract:
            raise ValueError("uncertainty contract must be nonempty")

    def state_sort_key(self, state: Hashable) -> str:
        return self.state_ids[state]


def _action_outcomes(
    action: GameAction,
    *,
    nominal_only: bool,
) -> tuple[GameOutcome, ...]:
    return (action.nominal_outcome,) if nominal_only else action.outcomes


def _eligible_against(
    action: GameAction,
    target: frozenset[Hashable] | set[Hashable],
    *,
    nominal_only: bool,
) -> bool:
    outcomes = _action_outcomes(action, nominal_only=nominal_only)
    return all(item.tube_safe and item.successor in target for item in outcomes)


def compute_kernel(
    game: FiniteRobustGame,
    *,
    nominal_only: bool,
) -> KernelResult:
    """Compute the least completion fixed point synchronously."""

    current = set(game.terminal_states)
    ranks: dict[Hashable, int] = {state: 0 for state in current}
    layers: list[tuple[Hashable, ...]] = [
        tuple(sorted(current, key=game.state_sort_key))
    ]
    rank = 0
    while True:
        added = tuple(
            sorted(
                (
                    state
                    for state in game.states - current - game.unsafe_states
                    if any(
                        _eligible_against(
                            action,
                            current,
                            nominal_only=nominal_only,
                        )
                        for action in game.actions[state]
                    )
                ),
                key=game.state_sort_key,
            )
        )
        if not added:
            break
        rank += 1
        current.update(added)
        ranks.update({state: rank for state in added})
        layers.append(added)
    return KernelResult(
        states=frozenset(current),
        ranks=ranks,
        layers=tuple(layers),
    )


def compute_one_step_set(
    game: FiniteRobustGame,
    nominal: KernelResult,
) -> frozenset[Hashable]:
    """One-macro robust containment against the nominal recovery kernel."""

    return frozenset(
        set(game.terminal_states)
        | {
            state
            for state in game.states - game.terminal_states - game.unsafe_states
            if any(
                _eligible_against(
                    action,
                    nominal.states,
                    nominal_only=False,
                )
                for action in game.actions[state]
            )
        }
    )


def nominal_frontier(
    game: FiniteRobustGame,
    state: Hashable,
    nominal: KernelResult,
) -> tuple[GameAction, ...]:
    return tuple(
        action
        for action in game.actions[state]
        if _eligible_against(action, nominal.states, nominal_only=True)
    )


def one_step_frontier(
    game: FiniteRobustGame,
    state: Hashable,
    nominal: KernelResult,
) -> tuple[GameAction, ...]:
    return tuple(
        action
        for action in game.actions[state]
        if _eligible_against(action, nominal.states, nominal_only=False)
    )


def robust_invariant_frontier(
    game: FiniteRobustGame,
    state: Hashable,
    robust: KernelResult,
) -> tuple[GameAction, ...]:
    return tuple(
        action
        for action in game.actions[state]
        if _eligible_against(action, robust.states, nominal_only=False)
    )


def _strictly_decreases(
    action: GameAction,
    state: Hashable,
    kernel: KernelResult,
    *,
    nominal_only: bool,
) -> bool:
    current_rank = kernel.rank(state)
    if current_rank is None or current_rank == 0:
        return False
    outcomes = _action_outcomes(action, nominal_only=nominal_only)
    return all(
        item.tube_safe
        and kernel.rank(item.successor) is not None
        and int(kernel.rank(item.successor)) < current_rank
        for item in outcomes
    )


def decreasing_frontier(
    candidates: Iterable[GameAction],
    state: Hashable,
    kernel: KernelResult,
    *,
    nominal_only: bool,
) -> tuple[GameAction, ...]:
    return tuple(
        action
        for action in candidates
        if _strictly_decreases(
            action,
            state,
            kernel,
            nominal_only=nominal_only,
        )
    )


def nonincreasing_frontier(
    candidates: Iterable[GameAction],
    state: Hashable,
    kernel: KernelResult,
    *,
    nominal_only: bool,
) -> tuple[GameAction, ...]:
    """Keep actions whose modeled successors cannot increase kernel rank."""

    current_rank = kernel.rank(state)
    if current_rank is None:
        return ()
    return tuple(
        action
        for action in candidates
        if all(
            outcome.tube_safe
            and kernel.rank(outcome.successor) is not None
            and int(kernel.rank(outcome.successor)) <= current_rank
            for outcome in _action_outcomes(
                action,
                nominal_only=nominal_only,
            )
        )
    )


def select_action(
    candidates: Sequence[GameAction],
    *,
    handling_lambda: float,
) -> GameAction:
    if not candidates:
        raise ValueError("cannot select from an empty frontier")
    handling_lambda = float(handling_lambda)
    if not math.isfinite(handling_lambda) or handling_lambda < 0.0:
        raise ValueError("handling_lambda must be finite and non-negative")
    return min(
        candidates,
        key=lambda action: (
            -(
                action.q_operational
                - handling_lambda * action.q_rehandles
            ),
            action.key,
        ),
    )


def _frontier_for_method(
    method: str,
    game: FiniteRobustGame,
    state: Hashable,
    nominal: KernelResult,
    robust: KernelResult,
) -> tuple[GameAction, ...]:
    if method == "nominal":
        return nominal_frontier(game, state, nominal)
    if method == "one_step":
        return one_step_frontier(game, state, nominal)
    if method == "recursive":
        return robust_invariant_frontier(game, state, robust)
    raise ValueError(f"unknown method: {method}")


def _kernel_for_method(
    method: str,
    nominal: KernelResult,
    robust: KernelResult,
) -> KernelResult:
    return robust if method == "recursive" else nominal


def exhaust_policy_branches(
    game: FiniteRobustGame,
    *,
    start_state: Hashable,
    method: str,
    handling_lambda: float,
    nominal: KernelResult,
    one_step: frozenset[Hashable],
    robust: KernelResult,
    max_nonprogress_macros: int = 1,
) -> dict:
    """Exhaust every declared disturbance branch under one fixed policy."""

    if max_nonprogress_macros < 0:
        raise ValueError("max_nonprogress_macros must be non-negative")
    method_kernel = _kernel_for_method(method, nominal, robust)
    method_members = (
        robust.states
        if method == "recursive"
        else one_step
        if method == "one_step"
        else nominal.states
    )
    leaves = []
    empty_frontier_visits = 0

    def walk(
        state: Hashable,
        nonprogress: int,
        path_memory: frozenset[tuple[Hashable, int]],
        trace: tuple[dict, ...],
        primitive_steps: int,
        rehandles: int,
    ) -> None:
        nonlocal empty_frontier_visits
        if state in game.terminal_states:
            leaves.append(
                {
                    "status": "complete",
                    "primitive_steps": primitive_steps,
                    "physical_rehandles": rehandles,
                    "macro_count": len(trace),
                    "trace": list(trace),
                }
            )
            return
        if state in game.unsafe_states:
            leaves.append(
                {
                    "status": "unsafe_successor",
                    "primitive_steps": primitive_steps,
                    "physical_rehandles": rehandles,
                    "macro_count": len(trace),
                    "trace": list(trace),
                }
            )
            return
        # Once the allowance is exhausted, larger counter values represent
        # the same policy phase.  Saturating the key makes even non-robust
        # comparison methods terminate cleanly if realized disturbances form
        # a cycle outside their nominal progress model.
        memory_key = (
            state,
            min(nonprogress, max_nonprogress_macros),
        )
        if memory_key in path_memory:
            leaves.append(
                {
                    "status": "cycle",
                    "primitive_steps": primitive_steps,
                    "physical_rehandles": rehandles,
                    "macro_count": len(trace),
                    "trace": list(trace),
                }
            )
            return
        frontier = _frontier_for_method(
            method, game, state, nominal, robust
        )
        # The recursive liveness guard permits one preference-selected macro
        # only when every realized successor has non-increasing robust rank.
        # It then forces a strict decrease, so robust rank cannot oscillate.
        if method == "recursive":
            frontier = nonincreasing_frontier(
                frontier,
                state,
                robust,
                nominal_only=False,
            )
        if nonprogress >= max_nonprogress_macros:
            frontier = decreasing_frontier(
                frontier,
                state,
                method_kernel,
                nominal_only=(method == "nominal"),
            )
        if not frontier:
            empty_frontier_visits += 1
            leaves.append(
                {
                    "status": "empty_frontier",
                    "primitive_steps": primitive_steps,
                    "physical_rehandles": rehandles,
                    "macro_count": len(trace),
                    "trace": list(trace),
                }
            )
            return
        action = select_action(frontier, handling_lambda=handling_lambda)
        current_rank = method_kernel.rank(state)
        for outcome in action.outcomes:
            event = {
                "state": game.state_ids[state],
                "action": action.key,
                "disturbance": outcome.disturbance_id,
                "successor": game.state_ids[outcome.successor],
                "tube_safe": outcome.tube_safe,
            }
            if not outcome.tube_safe:
                leaves.append(
                    {
                        "status": "unsafe_tube",
                        "primitive_steps": primitive_steps
                        + outcome.primitive_steps,
                        "physical_rehandles": rehandles
                        + action.physical_rehandles,
                        "macro_count": len(trace) + 1,
                        "trace": list(trace + (event,)),
                    }
                )
                continue
            successor_rank = method_kernel.rank(outcome.successor)
            progressed = (
                current_rank is not None
                and successor_rank is not None
                and successor_rank < current_rank
            )
            walk(
                outcome.successor,
                0 if progressed else nonprogress + 1,
                path_memory | {memory_key},
                trace + (event,),
                primitive_steps + outcome.primitive_steps,
                rehandles + action.physical_rehandles,
            )

    walk(start_state, 0, frozenset(), (), 0, 0)
    counts = Counter(item["status"] for item in leaves)
    completed = [item for item in leaves if item["status"] == "complete"]
    failures = [item for item in leaves if item["status"] != "complete"]
    return {
        "method": method,
        "handling_lambda": handling_lambda,
        "start_state": game.state_ids[start_state],
        "start_in_method_admissible_set": start_state in method_members,
        "rank_reference": (
            "recursive_completion_rank"
            if method == "recursive"
            else "nominal_completion_rank"
        ),
        "branch_count": len(leaves),
        "status_counts": dict(sorted(counts.items())),
        "all_branches_complete": bool(leaves) and not failures,
        "empty_frontier_visits": empty_frontier_visits,
        "completed_worst_primitive_steps": (
            max(item["primitive_steps"] for item in completed)
            if completed
            else None
        ),
        "completed_best_primitive_steps": (
            min(item["primitive_steps"] for item in completed)
            if completed
            else None
        ),
        "completed_max_rehandles": (
            max(item["physical_rehandles"] for item in completed)
            if completed
            else None
        ),
        "completed_min_rehandles": (
            min(item["physical_rehandles"] for item in completed)
            if completed
            else None
        ),
        "counterexample": None if not failures else failures[0],
        "complete_example": None if not completed else completed[0],
    }


def _recovery_action_key(action: RecoveryAction) -> str:
    return (
        f"{action.kind.value}:{action.block_label}:"
        f"{action.destination[0]}:{action.destination[1]}"
    )


def _action_merits(action: RecoveryAction) -> tuple[float, float, int]:
    rehandle = int(action.kind is RecoveryActionKind.RELOCATION)
    return -float(action.steps) / 10.0, float(rehandle), rehandle


def _physical_state_key(state: RecoveryState) -> tuple:
    return (
        len(state.blocks),
        tuple((item.label, item.position) for item in state.blocks),
        state.agent_position,
    )


def _duration_initial_state() -> RecoveryState:
    return RecoveryState(
        rows=4,
        cols=4,
        traversable=frozenset(
            {(0, 2), (1, 1), (1, 2), (2, 1), (2, 2), (3, 2)}
        ),
        storage_cells=frozenset({(1, 1), (2, 1), (2, 2)}),
        exits=((3, 2),),
        blocks=(
            BlockView("A", (1, 1), 0.0),
            BlockView("B", (2, 1), 0.0),
        ),
        agent_position=(1, 1),
        fixed_obstacles=frozenset({(0, 2), (1, 2)}),
        pickup_cells=frozenset({(1, 2)}),
        wait_cells=frozenset({(0, 2)}),
    )


def _reachable_physical_states(initial: RecoveryState) -> frozenset[RecoveryState]:
    states = {initial}
    frontier = deque([initial])
    while frontier:
        current = frontier.popleft()
        for action in legal_recovery_actions(current):
            successor = apply_recovery_action(current, action)
            if successor not in states:
                states.add(successor)
                frontier.append(successor)
    return frozenset(states)


def build_duration_game(
    *,
    max_budget: int = 20,
    delays: Sequence[int] = (0, 1),
) -> tuple[FiniteRobustGame, Mapping[int, BudgetedRecoveryNode]]:
    delays = tuple(int(value) for value in delays)
    if not delays or 0 not in delays or any(value < 0 for value in delays):
        raise ValueError("delays must be non-negative and include nominal zero")
    if len(delays) != len(set(delays)):
        raise ValueError("delays must be unique")
    initial = _duration_initial_state()
    physical_states = tuple(
        sorted(_reachable_physical_states(initial), key=_physical_state_key)
    )
    failure = BudgetedRecoveryNode(None, -1, failure=True)
    nodes = {
        BudgetedRecoveryNode(state, budget)
        for state in physical_states
        for budget in range(max_budget + 1)
    }
    nodes.add(failure)
    initial_nodes = {
        budget: BudgetedRecoveryNode(initial, budget)
        for budget in range(max_budget + 1)
    }

    def node_id(node: BudgetedRecoveryNode) -> str:
        if node.failure:
            return "FAIL:horizon"
        state = node.recovery_state
        blocks = ",".join(
            f"{item.label}@{item.position[0]}:{item.position[1]}"
            for item in state.blocks
        ) or "empty"
        return (
            f"budget:{node.remaining_primitive_steps}|agent:"
            f"{state.agent_position[0]}:{state.agent_position[1]}|{blocks}"
        )

    actions: dict[Hashable, tuple[GameAction, ...]] = {}
    for node in nodes:
        if node.failure or not node.recovery_state.blocks:
            actions[node] = ()
            continue
        choices = []
        for action in legal_recovery_actions(node.recovery_state):
            physical_successor = apply_recovery_action(
                node.recovery_state, action
            )
            outcomes = []
            for delay in delays:
                cost = action.steps + delay
                successor = (
                    BudgetedRecoveryNode(
                        physical_successor,
                        node.remaining_primitive_steps - cost,
                    )
                    if cost <= node.remaining_primitive_steps
                    else failure
                )
                outcomes.append(
                    GameOutcome(
                        disturbance_id=f"delay:{delay}",
                        successor=successor,
                        tube_safe=True,
                        primitive_steps=cost,
                        uncertainty_kind="extra_macro_duration",
                    )
                )
            qop, qn, rehandles = _action_merits(action)
            choices.append(
                GameAction(
                    key=_recovery_action_key(action),
                    kind=action.kind.value,
                    nominal_disturbance_id="delay:0",
                    outcomes=tuple(outcomes),
                    q_operational=qop,
                    q_rehandles=qn,
                    physical_rehandles=rehandles,
                )
            )
        actions[node] = tuple(sorted(choices, key=lambda item: item.key))

    game = FiniteRobustGame(
        states=frozenset(nodes),
        terminal_states=frozenset(
            node
            for node in nodes
            if not node.failure and not node.recovery_state.blocks
        ),
        unsafe_states=frozenset({failure}),
        actions=actions,
        state_ids={node: node_id(node) for node in nodes},
        uncertainty_contract=(
            "bounded_extra_primitive_steps_per_strict_macro:"
            + ",".join(str(value) for value in delays)
        ),
    )
    return game, initial_nodes


def _corridor_states() -> frozenset[RecoveryState]:
    traversable = frozenset((0, column) for column in range(5))
    storage = ((0, 1), (0, 2), (0, 3))
    exits = ((0, 4),)
    states = set()
    for agent in sorted(traversable):
        states.add(
            RecoveryState(
                1, 5, traversable, frozenset(storage), exits, (), agent
            )
        )
    for label in ("A", "B"):
        for position in storage:
            for agent in sorted(traversable):
                states.add(
                    RecoveryState(
                        1,
                        5,
                        traversable,
                        frozenset(storage),
                        exits,
                        (BlockView(label, position, 0.0),),
                        agent,
                    )
                )
    for a_position, b_position in permutations(storage, 2):
        for agent in sorted(traversable):
            states.add(
                RecoveryState(
                    1,
                    5,
                    traversable,
                    frozenset(storage),
                    exits,
                    (
                        BlockView("A", a_position, 0.0),
                        BlockView("B", b_position, 0.0),
                    ),
                    agent,
                )
            )
    return frozenset(states)


def build_endpoint_game() -> FiniteRobustGame:
    states = _corridor_states()

    def state_id(state: RecoveryState) -> str:
        blocks = ",".join(
            f"{item.label}@{item.position[1]}" for item in state.blocks
        ) or "empty"
        return f"agent:{state.agent_position[1]}|{blocks}"

    actions = {}
    for state in states:
        choices = []
        for action in legal_recovery_actions(state):
            nominal = apply_recovery_action(state, action)
            occupied = frozenset(nominal.occupancy())
            destination = action.destination
            endpoints = {destination}
            for neighbor in (
                (destination[0], destination[1] - 1),
                (destination[0], destination[1] + 1),
            ):
                if neighbor in state.traversable and neighbor not in occupied:
                    endpoints.add(neighbor)
            outcomes = tuple(
                GameOutcome(
                    disturbance_id=f"endpoint:{endpoint[0]}:{endpoint[1]}",
                    successor=replace(nominal, agent_position=endpoint),
                    tube_safe=True,
                    primitive_steps=action.steps,
                    uncertainty_kind="terminal_position_radius_one",
                )
                for endpoint in sorted(endpoints)
            )
            qop, qn, rehandles = _action_merits(action)
            choices.append(
                GameAction(
                    key=_recovery_action_key(action),
                    kind=action.kind.value,
                    nominal_disturbance_id=(
                        f"endpoint:{destination[0]}:{destination[1]}"
                    ),
                    outcomes=outcomes,
                    q_operational=qop,
                    q_rehandles=qn,
                    physical_rehandles=rehandles,
                )
            )
        actions[state] = tuple(sorted(choices, key=lambda item: item.key))
    return FiniteRobustGame(
        states=states,
        terminal_states=frozenset(state for state in states if not state.blocks),
        unsafe_states=frozenset(),
        actions=actions,
        state_ids={state: state_id(state) for state in states},
        uncertainty_contract=(
            "nominal_or_clear_traversable_radius_one_terminal_agent_position_v1"
        ),
    )


def _kernel_digest(
    game: FiniteRobustGame,
    kernel: KernelResult,
    *,
    include_invariant_frontier: bool,
) -> str:
    payload = {
        "states": sorted(game.state_ids[state] for state in kernel.states),
        "ranks": {
            game.state_ids[state]: rank
            for state, rank in sorted(
                kernel.ranks.items(), key=lambda item: game.state_sort_key(item[0])
            )
        },
    }
    if include_invariant_frontier:
        payload["frontiers"] = {
            game.state_ids[state]: sorted(
                action.key
                for action in robust_invariant_frontier(game, state, kernel)
            )
            for state in sorted(kernel.states, key=game.state_sort_key)
            if state not in game.terminal_states
        }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _rank_histogram(kernel: KernelResult) -> dict[str, int]:
    return {
        str(rank): count
        for rank, count in sorted(Counter(kernel.ranks.values()).items())
    }


def _kernel_summary(
    game: FiniteRobustGame,
    nominal: KernelResult,
    one_step: frozenset[Hashable],
    robust: KernelResult,
) -> dict:
    action_count = sum(len(value) for value in game.actions.values())
    outcome_count = sum(
        len(action.outcomes)
        for value in game.actions.values()
        for action in value
    )
    tube_removed = 0
    successor_removed = 0
    recursive_removed = 0
    for state in game.states - game.terminal_states - game.unsafe_states:
        nominal_actions = set(nominal_frontier(game, state, nominal))
        one_actions = set(one_step_frontier(game, state, nominal))
        recursive_actions = set(robust_invariant_frontier(game, state, robust))
        for action in nominal_actions - one_actions:
            if any(not item.tube_safe for item in action.outcomes):
                tube_removed += 1
            else:
                successor_removed += 1
        recursive_removed += len(one_actions - recursive_actions)
    return {
        "state_count": len(game.states),
        "terminal_state_count": len(game.terminal_states),
        "unsafe_state_count": len(game.unsafe_states),
        "physical_action_count": action_count,
        "declared_outcome_count": outcome_count,
        "nominal_kernel_count": len(nominal.states),
        "one_step_set_count": len(one_step),
        "recursive_kernel_count": len(robust.states),
        "one_step_not_recursive_count": len(one_step - robust.states),
        "nominal_rank_histogram": _rank_histogram(nominal),
        "recursive_rank_histogram": _rank_histogram(robust),
        "nominal_to_one_step_tube_removals": tube_removed,
        "nominal_to_one_step_successor_removals": successor_removed,
        "one_step_to_recursive_action_removals": recursive_removed,
        "recursive_kernel_digest": _kernel_digest(
            game, robust, include_invariant_frontier=True
        ),
    }


def _minimum_initial_budget(
    initial_nodes: Mapping[int, BudgetedRecoveryNode],
    members: frozenset[Hashable],
) -> Optional[int]:
    values = [budget for budget, state in initial_nodes.items() if state in members]
    return min(values) if values else None


def _duration_report() -> dict:
    game, initial = build_duration_game(max_budget=20, delays=(0, 1))
    nominal = compute_kernel(game, nominal_only=True)
    one_step = compute_one_step_set(game, nominal)
    robust = compute_kernel(game, nominal_only=False)
    summary = _kernel_summary(game, nominal, one_step, robust)
    thresholds = {
        "nominal_minimum_budget": _minimum_initial_budget(initial, nominal.states),
        "one_step_minimum_budget": _minimum_initial_budget(initial, one_step),
        "recursive_minimum_budget": _minimum_initial_budget(initial, robust.states),
    }
    comparisons = {
        str(value): {
            method: exhaust_policy_branches(
                game,
                start_state=initial[16],
                method=method,
                handling_lambda=value,
                nominal=nominal,
                one_step=one_step,
                robust=robust,
                max_nonprogress_macros=1,
            )
            for method in ("nominal", "one_step", "recursive")
        }
        for value in HANDLING_LAMBDAS
    }
    recursive_tradeoff = {
        str(value): exhaust_policy_branches(
            game,
            start_state=initial[18],
            method="recursive",
            handling_lambda=value,
            nominal=nominal,
            one_step=one_step,
            robust=robust,
            max_nonprogress_macros=1,
        )
        for value in HANDLING_LAMBDAS
    }

    expanded_game, expanded_initial = build_duration_game(
        max_budget=20, delays=(0, 1, 2)
    )
    expanded_nominal = compute_kernel(expanded_game, nominal_only=True)
    expanded_robust = compute_kernel(expanded_game, nominal_only=False)
    out_of_set = {
        "declared_delays": [0, 1],
        "observed_delay": 2,
        "covered_by_declared_guarantee": False,
        "budget_15_in_declared_recursive_kernel": initial[15] in robust.states,
        "budget_15_in_expanded_recursive_kernel": (
            expanded_initial[15] in expanded_robust.states
        ),
        "expanded_recursive_minimum_budget": _minimum_initial_budget(
            expanded_initial, expanded_robust.states
        ),
        "expanded_nominal_kernel_count": len(expanded_nominal.states),
    }
    return {
        "uncertainty_contract": game.uncertainty_contract,
        "resource_coordinate_in_state": "remaining_primitive_steps",
        "summary": summary,
        "initial_budget_thresholds": thresholds,
        "common_domain_budget_16_comparison": comparisons,
        "recursive_budget_18_lambda_sweep": recursive_tradeoff,
        "out_of_set_scope": out_of_set,
    }


def _endpoint_example_state(game: FiniteRobustGame) -> RecoveryState:
    return next(
        state
        for state in game.states
        if isinstance(state, RecoveryState)
        and state.agent_position == (0, 1)
        and tuple((item.label, item.position) for item in state.blocks)
        == (("A", (0, 1)), ("B", (0, 3)))
    )


def _endpoint_report() -> dict:
    game = build_endpoint_game()
    nominal = compute_kernel(game, nominal_only=True)
    one_step = compute_one_step_set(game, nominal)
    robust = compute_kernel(game, nominal_only=False)
    example = _endpoint_example_state(game)
    one_frontier = one_step_frontier(game, example, nominal)
    recursive_frontier = robust_invariant_frontier(game, example, robust)
    return {
        "uncertainty_contract": game.uncertainty_contract,
        "summary": _kernel_summary(game, nominal, one_step, robust),
        "separating_state": {
            "state": game.state_ids[example],
            "nominal_rank": nominal.rank(example),
            "recursive_rank": robust.rank(example),
            "one_step_frontier": [action.key for action in one_frontier],
            "recursive_invariant_frontier": [
                action.key for action in recursive_frontier
            ],
            "one_step_only_actions": sorted(
                {action.key for action in one_frontier}
                - {action.key for action in recursive_frontier}
            ),
        },
    }


def _assert_kernel_contract(
    game: FiniteRobustGame,
    robust: KernelResult,
) -> None:
    for state in robust.states - game.terminal_states:
        invariant = robust_invariant_frontier(game, state, robust)
        if not invariant:
            raise RuntimeError("robust kernel state has an empty invariant frontier")
        for action in invariant:
            if not all(
                outcome.tube_safe and outcome.successor in robust.states
                for outcome in action.outcomes
            ):
                raise RuntimeError("robust invariant frontier escaped the kernel")
        decreasing = decreasing_frontier(
            invariant,
            state,
            robust,
            nominal_only=False,
        )
        if not decreasing:
            raise RuntimeError("robust kernel state has no rank-decreasing action")


def build_report() -> dict:
    duration_game, _ = build_duration_game()
    duration_robust = compute_kernel(duration_game, nominal_only=False)
    _assert_kernel_contract(duration_game, duration_robust)
    endpoint_game = build_endpoint_game()
    endpoint_robust = compute_kernel(endpoint_game, nominal_only=False)
    _assert_kernel_contract(endpoint_game, endpoint_robust)

    duration = _duration_report()
    endpoint = _endpoint_report()
    thresholds = duration["initial_budget_thresholds"]
    duration_comparison = duration["common_domain_budget_16_comparison"]
    recursive_tradeoff = duration["recursive_budget_18_lambda_sweep"]
    checks = {
        "resource_augmentation_separates_membership": thresholds
        == {
            "nominal_minimum_budget": 13,
            "one_step_minimum_budget": 14,
            "recursive_minimum_budget": 15,
        },
        "one_step_and_recursive_sets_differ": (
            endpoint["summary"]["one_step_not_recursive_count"] == 6
        ),
        "recursive_completes_every_budget16_branch_for_all_lambdas": all(
            rows["recursive"]["all_branches_complete"]
            for rows in duration_comparison.values()
        ),
        "one_step_has_a_budget16_counterexample_at_low_lambda": (
            not duration_comparison["0.0"]["one_step"][
                "all_branches_complete"
            ]
        ),
        "recursive_lambda_changes_rehandling_without_changing_certificate": (
            recursive_tradeoff["0.0"]["completed_max_rehandles"] == 1
            and recursive_tradeoff["0.2"]["completed_max_rehandles"] == 0
            and all(
                row["all_branches_complete"]
                for row in recursive_tradeoff.values()
            )
        ),
        "out_of_set_delay_is_theorem_silent": (
            duration["out_of_set_scope"][
                "covered_by_declared_guarantee"
            ]
            is False
            and duration["out_of_set_scope"][
                "budget_15_in_declared_recursive_kernel"
            ]
            is True
            and duration["out_of_set_scope"][
                "budget_15_in_expanded_recursive_kernel"
            ]
            is False
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"recursive robust experiment failed: {checks!r}")
    return {
        "protocol": PROTOCOL,
        "claim_scope": CLAIM_SCOPE,
        "merit_contract": MERIT_CONTRACT,
        "handling_lambdas": list(HANDLING_LAMBDAS),
        "fixed_point_object": (
            "least_fixed_point_robust_completion_winning_set"
        ),
        "completion_policy_contract": {
            "preference_frontier": (
                "all_declared_successors_in_recursive_set_with_"
                "nonincreasing_robust_rank"
            ),
            "same_rank_macro_allowance": 1,
            "then_force_strict_rank_decrease": True,
            "completion_bound_macros": (
                "at_most_2_times_initial_recursive_rank"
            ),
            "invariant_frontier_alone_implies_completion": False,
        },
        "uncertainty_enumeration": {
            "declared_outcomes_are_finite_and_complete": True,
            "empty_outcome_sets_rejected": True,
            "out_of_universe_successors_rejected": True,
            "unknown_successors_present": False,
        },
        "tube_scope": {
            "native_strict_macro_routes_used": True,
            "recursive_games_add_tube_perturbations": False,
            "all_recursive_game_tubes_are_safe_by_construction": True,
            "unsafe_tube_rejection_is_covered_by": (
                "vcg_robust_execution_probe.py"
            ),
        },
        "training_or_learning": False,
        "checkpoint_or_model_used": False,
        "closed_admission_current_work_only": True,
        "full_dynamic_episode_claim": False,
        "real_world_validation": False,
        "joint_duration_and_endpoint_kernel": False,
        "duration_game": duration,
        "endpoint_game": endpoint,
        "checks": checks,
        "status": "passed",
    }


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/vcg-recursive-robust-kernel-reduced-yard/"
            "recursive-robust-report.json"
        ),
    )
    args = parser.parse_args()
    report = build_report()
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output.resolve()),
                "checks": report["checks"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
