"""Execute an explicit bounded macro realization in the live 5x5 environment.

This module is deliberately additive: the selected SAFE candidate is executed
by the repository's existing option executor, which reaches the authoritative
``env.step`` implementation for every primitive.  Only after that option has
completed successfully do we realize the declared disturbance with zero or one
real ``WAIT`` and, optionally, one real cardinal move.

The adjacent stop set comes from the candidate's certified successor
``RecoveryState``.  It excludes occupied, fixed-obstacle, reserved, pickup,
and waiting cells exactly as the finite snapshot envelope does.  The target
is checked again against the mutated live environment immediately before
movement.

This is an execution adapter, not a robustness certificate.  In particular,
``WAIT`` advances the complete live environment (block clocks and arrivals),
whereas the current frozen-snapshot certificate models delay only as primitive
budget consumption.  A caller must keep that abstraction gap explicit.

The enforced disturbed-policy scope is strict ``Deliver`` and ``Reconfigure``
recovery macros with a bound ``RecoveryAction``.  The modeled path is fixed:
even a successful live option is rejected for theorem-facing use if it
replanned.  If a final delivery makes the live environment terminal, only its
nominal realization is executable: a requested post-macro WAIT or move is
suppressed and reported as an incomplete realization.  The adapter never calls
``env.step`` after a terminal transition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional

from PSLAP.dynamic_yard import Cell
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    ViabilityActionCandidate,
    ViabilityActionType,
)
from train_viability_graph_smdp import MacroExecution, execute_certified_macro


LIVE_BOUNDED_MACRO_EXECUTION_CONTRACT = (
    "selected_safe_option_via_live_env_step_then_wait_0_or_1_"
    "and_optional_declared_clear_cardinal_stop_fixed_path_no_replan_v1"
)

_CARDINAL_ACTION_FOR_DELTA = {
    (-1, 0): "UP",
    (1, 0): "DOWN",
    (0, -1): "LEFT",
    (0, 1): "RIGHT",
}


@dataclass(frozen=True)
class BoundedMacroRealization:
    """One deterministic member of the bounded live-execution envelope."""

    delay_steps: int
    adjacent_stop: Optional[Cell] = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.delay_steps, bool)
            or not isinstance(self.delay_steps, int)
            or self.delay_steps not in (0, 1)
        ):
            raise ValueError("delay_steps must be exactly 0 or 1")
        if self.adjacent_stop is not None:
            cell = tuple(self.adjacent_stop)
            if (
                len(cell) != 2
                or any(isinstance(value, bool) for value in cell)
                or not all(isinstance(value, int) for value in cell)
            ):
                raise TypeError("adjacent_stop must be a two-integer cell")
            object.__setattr__(self, "adjacent_stop", cell)

    @property
    def requested_injected_steps(self) -> int:
        return int(self.delay_steps) + int(self.adjacent_stop is not None)

    @property
    def realization_id(self) -> str:
        stop = self.adjacent_stop
        suffix = "nominal" if stop is None else f"{stop[0]}:{stop[1]}"
        return f"delay:{self.delay_steps}|stop:{suffix}"


@dataclass(frozen=True)
class InjectedPrimitiveTransition:
    """Audit record for one injected primitive that really reached ``env.step``."""

    phase: str
    action_name: str
    action_id: int
    before_agent_position: Cell
    after_agent_position: Cell
    time_step_before: int
    time_step_after: int
    reward: float
    terminal: bool
    info: Mapping[str, object]


@dataclass(frozen=True)
class BoundedMacroExecution:
    """Observed base macro plus its attempted explicit disturbance realization."""

    candidate_key: str
    action_type: str
    realization: BoundedMacroRealization
    declared_adjacent_stops: tuple[Cell, ...]
    base_execution: MacroExecution
    base_replan_count: Optional[int]
    base_primitive_steps: int
    injected_primitive_steps: int
    total_primitive_steps: int
    start_agent_position: Cell
    base_endpoint: Cell
    realized_endpoint: Cell
    start_time_step: int
    base_end_time_step: int
    realized_end_time_step: int
    injected_transitions: tuple[InjectedPrimitiveTransition, ...]
    injected_raw_return: float
    injected_discounted_return_from_macro_start: float
    total_raw_return: float
    total_discounted_return: float
    env_terminal: bool
    realization_complete: bool
    failure_reason: Optional[str]

    @property
    def replay_terminal(self) -> bool:
        return bool(
            self.env_terminal
            or self.base_execution.replay_terminal
            or not self.realization_complete
        )


def _neighbors(cell: Cell) -> tuple[Cell, ...]:
    row, column = cell
    return (
        (row - 1, column),
        (row, column - 1),
        (row, column + 1),
        (row + 1, column),
    )


def _validate_recovery_candidate(candidate: ViabilityActionCandidate) -> None:
    if not isinstance(candidate, ViabilityActionCandidate):
        raise TypeError("candidate must be a ViabilityActionCandidate")
    if candidate.certificate.status is not ViabilityStatus.SAFE:
        raise ValueError("only exactly SAFE candidates may be executed")
    if candidate.action_type not in (
        ViabilityActionType.DELIVER,
        ViabilityActionType.RECONFIGURE,
    ):
        raise ValueError(
            "bounded disturbance execution is restricted to strict recovery "
            "candidates"
        )
    if candidate.recovery_action is None:
        raise ValueError("strict recovery candidate lacks a RecoveryAction")


def declared_clear_adjacent_stops(
    candidate: ViabilityActionCandidate,
) -> tuple[Cell, ...]:
    """Return the candidate-successor cells declared for one-cell stopping error."""

    _validate_recovery_candidate(candidate)
    successor = candidate.successor_state
    excluded = (
        successor.fixed_obstacles
        | successor.reserved_cells
        | successor.pickup_cells
        | successor.wait_cells
        | frozenset(successor.occupancy())
    )
    return tuple(
        sorted(
            cell
            for cell in _neighbors(successor.agent_position)
            if cell in successor.traversable and cell not in excluded
        )
    )


def _agent_position(env) -> Cell:
    value = tuple(env.current_state)
    if len(value) != 2 or not all(isinstance(item, int) for item in value):
        raise TypeError("env.current_state must be a two-integer agent cell")
    return value


def _time_step(env) -> int:
    value = env.time_steps
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("env.time_steps must be a non-negative integer")
    return value


def _action_id(env, name: str) -> int:
    action_ids = getattr(env, "ACTION_IDS", None)
    if not isinstance(action_ids, Mapping) or name not in action_ids:
        raise TypeError(f"environment has no authoritative {name} action")
    value = action_ids[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"environment {name} action id must be an integer")
    return value


def _is_live_clear_stop(env, cell: Cell) -> bool:
    rows, cols = env.rooms.shape
    row, column = cell
    if not (0 <= row < rows and 0 <= column < cols):
        return False
    if env.rooms[row][column] == "#":
        return False
    if cell in {
        tuple(env.pickup_cell),
        tuple(env.waiting_cell),
    }:
        return False
    return not any(
        not block.delivered and block.position == cell for block in env.blocks
    )


def _execute_injected_primitive(
    env,
    *,
    phase: str,
    action_name: str,
) -> InjectedPrimitiveTransition:
    before_position = _agent_position(env)
    before_time = _time_step(env)
    action = _action_id(env, action_name)
    _, reward, terminal, info = env.step(action)
    after_position = _agent_position(env)
    after_time = _time_step(env)
    if after_time != before_time + 1:
        raise RuntimeError("one injected env.step did not advance time_steps by one")
    reward = float(reward)
    if not math.isfinite(reward):
        raise ValueError("injected environment reward is not finite")
    if not isinstance(info, Mapping):
        raise TypeError("injected env.step info must be a mapping")
    return InjectedPrimitiveTransition(
        phase=phase,
        action_name=action_name,
        action_id=action,
        before_agent_position=before_position,
        after_agent_position=after_position,
        time_step_before=before_time,
        time_step_after=after_time,
        reward=reward,
        terminal=bool(terminal),
        info=dict(info),
    )


def _base_replan_count(option) -> Optional[int]:
    outcome = getattr(option, "last_outcome", None)
    if not isinstance(outcome, Mapping):
        return None
    value = outcome.get("replans")
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        return None
    return value


def execute_bounded_macro_realization(
    env,
    candidate: ViabilityActionCandidate,
    *,
    realization: BoundedMacroRealization,
    gamma: float,
    remaining_steps: int,
    evaluation: bool,
) -> BoundedMacroExecution:
    """Execute one SAFE candidate and one explicit live disturbance realization.

    ``remaining_steps`` is the total base-plus-injection horizon.  The requested
    one or two injection steps are reserved before invoking the base option, so
    this function never intentionally exceeds that horizon.  Invalid declared
    targets are rejected before the environment is mutated.  If the base option
    fails, truncates, ends the environment, or disagrees with its certified
    endpoint, no disturbance primitive is injected and the result fails closed.
    """

    _validate_recovery_candidate(candidate)
    if not isinstance(realization, BoundedMacroRealization):
        raise TypeError("realization must be a BoundedMacroRealization")
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must satisfy 0 <= gamma < 1")
    if (
        isinstance(remaining_steps, bool)
        or not isinstance(remaining_steps, int)
        or remaining_steps <= realization.requested_injected_steps
    ):
        raise ValueError(
            "remaining_steps must leave at least one step for the base macro"
        )

    declared_stops = declared_clear_adjacent_stops(candidate)
    if (
        realization.adjacent_stop is not None
        and realization.adjacent_stop not in declared_stops
    ):
        raise ValueError(
            "adjacent_stop is not in the candidate's declared clear stop set"
        )

    start_position = _agent_position(env)
    start_time = _time_step(env)
    base = execute_certified_macro(
        env,
        candidate,
        gamma=gamma,
        remaining_steps=(
            remaining_steps - realization.requested_injected_steps
        ),
        evaluation=bool(evaluation),
    )
    base_position = _agent_position(env)
    base_time = _time_step(env)
    base_replans = _base_replan_count(candidate.option)

    transitions: list[InjectedPrimitiveTransition] = []
    failure: Optional[str] = None
    if base_time != start_time + base.duration:
        failure = "base_duration_disagrees_with_live_time_steps"
    elif not base.option_success:
        failure = f"base_macro_failed:{base.failure_reason or 'unknown'}"
    elif base_replans is None:
        failure = "base_macro_missing_replan_audit"
    elif base_replans != 0:
        failure = "base_macro_replanned_outside_fixed_path_contract"
    elif base_position != tuple(candidate.successor_state.agent_position):
        failure = "base_endpoint_disagrees_with_certified_successor"
    elif realization.requested_injected_steps and base.env_terminal:
        failure = "environment_terminal_before_disturbance_injection"
    elif any(block.carrying for block in env.blocks):
        failure = "base_macro_left_a_carried_block"

    if failure is None:
        for _ in range(realization.delay_steps):
            transition = _execute_injected_primitive(
                env,
                phase="delay",
                action_name="WAIT",
            )
            transitions.append(transition)
            if transition.after_agent_position != transition.before_agent_position:
                failure = "live_WAIT_changed_agent_position"
                break
            if transition.terminal and realization.adjacent_stop is not None:
                failure = "environment_terminal_before_adjacent_stop"
                break

    if failure is None and realization.adjacent_stop is not None:
        before = _agent_position(env)
        target = realization.adjacent_stop
        delta = (target[0] - before[0], target[1] - before[1])
        action_name = _CARDINAL_ACTION_FOR_DELTA.get(delta)
        if action_name is None:
            failure = "declared_stop_is_not_adjacent_to_live_base_endpoint"
        elif not _is_live_clear_stop(env, target):
            failure = "declared_stop_is_not_clear_in_live_environment"
        else:
            action = _action_id(env, action_name)
            if tuple(env._get_intended_cell(before, action)) != target:
                failure = "live_environment_rejects_declared_cardinal_stop"
            else:
                transition = _execute_injected_primitive(
                    env,
                    phase="adjacent_stop",
                    action_name=action_name,
                )
                transitions.append(transition)
                if transition.after_agent_position != target:
                    failure = "cardinal_stop_did_not_reach_declared_endpoint"

    injected_steps = len(transitions)
    injected_raw = sum(item.reward for item in transitions)
    injected_discounted = sum(
        (gamma ** (base.duration + index)) * item.reward
        for index, item in enumerate(transitions)
    )
    realized_position = _agent_position(env)
    realized_time = _time_step(env)
    terminal = bool(env.is_state_terminal(env.current_state))
    complete = bool(
        failure is None
        and base.option_success
        and injected_steps == realization.requested_injected_steps
    )
    return BoundedMacroExecution(
        candidate_key=candidate.key,
        action_type=candidate.action_type.value,
        realization=realization,
        declared_adjacent_stops=declared_stops,
        base_execution=base,
        base_replan_count=base_replans,
        base_primitive_steps=base.duration,
        injected_primitive_steps=injected_steps,
        total_primitive_steps=base.duration + injected_steps,
        start_agent_position=start_position,
        base_endpoint=base_position,
        realized_endpoint=realized_position,
        start_time_step=start_time,
        base_end_time_step=base_time,
        realized_end_time_step=realized_time,
        injected_transitions=tuple(transitions),
        injected_raw_return=float(injected_raw),
        injected_discounted_return_from_macro_start=float(
            injected_discounted
        ),
        total_raw_return=float(base.raw_return + injected_raw),
        total_discounted_return=float(
            base.discounted_return + injected_discounted
        ),
        env_terminal=terminal,
        realization_complete=complete,
        failure_reason=failure,
    )


__all__ = [
    "BoundedMacroExecution",
    "BoundedMacroRealization",
    "InjectedPrimitiveTransition",
    "LIVE_BOUNDED_MACRO_EXECUTION_CONTRACT",
    "declared_clear_adjacent_stops",
    "execute_bounded_macro_realization",
]
