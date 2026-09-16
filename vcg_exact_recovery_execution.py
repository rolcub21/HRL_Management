"""Execute a certified RecoveryAction path literally through the live env.

The historical recovery options are intentionally allowed to recompute A*
paths.  That behavior is useful operationally but is not sufficient when a
theorem-facing dynamic experiment must execute the exact transition certified
by the finite recovery model.  This additive module therefore emits the bound
approach cells, PICKUP, bound transport cells, and PUTDOWN directly through the
authoritative ``env.step`` API.  It never invokes a planner and never replans.

The optional post-macro delay and adjacent-stop realization is the same fixed
member used by the v1 live-disturbance adapter.  It is not an adversarial or
exhaustive realization of the certificate's complete uncertainty set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Optional

from PSLAP.dynamic_yard import Cell
from PSLAP.viability import RecoveryAction, RecoveryActionKind
from PSLAP.viability_candidates import (
    ViabilityActionCandidate,
    ViabilityActionType,
)
from train_viability_graph_smdp import MacroExecution
import vcg_bounded_macro_execution as bounded


EXACT_RECOVERY_EXECUTION_CONTRACT = (
    "certified_RecoveryAction_cells_pickup_transport_putdown_via_live_env_step_"
    "without_planner_or_replanning_then_fixed_bounded_realization_v2"
)

_MOVEMENT_NAME = {
    (-1, 0): "UP",
    (1, 0): "DOWN",
    (0, -1): "LEFT",
    (0, 1): "RIGHT",
}


class ExactRecoveryExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CertifiedPrimitive:
    phase: str
    action_name: str
    before_cell: Cell
    after_cell: Cell


@dataclass(frozen=True)
class ExactPrimitiveTransition:
    primitive_index: int
    phase: str
    action_name: str
    action_id: int
    expected_before_cell: Cell
    expected_after_cell: Cell
    actual_before_cell: Cell
    actual_after_cell: Cell
    time_step_before: int
    time_step_after: int
    carried_label_before: Optional[str]
    carried_label_after: Optional[str]
    reward: float
    terminal: bool
    info: Mapping[str, object]


@dataclass(frozen=True)
class ExactPathMacroExecution:
    macro_execution: MacroExecution
    recovery_action: RecoveryAction
    certified_primitives: tuple[CertifiedPrimitive, ...]
    emitted_transitions: tuple[ExactPrimitiveTransition, ...]
    expected_action_names: tuple[str, ...]
    emitted_action_names: tuple[str, ...]
    expected_cell_trace: tuple[Cell, ...]
    observed_cell_trace: tuple[Cell, ...]
    trace_matches_certified_action: bool
    planner_calls: int = 0
    replan_count: int = 0


@dataclass(frozen=True)
class ExactBoundedRecoveryExecution:
    candidate_key: str
    action_type: str
    realization: bounded.BoundedMacroRealization
    declared_adjacent_stops: tuple[Cell, ...]
    base_execution: MacroExecution
    exact_base: ExactPathMacroExecution
    base_replan_count: int
    base_primitive_steps: int
    injected_primitive_steps: int
    total_primitive_steps: int
    start_agent_position: Cell
    base_endpoint: Cell
    realized_endpoint: Cell
    start_time_step: int
    base_end_time_step: int
    realized_end_time_step: int
    injected_transitions: tuple[bounded.InjectedPrimitiveTransition, ...]
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


def _carried_label(env) -> Optional[str]:
    labels = [str(block.label) for block in env.blocks if block.carrying]
    if len(labels) > 1:
        raise ExactRecoveryExecutionError("environment carries multiple blocks")
    return None if not labels else labels[0]


def _target_block(env, label: str):
    matches = [block for block in env.blocks if str(block.label) == str(label)]
    if len(matches) != 1:
        raise ExactRecoveryExecutionError(
            "certified target block is absent or non-unique"
        )
    return matches[0]


def _movement_primitive(before: Cell, after: Cell, phase: str) -> CertifiedPrimitive:
    delta = (after[0] - before[0], after[1] - before[1])
    name = _MOVEMENT_NAME.get(delta)
    if name is None:
        raise ExactRecoveryExecutionError(
            "certified path contains a non-cardinal or zero-length edge"
        )
    return CertifiedPrimitive(phase, name, before, after)


def certified_primitives(action: RecoveryAction) -> tuple[CertifiedPrimitive, ...]:
    if not isinstance(action, RecoveryAction):
        raise TypeError("action must be a RecoveryAction")
    if not action.approach_path or not action.transport_path:
        raise ExactRecoveryExecutionError("certified path leg is empty")
    if action.approach_path[-1] != action.source:
        raise ExactRecoveryExecutionError("approach does not end at source")
    if action.transport_path[0] != action.source:
        raise ExactRecoveryExecutionError("transport does not start at source")
    if action.transport_path[-1] != action.destination:
        raise ExactRecoveryExecutionError("transport does not end at destination")
    result = []
    result.extend(
        _movement_primitive(before, after, "approach_move")
        for before, after in zip(action.approach_path, action.approach_path[1:])
    )
    result.append(CertifiedPrimitive("pickup", "PICKUP", action.source, action.source))
    result.extend(
        _movement_primitive(before, after, "transport_move")
        for before, after in zip(action.transport_path, action.transport_path[1:])
    )
    result.append(
        CertifiedPrimitive(
            "putdown",
            "PUTDOWN",
            action.destination,
            action.destination,
        )
    )
    if len(result) != action.steps:
        raise ExactRecoveryExecutionError(
            "certified primitive expansion disagrees with RecoveryAction.steps"
        )
    return tuple(result)


def _validate_live_binding(
    env,
    candidate: ViabilityActionCandidate,
    action: RecoveryAction,
    primitives: tuple[CertifiedPrimitive, ...],
    *,
    remaining_steps: int,
) -> object:
    bounded._validate_recovery_candidate(candidate)
    if candidate.recovery_action != action:
        raise ExactRecoveryExecutionError(
            "candidate does not contain the supplied certified RecoveryAction"
        )
    expected_type = (
        ViabilityActionType.DELIVER
        if action.kind is RecoveryActionKind.DELIVERY
        else ViabilityActionType.RECONFIGURE
    )
    if candidate.action_type is not expected_type:
        raise ExactRecoveryExecutionError("candidate/action kind mismatch")
    if tuple(env.current_state) != tuple(action.approach_path[0]):
        raise ExactRecoveryExecutionError("live agent is not at certified path start")
    if action.steps > remaining_steps:
        raise ExactRecoveryExecutionError(
            "remaining live horizon cannot fit certified recovery path"
        )
    if _carried_label(env) is not None:
        raise ExactRecoveryExecutionError("recovery must start empty-handed")
    target = _target_block(env, action.block_label)
    if (
        not target.stored
        or target.delivered
        or target.carrying
        or tuple(target.position) != tuple(action.source)
        or tuple(target.storage_location) != tuple(action.source)
    ):
        raise ExactRecoveryExecutionError("live target is not bound at source")
    if action.kind is RecoveryActionKind.DELIVERY:
        exits = {tuple(cell) for cell in env.exit_states}
        if tuple(action.destination) not in exits:
            raise ExactRecoveryExecutionError("delivery destination is not a live exit")
    else:
        if tuple(action.destination) not in {
            tuple(cell) for cell in env.storage_positions
        }:
            raise ExactRecoveryExecutionError(
                "relocation destination is not a storage cell"
            )
        if tuple(action.destination) == tuple(action.source):
            raise ExactRecoveryExecutionError("relocation destination equals source")
    occupied_by_other = {
        tuple(block.position)
        for block in env.blocks
        if (
            str(block.label) != str(action.block_label)
            and not block.delivered
            and not block.carrying
            and block.position is not None
        )
    }
    for primitive in primitives:
        for cell in (primitive.before_cell, primitive.after_cell):
            row, column = cell
            if not (
                0 <= row < env.rooms.shape[0]
                and 0 <= column < env.rooms.shape[1]
                and env.rooms[row][column] != "#"
            ):
                raise ExactRecoveryExecutionError(
                    "certified primitive visits a non-traversable live cell"
                )
        if primitive.phase in ("approach_move", "transport_move"):
            if (
                primitive.after_cell in occupied_by_other
                and primitive.after_cell != tuple(action.destination)
            ):
                raise ExactRecoveryExecutionError(
                    "certified path is obstructed by other live inventory"
                )
    if tuple(action.destination) in occupied_by_other:
        raise ExactRecoveryExecutionError("certified destination is occupied")
    if tuple(candidate.successor_state.agent_position) != tuple(action.destination):
        raise ExactRecoveryExecutionError(
            "candidate successor endpoint differs from certified destination"
        )
    return target


def execute_exact_recovery_macro(
    env,
    candidate: ViabilityActionCandidate,
    *,
    gamma: float,
    remaining_steps: int,
    evaluation: bool,
) -> ExactPathMacroExecution:
    """Emit the certified primitive sequence exactly once through ``env.step``."""

    del evaluation  # The exact executor has no exploratory or test behavior.
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must satisfy 0 <= gamma < 1")
    if (
        isinstance(remaining_steps, bool)
        or not isinstance(remaining_steps, int)
        or remaining_steps <= 0
    ):
        raise ValueError("remaining_steps must be a positive integer")
    action = candidate.recovery_action
    if not isinstance(action, RecoveryAction):
        raise ExactRecoveryExecutionError("candidate lacks a RecoveryAction")
    primitives = certified_primitives(action)
    target = _validate_live_binding(
        env,
        candidate,
        action,
        primitives,
        remaining_steps=remaining_steps,
    )
    stored_time_step_before = target.stored_time_step
    expected_names = tuple(item.action_name for item in primitives)
    expected_cells = (primitives[0].before_cell,) + tuple(
        item.after_cell for item in primitives
    )
    emitted = []
    discounted = 0.0
    raw = 0.0
    deviations = []
    relocations = 0
    illegal_drops = 0
    failure = None

    for index, primitive in enumerate(primitives):
        before_cell = bounded._agent_position(env)
        before_time = bounded._time_step(env)
        carried_before = _carried_label(env)
        if before_cell != primitive.before_cell:
            failure = "live_cell_disagrees_before_certified_primitive"
            break
        if bool(env.is_state_terminal(env.current_state)):
            failure = "environment_terminal_before_certified_path_complete"
            break
        action_id = bounded._action_id(env, primitive.action_name)
        if primitive.phase in ("approach_move", "transport_move"):
            live_occupied_by_other = {
                tuple(block.position)
                for block in env.blocks
                if (
                    str(block.label) != str(action.block_label)
                    and not block.delivered
                    and not block.carrying
                    and block.position is not None
                )
            }
            if primitive.after_cell in live_occupied_by_other:
                failure = "certified_next_cell_became_occupied_before_movement"
                break
            intended = tuple(env._get_intended_cell(before_cell, action_id))
            if intended != primitive.after_cell:
                failure = "live_action_model_disagrees_with_certified_edge"
                break
        if primitive.phase == "pickup":
            if carried_before is not None or before_cell != tuple(action.source):
                failure = "certified_PICKUP_precondition_failed"
                break
        if primitive.phase == "putdown":
            if (
                carried_before != str(action.block_label)
                or before_cell != tuple(action.destination)
            ):
                failure = "certified_PUTDOWN_precondition_failed"
                break
            if action.kind is RecoveryActionKind.RELOCATION:
                # The live environment recognizes a stored-block relocation by
                # the newly bound storage_location at the consumed PUTDOWN.
                target.storage_location = tuple(action.destination)
        _, reward, terminal, info = env.step(action_id)
        after_cell = bounded._agent_position(env)
        after_time = bounded._time_step(env)
        carried_after = _carried_label(env)
        reward = float(reward)
        if not math.isfinite(reward):
            raise ExactRecoveryExecutionError("live reward is not finite")
        if not isinstance(info, Mapping):
            raise ExactRecoveryExecutionError("live env.step info is not a mapping")
        transition = ExactPrimitiveTransition(
            primitive_index=index,
            phase=primitive.phase,
            action_name=primitive.action_name,
            action_id=action_id,
            expected_before_cell=primitive.before_cell,
            expected_after_cell=primitive.after_cell,
            actual_before_cell=before_cell,
            actual_after_cell=after_cell,
            time_step_before=before_time,
            time_step_after=after_time,
            carried_label_before=carried_before,
            carried_label_after=carried_after,
            reward=reward,
            terminal=bool(terminal),
            info=dict(info),
        )
        emitted.append(transition)
        duration = len(emitted)
        discounted += (gamma ** (duration - 1)) * reward
        raw += reward
        if "delivery_error_time" in info:
            deviations.append(float(info["delivery_error_time"]))
        relocations += int(bool(info.get("relocated_block")))
        illegal_drops += int(bool(info.get("illegal_drop")))
        if after_time != before_time + 1:
            failure = "certified_env_step_did_not_advance_one_time_step"
            break
        if after_cell != primitive.after_cell:
            failure = "live_cell_disagrees_after_certified_primitive"
            break
        if primitive.phase == "pickup" and (
            info.get("picked_block") != action.block_label
            or carried_after != str(action.block_label)
        ):
            failure = "live_PICKUP_did_not_bind_certified_block"
            break
        if primitive.phase == "putdown":
            if action.kind is RecoveryActionKind.DELIVERY:
                if (
                    info.get("delivered_block") != action.block_label
                    or not target.delivered
                    or target.carrying
                ):
                    failure = "live_PUTDOWN_did_not_deliver_certified_block"
            elif (
                info.get("relocated_block") != action.block_label
                or target.delivered
                or target.carrying
                or tuple(target.position) != tuple(action.destination)
                or tuple(target.storage_location) != tuple(action.destination)
            ):
                failure = "live_PUTDOWN_did_not_relocate_certified_block"
        if terminal and index != len(primitives) - 1:
            failure = "environment_terminal_before_certified_path_complete"
            break
        if failure is not None:
            break

    emitted_names = tuple(item.action_name for item in emitted)
    observed_cells = (
        (primitives[0].before_cell,)
        if not emitted
        else (emitted[0].actual_before_cell,)
        + tuple(item.actual_after_cell for item in emitted)
    )
    trace_matches = bool(
        failure is None
        and len(emitted) == len(primitives)
        and emitted_names == expected_names
        and observed_cells == expected_cells
        and all(
            transition.expected_before_cell == transition.actual_before_cell
            and transition.expected_after_cell == transition.actual_after_cell
            for transition in emitted
        )
    )
    if failure is None and not trace_matches:
        failure = "emitted_trace_disagrees_with_certified_action"
    if failure is None and target.stored_time_step != stored_time_step_before:
        failure = "stored_time_step_changed_during_exact_recovery"
    delivered_infos = [
        transition.info.get("delivered_block")
        for transition in emitted
        if transition.info.get("delivered_block")
    ]
    relocated_infos = [
        transition.info.get("relocated_block")
        for transition in emitted
        if transition.info.get("relocated_block")
    ]
    illegal_infos = [
        transition.info.get("illegal_drop")
        for transition in emitted
        if transition.info.get("illegal_drop")
    ]
    if failure is None and action.kind is RecoveryActionKind.DELIVERY:
        expected_deviation = int(target.delivered_time_step) - int(
            target.stored_time_step + target.storage_steps_needed
        )
        if (
            delivered_infos != [action.block_label]
            or relocated_infos
            or illegal_infos
            or relocations != 0
            or illegal_drops != 0
            or deviations != [float(expected_deviation)]
            or target.delivery_error_time != expected_deviation
        ):
            failure = "exact_delivery_metrics_or_outcome_disagree"
    if failure is None and action.kind is RecoveryActionKind.RELOCATION:
        if (
            relocated_infos != [action.block_label]
            or delivered_infos
            or illegal_infos
            or relocations != 1
            or illegal_drops != 0
            or deviations
        ):
            failure = "exact_relocation_metrics_or_outcome_disagree"
    terminal = bool(env.is_state_terminal(env.current_state))
    success = bool(failure is None and trace_matches)
    macro = MacroExecution(
        candidate_key=candidate.key,
        action_type=candidate.action_type.value,
        mode=candidate.mode.value,
        discounted_return=float(discounted),
        raw_return=float(raw),
        duration=len(emitted),
        env_terminal=terminal,
        truncated=False,
        option_terminated=success,
        option_success=success,
        failure_reason=failure,
        delivery_deviations=tuple(deviations),
        relocations=int(relocations),
        illegal_drops=int(illegal_drops),
    )
    return ExactPathMacroExecution(
        macro_execution=macro,
        recovery_action=action,
        certified_primitives=primitives,
        emitted_transitions=tuple(emitted),
        expected_action_names=expected_names,
        emitted_action_names=emitted_names,
        expected_cell_trace=expected_cells,
        observed_cell_trace=observed_cells,
        trace_matches_certified_action=trace_matches,
    )


def execute_exact_bounded_recovery_realization(
    env,
    candidate: ViabilityActionCandidate,
    *,
    realization: bounded.BoundedMacroRealization,
    gamma: float,
    remaining_steps: int,
    evaluation: bool,
) -> ExactBoundedRecoveryExecution:
    """Execute one exact certified path and the fixed live realization."""

    bounded._validate_recovery_candidate(candidate)
    if not isinstance(realization, bounded.BoundedMacroRealization):
        raise TypeError("realization must be a BoundedMacroRealization")
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must satisfy 0 <= gamma < 1")
    action = candidate.recovery_action
    required = int(action.steps) + realization.requested_injected_steps
    if (
        isinstance(remaining_steps, bool)
        or not isinstance(remaining_steps, int)
        or remaining_steps < required
    ):
        raise ValueError("remaining_steps cannot fit exact path and realization")
    declared_stops = bounded.declared_clear_adjacent_stops(candidate)
    if (
        realization.adjacent_stop is not None
        and realization.adjacent_stop not in declared_stops
    ):
        raise ValueError("adjacent_stop is outside declared clear stop set")

    start_position = bounded._agent_position(env)
    start_time = bounded._time_step(env)
    exact = execute_exact_recovery_macro(
        env,
        candidate,
        gamma=gamma,
        remaining_steps=remaining_steps - realization.requested_injected_steps,
        evaluation=evaluation,
    )
    base = exact.macro_execution
    base_position = bounded._agent_position(env)
    base_time = bounded._time_step(env)
    transitions = []
    failure = None
    if base_time != start_time + base.duration:
        failure = "base_duration_disagrees_with_live_time_steps"
    elif not base.option_success:
        failure = f"exact_base_macro_failed:{base.failure_reason or 'unknown'}"
    elif not exact.trace_matches_certified_action:
        failure = "exact_base_trace_disagrees_with_certified_action"
    elif base.duration != action.steps:
        failure = "exact_base_duration_disagrees_with_RecoveryAction.steps"
    elif base_position != tuple(candidate.successor_state.agent_position):
        failure = "exact_base_endpoint_disagrees_with_certified_successor"
    elif realization.requested_injected_steps and base.env_terminal:
        failure = "environment_terminal_before_disturbance_injection"
    elif _carried_label(env) is not None:
        failure = "exact_base_macro_left_a_carried_block"

    if failure is None:
        for _ in range(realization.delay_steps):
            transition = bounded._execute_injected_primitive(
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
        before = bounded._agent_position(env)
        target = realization.adjacent_stop
        delta = (target[0] - before[0], target[1] - before[1])
        action_name = bounded._CARDINAL_ACTION_FOR_DELTA.get(delta)
        if action_name is None:
            failure = "declared_stop_is_not_adjacent_to_live_base_endpoint"
        elif not bounded._is_live_clear_stop(env, target):
            failure = "declared_stop_is_not_clear_in_live_environment"
        else:
            action_id = bounded._action_id(env, action_name)
            if tuple(env._get_intended_cell(before, action_id)) != target:
                failure = "live_environment_rejects_declared_cardinal_stop"
            else:
                transition = bounded._execute_injected_primitive(
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
    realized_position = bounded._agent_position(env)
    realized_time = bounded._time_step(env)
    terminal = bool(env.is_state_terminal(env.current_state))
    complete = bool(
        failure is None
        and base.option_success
        and exact.trace_matches_certified_action
        and injected_steps == realization.requested_injected_steps
    )
    return ExactBoundedRecoveryExecution(
        candidate_key=candidate.key,
        action_type=candidate.action_type.value,
        realization=realization,
        declared_adjacent_stops=declared_stops,
        base_execution=base,
        exact_base=exact,
        base_replan_count=0,
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
        injected_discounted_return_from_macro_start=float(injected_discounted),
        total_raw_return=float(base.raw_return + injected_raw),
        total_discounted_return=float(
            base.discounted_return + injected_discounted
        ),
        env_terminal=terminal,
        realization_complete=complete,
        failure_reason=failure,
    )


def exact_execution_to_dict(execution: ExactBoundedRecoveryExecution) -> dict:
    exact = execution.exact_base
    return {
        "execution_contract": EXACT_RECOVERY_EXECUTION_CONTRACT,
        "candidate_key": execution.candidate_key,
        "action_type": execution.action_type,
        "realization": asdict(execution.realization),
        "realization_id": execution.realization.realization_id,
        "declared_adjacent_stops": list(execution.declared_adjacent_stops),
        "base_replan_count": execution.base_replan_count,
        "base_primitive_steps": execution.base_primitive_steps,
        "injected_primitive_steps": execution.injected_primitive_steps,
        "total_primitive_steps": execution.total_primitive_steps,
        "start_agent_position": execution.start_agent_position,
        "base_endpoint": execution.base_endpoint,
        "realized_endpoint": execution.realized_endpoint,
        "start_time_step": execution.start_time_step,
        "base_end_time_step": execution.base_end_time_step,
        "realized_end_time_step": execution.realized_end_time_step,
        "injected_transitions": [asdict(item) for item in execution.injected_transitions],
        "injected_raw_return": execution.injected_raw_return,
        "injected_discounted_return_from_macro_start": (
            execution.injected_discounted_return_from_macro_start
        ),
        "total_raw_return": execution.total_raw_return,
        "total_discounted_return": execution.total_discounted_return,
        "env_terminal": execution.env_terminal,
        "realization_complete": execution.realization_complete,
        "failure_reason": execution.failure_reason,
        "exact_path": {
            "recovery_action": {
                "kind": exact.recovery_action.kind.value,
                "block_label": exact.recovery_action.block_label,
                "source": exact.recovery_action.source,
                "destination": exact.recovery_action.destination,
                "approach_path": exact.recovery_action.approach_path,
                "transport_path": exact.recovery_action.transport_path,
                "steps": exact.recovery_action.steps,
            },
            "certified_primitives": [asdict(item) for item in exact.certified_primitives],
            "emitted_transitions": [asdict(item) for item in exact.emitted_transitions],
            "expected_action_names": exact.expected_action_names,
            "emitted_action_names": exact.emitted_action_names,
            "expected_cell_trace": exact.expected_cell_trace,
            "observed_cell_trace": exact.observed_cell_trace,
            "trace_matches_certified_action": exact.trace_matches_certified_action,
            "planner_calls": exact.planner_calls,
            "replan_count": exact.replan_count,
        },
        "base": {
            "duration": execution.base_execution.duration,
            "option_success": execution.base_execution.option_success,
            "failure_reason": execution.base_execution.failure_reason,
            "delivery_deviations": list(
                execution.base_execution.delivery_deviations
            ),
            "relocations": execution.base_execution.relocations,
            "illegal_drops": execution.base_execution.illegal_drops,
        },
    }


__all__ = [
    "EXACT_RECOVERY_EXECUTION_CONTRACT",
    "CertifiedPrimitive",
    "ExactBoundedRecoveryExecution",
    "ExactPathMacroExecution",
    "ExactPrimitiveTransition",
    "ExactRecoveryExecutionError",
    "certified_primitives",
    "exact_execution_to_dict",
    "execute_exact_bounded_recovery_realization",
    "execute_exact_recovery_macro",
]
