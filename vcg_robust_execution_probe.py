"""Finite bounded-execution mechanism probe for VCG viability.

This module is deliberately additive.  It does not modify the authoritative
deterministic verifier or any trained controller.  Instead, it enumerates a
small declared set of execution realizations and admits a candidate only when
every primitive trajectory stays inside its declared tube and every realized
successor is certified recoverable by :mod:`PSLAP.viability`.

The result is a one-step, finite-set containment probe.  It is not a recursive
robust viability kernel and it is not real-world validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from enum import Enum
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from example.yard_geometry import make_shipyard_env
from PSLAP.dynamic_yard import BlockView, YardSnapshot
from PSLAP.viability import (
    RecoveryAction,
    RecoveryActionKind,
    RecoveryState,
    ViabilityStatus,
    analyze_recoverability,
    apply_recovery_action,
    legal_recovery_actions,
)
from PSLAP.viability_candidates import (
    ViabilityActionCandidate,
    ViabilityActionType,
    enumerate_viability_candidates,
)
from PSLAP.viability_filter import ViabilitySearchConfig


Cell = tuple[int, int]

PROTOCOL = "vcg_strict_recovery_macro_bounded_execution_probe_v1"
CLAIM_SCOPE = "one_step_execution_set_containment_not_recursive_robust_kernel_v1"
UNCERTAINTY_CONTRACT = (
    "finite_predeclared_duration_endpoint_and_route_obstacle_outcomes_v1"
)


class RobustExecutionStatus(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ExecutionRealization:
    """One explicit member of a finite execution-outcome set."""

    disturbance_id: str
    successor_state: RecoveryState
    trajectory_cells: tuple[Cell, ...]
    realized_primitive_steps: int
    primitive_step_budget: Optional[int]
    transient_blocked_cells: frozenset[Cell] = frozenset()
    verifier_max_nodes: Optional[int] = None
    in_declared_set: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.disturbance_id, str) or not self.disturbance_id:
            raise ValueError("disturbance_id must be a nonempty string")
        if not isinstance(self.successor_state, RecoveryState):
            raise TypeError("successor_state must be a RecoveryState")
        trajectory = tuple(tuple(cell) for cell in self.trajectory_cells)
        if not trajectory:
            raise ValueError("trajectory_cells must be nonempty")
        object.__setattr__(self, "trajectory_cells", trajectory)
        object.__setattr__(
            self,
            "transient_blocked_cells",
            frozenset(tuple(cell) for cell in self.transient_blocked_cells),
        )
        for name, value in (
            ("realized_primitive_steps", self.realized_primitive_steps),
            ("primitive_step_budget", self.primitive_step_budget),
        ):
            if value is None and name == "primitive_step_budget":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.verifier_max_nodes is not None and (
            isinstance(self.verifier_max_nodes, bool)
            or not isinstance(self.verifier_max_nodes, int)
            or self.verifier_max_nodes <= 0
        ):
            raise ValueError("verifier_max_nodes must be a positive integer or None")
        if not isinstance(self.in_declared_set, bool):
            raise TypeError("in_declared_set must be bool")


@dataclass(frozen=True)
class RealizationAudit:
    disturbance_id: str
    in_declared_set: bool
    status: RobustExecutionStatus
    tube_safe: bool
    tube_violations: tuple[Cell, ...]
    trajectory_cells: tuple[Cell, ...]
    transient_blocked_cells: tuple[Cell, ...]
    realized_primitive_steps: int
    primitive_step_budget: Optional[int]
    remaining_primitive_steps: Optional[int]
    successor_status: ViabilityStatus
    successor_reason: str
    successor_agent_position: Cell

    def to_dict(self) -> dict:
        return {
            "disturbance_id": self.disturbance_id,
            "in_declared_set": self.in_declared_set,
            "status": self.status.value,
            "tube_safe": self.tube_safe,
            "tube_violations": [list(cell) for cell in self.tube_violations],
            "trajectory_cells": [list(cell) for cell in self.trajectory_cells],
            "transient_blocked_cells": [
                list(cell) for cell in self.transient_blocked_cells
            ],
            "realized_primitive_steps": self.realized_primitive_steps,
            "primitive_step_budget": self.primitive_step_budget,
            "remaining_primitive_steps": self.remaining_primitive_steps,
            "successor_status": self.successor_status.value,
            "successor_reason": self.successor_reason,
            "successor_agent_position": list(self.successor_agent_position),
        }


@dataclass(frozen=True)
class RobustExecutionCertificate:
    action_key: str
    nominal_disturbance_id: str
    status: RobustExecutionStatus
    reason: str
    in_set_realizations: tuple[RealizationAudit, ...]
    out_of_set_observations: tuple[RealizationAudit, ...] = ()

    @property
    def admitted(self) -> bool:
        return self.status is RobustExecutionStatus.SAFE

    @property
    def nominal_audit(self) -> RealizationAudit:
        return next(
            item
            for item in self.in_set_realizations
            if item.disturbance_id == self.nominal_disturbance_id
        )

    def to_dict(self) -> dict:
        return {
            "action_key": self.action_key,
            "nominal_disturbance_id": self.nominal_disturbance_id,
            "nominal_status": self.nominal_audit.status.value,
            "robust_status": self.status.value,
            "admitted": self.admitted,
            "reason": self.reason,
            "declared_realization_count": len(self.in_set_realizations),
            "in_set_realizations": [
                item.to_dict() for item in self.in_set_realizations
            ],
            "out_of_set_observations": [
                {
                    **item.to_dict(),
                    "covered_by_guarantee": False,
                    "assumption_breach": True,
                }
                for item in self.out_of_set_observations
            ],
        }


@dataclass(frozen=True)
class PreferenceCandidate:
    key: str
    certificate: RobustExecutionCertificate
    q_operational: float
    q_rehandles: float

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("preference candidate key must be nonempty")
        for value in (self.q_operational, self.q_rehandles):
            if not math.isfinite(float(value)):
                raise ValueError("preference values must be finite")


def _audit_realization(realization: ExecutionRealization) -> RealizationAudit:
    violations = tuple(
        cell
        for cell in realization.trajectory_cells
        if (
            cell not in realization.successor_state.traversable
            or cell in realization.transient_blocked_cells
        )
    )
    tube_safe = not violations
    remaining = None
    if realization.primitive_step_budget is not None:
        remaining = (
            realization.primitive_step_budget
            - realization.realized_primitive_steps
        )

    if remaining is not None and remaining < 0:
        successor_status = ViabilityStatus.UNSAFE
        successor_reason = "unsafe:declared_primitive_horizon_exhausted"
    else:
        successor = analyze_recoverability(
            realization.successor_state,
            max_nodes=realization.verifier_max_nodes,
            max_primitive_steps=remaining,
            search_order="breadth_first",
        )
        successor_status = successor.status
        successor_reason = successor.reason

    if not tube_safe or successor_status is ViabilityStatus.UNSAFE:
        status = RobustExecutionStatus.UNSAFE
    elif successor_status is ViabilityStatus.UNKNOWN:
        status = RobustExecutionStatus.UNKNOWN
    else:
        status = RobustExecutionStatus.SAFE

    return RealizationAudit(
        disturbance_id=realization.disturbance_id,
        in_declared_set=realization.in_declared_set,
        status=status,
        tube_safe=tube_safe,
        tube_violations=violations,
        trajectory_cells=realization.trajectory_cells,
        transient_blocked_cells=tuple(
            sorted(realization.transient_blocked_cells)
        ),
        realized_primitive_steps=realization.realized_primitive_steps,
        primitive_step_budget=realization.primitive_step_budget,
        remaining_primitive_steps=remaining,
        successor_status=successor_status,
        successor_reason=successor_reason,
        successor_agent_position=realization.successor_state.agent_position,
    )


def certify_execution_envelope(
    action_key: str,
    realizations: Sequence[ExecutionRealization],
    *,
    nominal_disturbance_id: str,
    out_of_set_observations: Sequence[ExecutionRealization] = (),
) -> RobustExecutionCertificate:
    """Certify complete finite-set tube and successor containment.

    ``UNKNOWN`` is fail-closed: it is not admitted.  Out-of-set observations
    are audited and labeled, but do not change the conditional certificate.
    """

    if not isinstance(action_key, str) or not action_key:
        raise ValueError("action_key must be a nonempty string")
    in_set = tuple(realizations)
    outside = tuple(out_of_set_observations)
    if not in_set:
        raise ValueError("at least one declared execution realization is required")
    if any(not item.in_declared_set for item in in_set):
        raise ValueError("declared realizations must set in_declared_set=True")
    if any(item.in_declared_set for item in outside):
        raise ValueError("out-of-set observations must set in_declared_set=False")
    identifiers = [item.disturbance_id for item in in_set + outside]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("disturbance identifiers must be unique")
    if nominal_disturbance_id not in {
        item.disturbance_id for item in in_set
    }:
        raise ValueError("nominal disturbance must belong to the declared set")

    audits = tuple(_audit_realization(item) for item in in_set)
    outside_audits = tuple(_audit_realization(item) for item in outside)
    if any(item.status is RobustExecutionStatus.UNSAFE for item in audits):
        status = RobustExecutionStatus.UNSAFE
        reason = "rejected:at_least_one_declared_tube_or_successor_is_unsafe"
    elif any(item.status is RobustExecutionStatus.UNKNOWN for item in audits):
        status = RobustExecutionStatus.UNKNOWN
        reason = "rejected:at_least_one_declared_successor_is_unknown"
    else:
        status = RobustExecutionStatus.SAFE
        reason = "safe:all_declared_tubes_and_successors_certified"
    return RobustExecutionCertificate(
        action_key=action_key,
        nominal_disturbance_id=nominal_disturbance_id,
        status=status,
        reason=reason,
        in_set_realizations=audits,
        out_of_set_observations=outside_audits,
    )


def robust_safe_frontier(
    candidates: Iterable[PreferenceCandidate],
) -> tuple[PreferenceCandidate, ...]:
    return tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if candidate.certificate.admitted
            ),
            key=lambda item: item.key,
        )
    )


def select_preferred_robust(
    candidates: Iterable[PreferenceCandidate],
    *,
    handling_lambda: float,
) -> PreferenceCandidate:
    handling_lambda = float(handling_lambda)
    if not math.isfinite(handling_lambda) or handling_lambda < 0.0:
        raise ValueError("handling_lambda must be finite and non-negative")
    frontier = robust_safe_frontier(candidates)
    if not frontier:
        raise RuntimeError("no robustly certified candidate is available")
    return min(
        frontier,
        key=lambda item: (
            -(
                float(item.q_operational)
                - handling_lambda * float(item.q_rehandles)
            ),
            item.key,
        ),
    )


def _yard(
    traversable: Iterable[Cell],
    storage_cells: Iterable[Cell],
    exits: Iterable[Cell],
    blocks: Iterable[BlockView],
) -> YardSnapshot:
    cells = frozenset(tuple(cell) for cell in traversable)
    return YardSnapshot(
        rows=max(row for row, _ in cells) + 1,
        cols=max(col for _, col in cells) + 1,
        traversable=cells,
        storage_cells=frozenset(tuple(cell) for cell in storage_cells),
        exits=tuple(tuple(cell) for cell in exits),
        blocks=tuple(blocks),
    )


def _direct_state() -> RecoveryState:
    yard = _yard(
        ((0, 0), (0, 1), (0, 2)),
        ((0, 1),),
        ((0, 2),),
        (BlockView("A", (0, 1), 10.0),),
    )
    return RecoveryState.from_yard_snapshot(yard, (0, 0))


def _relocation_state() -> RecoveryState:
    yard = _yard(
        ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1)),
        ((0, 1), (0, 2), (1, 1)),
        ((0, 3),),
        (
            BlockView("A", (0, 1), 20.0),
            BlockView("B", (0, 2), 10.0),
        ),
    )
    return RecoveryState.from_yard_snapshot(yard, (0, 0))


def _only_action(
    state: RecoveryState,
    kind: RecoveryActionKind,
) -> RecoveryAction:
    actions = tuple(
        action for action in legal_recovery_actions(state) if action.kind is kind
    )
    if len(actions) != 1:
        raise RuntimeError(
            f"toy fixture expected one {kind.value} action, got {len(actions)}"
        )
    return actions[0]


def _action_trajectory(action: RecoveryAction) -> tuple[Cell, ...]:
    return action.approach_path + action.transport_path[1:]


def _duration_realization(
    state: RecoveryState,
    action: RecoveryAction,
    *,
    disturbance_id: str,
    extra_steps: int,
    total_budget: int,
    transient_blocked_cells: Iterable[Cell] = (),
) -> ExecutionRealization:
    return ExecutionRealization(
        disturbance_id=disturbance_id,
        successor_state=apply_recovery_action(state, action),
        trajectory_cells=_action_trajectory(action),
        realized_primitive_steps=action.steps + int(extra_steps),
        primitive_step_budget=int(total_budget),
        transient_blocked_cells=frozenset(transient_blocked_cells),
    )


def _make_stored(block, cell: Cell) -> None:
    block.position = tuple(cell)
    block.storage_location = tuple(cell)
    block.carrying = False
    block.stored = True
    block.delivered = False
    block.stored_time_step = 0
    block.storage_steps_elapsed = 0


def _live_accept_candidates(
    *,
    number_blocks: int,
    occupied: Sequence[Cell],
    instance_seed: int,
) -> dict[Cell, ViabilityActionCandidate]:
    """Return exact-SAFE Accept candidates from the real 5x5 frontier."""

    if len(occupied) != number_blocks - 1:
        raise ValueError("the live fixture requires one inbound block")
    env = make_shipyard_env(
        arrival_rate=10.0,
        proc_mean=80.0,
        grid_rows=5,
        grid_cols=5,
        exit_width=1,
        number_blocks=number_blocks,
    )
    env.reset(instance=env.sample_episode_instance(instance_seed))
    for block, cell in zip(env.blocks[:-1], occupied):
        _make_stored(block, cell)
    inbound = env.blocks[-1]
    inbound.position = tuple(env.pickup_cell)
    inbound.storage_location = None
    inbound.carrying = False
    inbound.stored = False
    inbound.delivered = False
    inbound.storage_steps_elapsed = 0
    env.current_state = tuple(env.pickup_cell)
    env.time_steps = 1
    snapshot = enumerate_viability_candidates(
        env,
        consecutive_defer_decisions=0,
        search_config=ViabilitySearchConfig(
            max_nodes=None,
            search_order="breadth_first",
        ),
    )
    return {
        tuple(candidate.destination): candidate
        for candidate in snapshot.candidates
        if candidate.action_type is ViabilityActionType.ACCEPT
        and candidate.destination is not None
    }


def _endpoint_set(
    candidate: ViabilityActionCandidate,
    endpoints: Sequence[Cell],
    *,
    out_of_set: Sequence[Cell] = (),
    verifier_max_nodes: Optional[int] = None,
) -> RobustExecutionCertificate:
    if candidate.action_type is not ViabilityActionType.ACCEPT:
        raise ValueError("endpoint probe requires an exact-SAFE Accept candidate")
    nominal = tuple(candidate.successor_state.agent_position)

    def realization(endpoint: Cell, *, inside: bool) -> ExecutionRealization:
        endpoint = tuple(endpoint)
        successor = replace(
            candidate.successor_state,
            agent_position=endpoint,
            fixed_obstacles=candidate.successor_state.fixed_obstacles - {endpoint},
        )
        return ExecutionRealization(
            disturbance_id=(
                f"endpoint:{endpoint[0]}:{endpoint[1]}"
                if inside
                else f"out_of_set_endpoint:{endpoint[0]}:{endpoint[1]}"
            ),
            successor_state=successor,
            trajectory_cells=(endpoint,),
            realized_primitive_steps=0,
            primitive_step_budget=None,
            verifier_max_nodes=verifier_max_nodes,
            in_declared_set=inside,
        )

    return certify_execution_envelope(
        candidate.key,
        tuple(realization(endpoint, inside=True) for endpoint in endpoints),
        nominal_disturbance_id=f"endpoint:{nominal[0]}:{nominal[1]}",
        out_of_set_observations=tuple(
            realization(endpoint, inside=False) for endpoint in out_of_set
        ),
    )


def build_probe_report() -> dict:
    relocation_state = _relocation_state()
    relocation = _only_action(relocation_state, RecoveryActionKind.RELOCATION)
    relocation_successor = apply_recovery_action(relocation_state, relocation)
    recovery = analyze_recoverability(
        relocation_successor,
        max_nodes=None,
        search_order="breadth_first",
    )
    if recovery.witness_primitive_steps is None:
        raise RuntimeError("relocation fixture must have an exact recovery witness")
    tight_budget = relocation.steps + recovery.witness_primitive_steps
    nominal_duration = certify_execution_envelope(
        "relocate:A:1:1",
        (
            _duration_realization(
                relocation_state,
                relocation,
                disturbance_id="delay:0",
                extra_steps=0,
                total_budget=tight_budget,
            ),
        ),
        nominal_disturbance_id="delay:0",
    )
    robust_duration = certify_execution_envelope(
        "relocate:A:1:1",
        tuple(
            _duration_realization(
                relocation_state,
                relocation,
                disturbance_id=f"delay:{delay}",
                extra_steps=delay,
                total_budget=tight_budget,
            )
            for delay in (0, 1)
        ),
        nominal_disturbance_id="delay:0",
    )

    direct_state = _direct_state()
    direct = _only_action(direct_state, RecoveryActionKind.DELIVERY)
    duration_slack = certify_execution_envelope(
        "deliver:A:0:2",
        tuple(
            _duration_realization(
                direct_state,
                direct,
                disturbance_id=f"delay:{delay}",
                extra_steps=delay,
                total_budget=direct.steps + 2,
            )
            for delay in (0, 2)
        ),
        nominal_disturbance_id="delay:0",
    )
    tube = certify_execution_envelope(
        "deliver:A:0:2",
        (
            _duration_realization(
                direct_state,
                direct,
                disturbance_id="route:clear",
                extra_steps=0,
                total_budget=direct.steps + 2,
            ),
            _duration_realization(
                direct_state,
                direct,
                disturbance_id="route:transient_obstruction",
                extra_steps=0,
                total_budget=direct.steps + 2,
                transient_blocked_cells=((0, 1),),
            ),
        ),
        nominal_disturbance_id="route:clear",
    )

    boundary_accepts = _live_accept_candidates(
        number_blocks=5,
        occupied=((1, 1), (1, 2), (2, 2), (3, 3)),
        instance_seed=908_006,
    )
    boundary_accept = boundary_accepts[(2, 3)]
    endpoint_nominal = _endpoint_set(boundary_accept, ((2, 3),))
    endpoint_boundary = _endpoint_set(
        boundary_accept, ((2, 3), (1, 3), (2, 2), (3, 3))
    )
    endpoint_declared_safe = _endpoint_set(
        boundary_accept,
        ((2, 3), (2, 2), (3, 3)),
        out_of_set=((1, 3),),
    )

    empty_accepts = _live_accept_candidates(
        number_blocks=1,
        occupied=(),
        instance_seed=908_002,
    )
    endpoint_robust_a = _endpoint_set(
        empty_accepts[(2, 2)],
        ((2, 2), (1, 2), (2, 1), (3, 2)),
    )
    endpoint_robust_b = _endpoint_set(
        empty_accepts[(3, 2)],
        ((3, 2), (2, 2), (3, 1), (3, 3)),
    )

    two_block_accepts = _live_accept_candidates(
        number_blocks=2,
        occupied=((1, 1),),
        instance_seed=908_003,
    )
    unknown = _endpoint_set(
        two_block_accepts[(1, 2)], ((1, 2),), verifier_max_nodes=1
    )

    # Both candidates come from the same live, empty-yard Accept frontier.
    # The values are intentionally hand-set to test only the separation
    # between the robust mask and the preference layer; they are not model
    # predictions.
    candidates = (
        PreferenceCandidate(
            "fast_robust",
            endpoint_robust_a,
            q_operational=10.0,
            q_rehandles=5.0,
        ),
        PreferenceCandidate(
            "careful_robust",
            endpoint_robust_b,
            q_operational=8.0,
            q_rehandles=0.0,
        ),
    )
    frontier = robust_safe_frontier(candidates)
    selected = {
        str(value): select_preferred_robust(
            candidates, handling_lambda=value
        ).key
        for value in (0.0, 1.0)
    }

    checks = {
        "nominal_only_failure_exposed": (
            nominal_duration.admitted and not robust_duration.admitted
        ),
        "bounded_slack_action_admitted": duration_slack.admitted,
        "tube_violation_rejected": (
            not tube.admitted
            and all(
                item.successor_status is ViabilityStatus.SAFE
                for item in tube.in_set_realizations
            )
        ),
        "endpoint_nominal_only_failure_exposed": (
            endpoint_nominal.admitted and not endpoint_boundary.admitted
        ),
        "unknown_fails_closed": (
            unknown.status is RobustExecutionStatus.UNKNOWN
            and not unknown.admitted
        ),
        "preference_changes_within_same_robust_frontier": (
            selected["0.0"] == "fast_robust"
            and selected["1.0"] == "careful_robust"
            and {item.key for item in frontier}
            == {"fast_robust", "careful_robust"}
        ),
        "out_of_set_observation_labeled_not_certified": (
            endpoint_declared_safe.admitted
            and len(endpoint_declared_safe.out_of_set_observations) == 1
            and endpoint_declared_safe.out_of_set_observations[0].status
            is RobustExecutionStatus.UNSAFE
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"robust execution probe failed: {checks!r}")

    return {
        "protocol": PROTOCOL,
        "claim_scope": CLAIM_SCOPE,
        "uncertainty_contract": UNCERTAINTY_CONTRACT,
        "current_admitted_work_closed_admission": True,
        "complete_episode_certificate": False,
        "recursive_robust_kernel_constructed": False,
        "real_world_validation": False,
        "training_or_learning": False,
        "cases": {
            "duration_boundary_nominal": nominal_duration.to_dict(),
            "duration_boundary_bounded_set": robust_duration.to_dict(),
            "duration_slack_bounded_set": duration_slack.to_dict(),
            "trajectory_tube_obstruction": tube.to_dict(),
            "endpoint_nominal": endpoint_nominal.to_dict(),
            "endpoint_radius_one_boundary": endpoint_boundary.to_dict(),
            "endpoint_declared_safe_with_out_of_set_failure": (
                endpoint_declared_safe.to_dict()
            ),
            "endpoint_radius_one_robust_a": endpoint_robust_a.to_dict(),
            "endpoint_radius_one_robust_b": endpoint_robust_b.to_dict(),
            "unknown_fail_closed": unknown.to_dict(),
        },
        "preference_invariance": {
            "merit": "q_operational - handling_lambda * q_rehandles",
            "robust_frontier": [item.key for item in frontier],
            "rejected_before_ranking": [
                item.key for item in candidates if not item.certificate.admitted
            ],
            "selected_by_handling_lambda": selected,
        },
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
            "results/vcg-robust-execution-probe/robust-execution-report.json"
        ),
    )
    args = parser.parse_args()
    report = build_probe_report()
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
