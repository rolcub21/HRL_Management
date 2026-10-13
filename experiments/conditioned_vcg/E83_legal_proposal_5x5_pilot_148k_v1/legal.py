"""Proof-free *proposal* frontier. Only selected actions may be executed after proof.

This module deliberately does not construct ViabilityActionCandidate objects or
claim that physical legality implies recoverability. The common rank and guard
features are masked in both pilot arms, since they are unavailable before proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from example.Options.DirectDeliverOption import DirectDeliverOption
from example.Options.ExplicitAcceptOption import ExplicitAcceptOption
from example.Options.ExplicitDeferOption import ExplicitDeferOption
from example.Options.ReconfigureOption import ReconfigureOption
from PSLAP.dynamic_yard import YardSnapshot
from PSLAP.neutral_protocol import shared_candidate_mask
from PSLAP.viability import (
    RecoveryAction, RecoveryActionKind, RecoveryState, apply_recovery_action,
    legal_recovery_actions,
)
from PSLAP.viability_candidates import (
    BoundedEventDeferRule, ViabilityActionCandidate, ViabilityActionType,
    ViabilityMode,
)
from PSLAP.viability_filter import (
    ViabilitySearchConfig, online_fixed_obstacles, post_accept_recovery_state,
)
from PSLAP import viability_candidates as vc
from viability_graph_hierarchy import (
    ACTION_FEATURE_DIM, ACTION_FEATURE_NAMES, MODE_TO_ID,
    ViabilityGraphCandidateRecord, candidate_action_features,
)


MASKED_NAMES = ACTION_FEATURE_NAMES[18:]
assert MASKED_NAMES == (
    "rank_before_known", "rank_before_per_storage_cell",
    "rank_after_known", "rank_after_per_storage_cell",
    "rank_delta_known", "rank_delta_per_storage_cell",
    "guard_active", "guard_nonprogress_fraction",
    "guard_remaining_witness_per_storage_cell", "guard_forced_frontier",
)


@dataclass(frozen=True)
class LegalProposal:
    key: str
    mode: ViabilityMode
    action_type: ViabilityActionType
    target_label: Optional[str]
    source: Optional[tuple[int, int]]
    destination: Optional[tuple[int, int]]
    successor_state: RecoveryState
    option: object
    recovery_action: Optional[RecoveryAction] = None
    horizon_steps: Optional[int] = None


def masked_certified_record(candidate: ViabilityActionCandidate,
                            current_state: RecoveryState) -> ViabilityGraphCandidateRecord:
    features = candidate_action_features(candidate)
    return ViabilityGraphCandidateRecord(
        candidate.key, MODE_TO_ID[candidate.mode.value],
        candidate.action_type.value, current_state, candidate.successor_state,
        features[:18] + (0.0,) * (ACTION_FEATURE_DIM - 18),
    )


def legal_record(proposal: LegalProposal,
                 current_state: RecoveryState) -> ViabilityGraphCandidateRecord:
    successor = proposal.successor_state
    cell_count = max(successor.rows * successor.cols, 1)
    source = proposal.source
    destination = proposal.destination
    steps = (proposal.recovery_action.steps if proposal.recovery_action is not None
             else proposal.horizon_steps)
    features = (
        *(float(proposal.mode.value == mode) for mode in ("accept", "recover", "defer")),
        *(float(proposal.action_type is kind) for kind in ViabilityActionType),
        float(proposal.target_label is not None),
        float(source is not None),
        0.0 if source is None else source[0] / max(successor.rows - 1, 1),
        0.0 if source is None else source[1] / max(successor.cols - 1, 1),
        float(destination is not None),
        0.0 if destination is None else destination[0] / max(successor.rows - 1, 1),
        0.0 if destination is None else destination[1] / max(successor.cols - 1, 1),
        float(steps is not None),
        0.0 if steps is None else float(steps) / cell_count,
        float(proposal.horizon_steps is not None),
        0.0 if proposal.horizon_steps is None else proposal.horizon_steps / cell_count,
        *((0.0,) * (ACTION_FEATURE_DIM - 18)),
    )
    assert len(features) == ACTION_FEATURE_DIM
    return ViabilityGraphCandidateRecord(
        proposal.key, MODE_TO_ID[proposal.mode.value],
        proposal.action_type.value, current_state, successor,
        tuple(float(value) for value in features),
    )


def physical_frontier(env, *, current_state: RecoveryState,
                      current_certificate, search: ViabilitySearchConfig,
                      liveness: BoundedEventDeferRule,
                      consecutive_defer: int) -> tuple[LegalProposal, ...]:
    """Enumerate executable macros without analyzing any successor."""
    vc._validate_strict_boundary(env)
    yard = YardSnapshot.from_env(env)
    inbound = vc._strict_inbound(env)
    proposals = []
    if inbound is not None:
        inbound_view = vc._inbound_view(env, inbound)
        for cell in shared_candidate_mask(
            yard, inbound_view, tuple(inbound.position), reserved_cells=(),
        ):
            option = ExplicitAcceptOption(env, inbound.label, cell, max_replans=8)
            if option.initiation(env.get_current_state()):
                proposals.append(LegalProposal(
                    f"accept:{inbound.label}:{cell[0]}:{cell[1]}",
                    ViabilityMode.ACCEPT, ViabilityActionType.ACCEPT,
                    str(inbound.label), tuple(inbound.position), tuple(cell),
                    post_accept_recovery_state(
                        env, yard, inbound_view, cell,
                        reserve_queue_cells=search.reserve_queue_cells,
                    ), option,
                ))
    for action in legal_recovery_actions(current_state):
        successor = apply_recovery_action(current_state, action)
        if action.kind is RecoveryActionKind.DELIVERY:
            kind = ViabilityActionType.DELIVER
            option = DirectDeliverOption.from_recovery_action(
                env, action, max_replans=8,
                fixed_obstacles=current_state.fixed_obstacles,
            )
        else:
            kind = ViabilityActionType.RECONFIGURE
            option = ReconfigureOption.from_recovery_action(
                env, action, max_replans=8,
                fixed_obstacles=current_state.fixed_obstacles,
            )
        if option.initiation(env.get_current_state()):
            proposals.append(LegalProposal(
                vc._recovery_candidate_key(action), ViabilityMode.RECOVER,
                kind, action.block_label, action.source, action.destination,
                successor, option, recovery_action=action,
            ))
    defer = liveness.evaluate(
        env, yard, inbound_present=inbound is not None,
        current_certificate=current_certificate,
        consecutive_defer_decisions=consecutive_defer,
        terminal=bool(env.is_state_terminal(env.current_state)),
    )
    if defer.allowed:
        option = ExplicitDeferOption(env, horizon_steps=defer.horizon_steps)
        if option.initiation(env.get_current_state()):
            proposals.append(LegalProposal(
                f"defer:{defer.horizon_steps}", ViabilityMode.DEFER,
                ViabilityActionType.DEFER, None, tuple(env.current_state), None,
                current_state, option, horizon_steps=defer.horizon_steps,
            ))
    return tuple(proposals)


def current_state_from_env(env, search: ViabilitySearchConfig) -> RecoveryState:
    yard = YardSnapshot.from_env(env)
    position = tuple(env.current_state)
    fixed = online_fixed_obstacles(
        env, reserve_queue_cells=search.reserve_queue_cells,
    ) - {position}
    return RecoveryState.from_yard_snapshot(
        yard, position, fixed_obstacles=fixed,
        pickup_cells=(tuple(env.pickup_cell),),
        wait_cells=(tuple(env.waiting_cell),),
    )
