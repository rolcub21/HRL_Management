"""Constructive relocation-family proofs; no global patches or policy changes."""

from dataclasses import dataclass

from PSLAP import viability as v
from PSLAP.viability_filter import ViabilitySearchConfig
from experiments.conditioned_vcg.E14_certification_scalability_95k.reuse import (
    timing_erased_state,
)


def targeted_action(state, kind, label, destination):
    """Construct exactly the action the canonical enumerator would return.

    Check one destination rather than enumerating every block/destination.
    The original shortest-path routine retains the native tie-breaking rules.
    """
    block = state.block(label)
    if block is None:
        return None
    if kind is v.RecoveryActionKind.DELIVERY:
        if destination not in state.exits:
            return None
    elif kind is v.RecoveryActionKind.RELOCATION:
        if destination not in (
            state.storage_cells - frozenset(state.occupancy())
            - state.fixed_obstacles - state.reserved_cells
        ):
            return None
    else:
        return None
    approach = v._shortest_clear_path(
        state, state.agent_position, block.position, ignore_label=label
    )
    transport = v._shortest_clear_path(
        state, block.position, destination, ignore_label=label
    )
    if approach is None or transport is None:
        return None
    return v.RecoveryAction(kind, label, block.position, destination, approach, transport)


@dataclass(frozen=True, init=False)
class ValidatedAnchor:
    """An immutable witness validated once using the native enumerator."""

    state: v.RecoveryState
    witness: tuple[v.RecoveryAction, ...]
    first_successor: v.RecoveryState | None

    def __init__(self, state, certificate):
        if (
            certificate.status is not v.ViabilityStatus.SAFE
            or certificate.contract != v.CLOSED_ADMISSION_CONTRACT
            or certificate.action_model != v.STRICT_MACRO_ACTION_MODEL
            or certificate.fixed_obstacles != state.fixed_obstacles
            or certificate.reserved_cells != state.reserved_cells
        ):
            raise ValueError("anchor requires a matching SAFE recovery certificate")
        physical = timing_erased_state(state)
        current = physical
        first_successor = None
        for index, action in enumerate(certificate.witness):
            current = v.apply_recovery_action(current, action)
            if index == 0:
                first_successor = current
        if current.blocks:
            raise ValueError("anchor witness does not complete the workload")
        if certificate.witness_primitive_steps != sum(a.steps for a in certificate.witness):
            raise ValueError("anchor witness duration mismatch")
        object.__setattr__(self, "state", physical)
        object.__setattr__(self, "witness", tuple(certificate.witness))
        object.__setattr__(self, "first_successor", first_successor)


@dataclass(frozen=True)
class Attempt:
    certificate: v.RecoverabilityCertificate | None
    reason: str


def try_relocation(anchor: ValidatedAnchor, action, successor, config):
    """Prove successor by legal undo, rebuilt first macro, and validated suffix.

    A miss is unresolved by this rule, never an impossibility certificate.
    Canonical equality at the join includes transporter position and geometry.
    """
    if config.search_order != "goal_directed":
        return Attempt(None, "unsupported_search_order")
    if action.kind is not v.RecoveryActionKind.RELOCATION or not anchor.witness:
        return Attempt(None, "not_nonempty_relocation_family")
    expected = targeted_action(anchor.state, action.kind, action.block_label, action.destination)
    if expected != action:
        return Attempt(None, "invalid_outgoing_action")
    physical = timing_erased_state(successor)
    if v._apply_legal_action(anchor.state, action) != physical:
        return Attempt(None, "successor_mismatch")
    undo = targeted_action(physical, action.kind, action.block_label, action.source)
    if undo is None:
        return Attempt(None, "undo_unavailable")
    restored = v._apply_legal_action(physical, undo)
    first = anchor.witness[0]
    bridge = targeted_action(restored, first.kind, first.block_label, first.destination)
    if bridge is None:
        return Attempt(None, "first_macro_unavailable")
    if v._apply_legal_action(restored, bridge) != anchor.first_successor:
        return Attempt(None, "join_mismatch")
    witness = (undo, bridge, *anchor.witness[1:])
    steps = sum(a.steps for a in witness)
    if config.max_depth is not None and len(witness) > config.max_depth:
        return Attempt(None, "depth_limit")
    if config.max_primitive_steps is not None and steps > config.max_primitive_steps:
        return Attempt(None, "primitive_horizon")
    certificate = v.RecoverabilityCertificate(
        status=v.ViabilityStatus.SAFE, witness=witness,
        witness_primitive_steps=steps, explored_nodes=0, generated_states=0,
        max_depth_reached=len(witness), max_primitive_steps_reached=steps,
        frontier_states=0, exhaustive=False,
        reason="safe:validated_relocation_undo_and_rejoin;completion_upper_bound",
        fixed_obstacles=physical.fixed_obstacles, reserved_cells=physical.reserved_cells,
        primitive_step_budget=config.max_primitive_steps, search_order="goal_directed",
    )
    return Attempt(certificate, "connected")


def certify_with_fallback(anchor, action, successor, config: ViabilitySearchConfig):
    """Standalone experimental API; exact native search handles every miss."""
    attempt = try_relocation(anchor, action, successor, config)
    if attempt.certificate is not None:
        return attempt.certificate, "family"
    return v.analyze_recoverability(
        successor, max_depth=config.max_depth, max_nodes=config.max_nodes,
        max_primitive_steps=config.max_primitive_steps, search_order=config.search_order,
    ), "exact_fallback"
