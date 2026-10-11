"""Certificate-preserving reuse mechanisms evaluated by E14.

Nothing in this module is installed into the production verifier.  E14 applies
the mechanisms under explicit scoped contexts and compares every result with
the frozen verifier before drawing an acceleration conclusion.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
from typing import Iterable, Mapping, Optional, Sequence
from unittest.mock import patch

import PSLAP.viability as viability
import PSLAP.viability_candidates as candidate_module
from PSLAP.dynamic_yard import BlockView, Cell
from PSLAP.viability import (
    RecoverabilityCertificate,
    RecoveryAction,
    RecoveryActionKind,
    RecoveryState,
    ViabilityStatus,
)
from PSLAP.viability_candidates import ViabilityCertificateCache
from PSLAP.viability_filter import ViabilitySearchConfig
from PSLAP.viability_dataset import recovery_state_to_dict


STAGE_CURRENT = "current"
STAGE_TIMING_KEY = "timing_invariant_key"
STAGE_PATH_CLEANUP = "path_enumeration_cleanup"
STAGE_WITNESS_SUFFIX = "witness_suffix_store"
STAGES = (
    STAGE_CURRENT,
    STAGE_TIMING_KEY,
    STAGE_PATH_CLEANUP,
    STAGE_WITNESS_SUFFIX,
)


class ReuseError(RuntimeError):
    pass


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def timing_erased_state(state: RecoveryState) -> RecoveryState:
    """Return the physical state used by the timing-independent verifier."""

    if not isinstance(state, RecoveryState):
        raise TypeError("timing erasure requires a RecoveryState")
    return replace(
        state,
        blocks=tuple(
            replace(block, remaining_time=0.0) for block in state.blocks
        ),
    )


def physical_state_id(state: RecoveryState) -> str:
    return digest(recovery_state_to_dict(timing_erased_state(state)))


def full_state_id(state: RecoveryState) -> str:
    return digest(recovery_state_to_dict(state))


def action_to_dict(action: RecoveryAction) -> dict:
    return {
        "kind": action.kind.value,
        "block_label": action.block_label,
        "source": list(action.source),
        "destination": list(action.destination),
        "approach_path": [list(cell) for cell in action.approach_path],
        "transport_path": [list(cell) for cell in action.transport_path],
        "steps": action.steps,
    }


def action_from_dict(value: Mapping) -> RecoveryAction:
    action = RecoveryAction(
        kind=RecoveryActionKind(str(value["kind"])),
        block_label=str(value["block_label"]),
        source=tuple(value["source"]),
        destination=tuple(value["destination"]),
        approach_path=tuple(tuple(cell) for cell in value["approach_path"]),
        transport_path=tuple(tuple(cell) for cell in value["transport_path"]),
    )
    if int(value.get("steps", action.steps)) != action.steps:
        raise ReuseError("serialized recovery action has inconsistent duration")
    return action


def certificate_to_dict(certificate: RecoverabilityCertificate) -> dict:
    return {
        "status": certificate.status.value,
        "witness": [action_to_dict(action) for action in certificate.witness],
        "witness_primitive_steps": certificate.witness_primitive_steps,
        "explored_nodes": certificate.explored_nodes,
        "generated_states": certificate.generated_states,
        "max_depth_reached": certificate.max_depth_reached,
        "max_primitive_steps_reached": certificate.max_primitive_steps_reached,
        "frontier_states": certificate.frontier_states,
        "exhaustive": certificate.exhaustive,
        "reason": certificate.reason,
        "fixed_obstacles": [list(cell) for cell in sorted(certificate.fixed_obstacles)],
        "reserved_cells": [list(cell) for cell in sorted(certificate.reserved_cells)],
        "primitive_step_budget": certificate.primitive_step_budget,
        "search_order": certificate.search_order,
        "contract": certificate.contract,
        "action_model": certificate.action_model,
    }


def certificate_from_dict(value: Mapping) -> RecoverabilityCertificate:
    certificate = RecoverabilityCertificate(
        status=ViabilityStatus(str(value["status"])),
        witness=tuple(action_from_dict(item) for item in value["witness"]),
        witness_primitive_steps=value["witness_primitive_steps"],
        explored_nodes=int(value["explored_nodes"]),
        generated_states=int(value["generated_states"]),
        max_depth_reached=int(value["max_depth_reached"]),
        max_primitive_steps_reached=int(value["max_primitive_steps_reached"]),
        frontier_states=int(value["frontier_states"]),
        exhaustive=bool(value["exhaustive"]),
        reason=str(value["reason"]),
        fixed_obstacles=frozenset(tuple(cell) for cell in value["fixed_obstacles"]),
        reserved_cells=frozenset(tuple(cell) for cell in value["reserved_cells"]),
        primitive_step_budget=value["primitive_step_budget"],
        search_order=str(value["search_order"]),
        contract=str(value["contract"]),
        action_model=str(value["action_model"]),
    )
    expected_steps = (
        sum(action.steps for action in certificate.witness)
        if certificate.status is ViabilityStatus.SAFE
        else None
    )
    if certificate.witness_primitive_steps != expected_steps:
        raise ReuseError("serialized certificate has inconsistent witness steps")
    if certificate.status is ViabilityStatus.UNSAFE and not certificate.exhaustive:
        raise ReuseError("UNSAFE certificate must be exhaustive")
    return certificate


def config_to_dict(config: ViabilitySearchConfig) -> dict:
    return {
        "max_depth": config.max_depth,
        "max_nodes": config.max_nodes,
        "max_primitive_steps": config.max_primitive_steps,
        "reserve_queue_cells": config.reserve_queue_cells,
        "search_order": config.search_order,
    }


def config_from_dict(value: Mapping) -> ViabilitySearchConfig:
    return ViabilitySearchConfig(**dict(value))


def _path_tree(
    state: RecoveryState,
    start: Cell,
    *,
    ignore_label: Optional[str],
) -> dict[Cell, Optional[Cell]]:
    """One canonical BFS tree replacing one BFS call per destination."""

    if start not in state.traversable or start in state.fixed_obstacles:
        return {}
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
        for neighbor in viability._neighbors(cell):
            if (
                neighbor in state.traversable
                and neighbor not in blocked
                and neighbor not in parent
            ):
                parent[neighbor] = cell
                queue.append(neighbor)
    return parent


def _tree_path(
    parent: Mapping[Cell, Optional[Cell]], goal: Cell
) -> Optional[tuple[Cell, ...]]:
    if goal not in parent:
        return None
    path = []
    cursor: Optional[Cell] = goal
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    return tuple(reversed(path))


def legal_recovery_actions_optimized(
    state: RecoveryState,
) -> tuple[RecoveryAction, ...]:
    """Generate the exact canonical action tuple with one transport BFS/block."""

    if not isinstance(state, RecoveryState):
        raise TypeError("legal_recovery_actions requires a RecoveryState")
    occupied = frozenset(state.occupancy())
    empty_storage = tuple(sorted(
        state.storage_cells
        - occupied
        - state.fixed_obstacles
        - state.reserved_cells
    ))
    actions = []
    for block in state.blocks:
        approach_parent = _path_tree(
            state, state.agent_position, ignore_label=block.label
        )
        approach = _tree_path(approach_parent, block.position)
        if approach is None:
            continue
        transport_parent = _path_tree(
            state, block.position, ignore_label=block.label
        )
        for exit_cell in state.exits:
            transport = _tree_path(transport_parent, exit_cell)
            if transport is not None:
                actions.append(RecoveryAction(
                    RecoveryActionKind.DELIVERY,
                    block.label,
                    block.position,
                    exit_cell,
                    approach,
                    transport,
                ))
        for destination in empty_storage:
            transport = _tree_path(transport_parent, destination)
            if transport is not None:
                actions.append(RecoveryAction(
                    RecoveryActionKind.RELOCATION,
                    block.label,
                    block.position,
                    destination,
                    approach,
                    transport,
                ))
    return tuple(sorted(actions, key=viability._action_sort_key))


def apply_enumerated_action(
    state: RecoveryState, action: RecoveryAction
) -> RecoveryState:
    """Apply an action already returned by the scoped canonical enumerator."""

    return viability._apply_legal_action(state, action)


@contextmanager
def path_cleanup_active():
    """Activate only ordering-preserving enumeration changes for E14."""

    with patch.object(
        viability, "legal_recovery_actions", legal_recovery_actions_optimized
    ), patch.object(
        candidate_module,
        "legal_recovery_actions",
        legal_recovery_actions_optimized,
    ), patch.object(
        candidate_module,
        "apply_recovery_action",
        apply_enumerated_action,
    ):
        yield


def canonical_cache_key(key):
    """Project only block clocks out of the current certificate-cache key."""

    if not isinstance(key, tuple) or len(key) != 5 or not isinstance(key[0], RecoveryState):
        raise ReuseError("unexpected certificate cache key")
    return (timing_erased_state(key[0]), *key[1:])


class TimingInvariantCertificateCache(ViabilityCertificateCache):
    """Drop clocks from cache identity without changing stored certificates."""

    def __getitem__(self, key):
        return self._entries[canonical_cache_key(key)]

    def __setitem__(self, key, value):
        if not isinstance(value, RecoverabilityCertificate):
            raise TypeError("viability cache values must be certificates")
        self._entries[canonical_cache_key(key)] = value

    def __delitem__(self, key):
        del self._entries[canonical_cache_key(key)]


def _witness_steps(witness: Sequence[RecoveryAction]) -> int:
    return sum(action.steps for action in witness)


@dataclass(frozen=True)
class StoredPositiveProof:
    state: RecoveryState
    witness: tuple[RecoveryAction, ...]

    @property
    def macro_count(self) -> int:
        return len(self.witness)

    @property
    def primitive_steps(self) -> int:
        return _witness_steps(self.witness)


class PositiveProofStore:
    """On-demand constructive subset of the physical completion basin."""

    def __init__(self) -> None:
        self._proofs: dict[RecoveryState, StoredPositiveProof] = {}
        self.insertions = 0
        self.replacements = 0
        self.lookups = 0
        self.hits = 0

    def __len__(self) -> int:
        return len(self._proofs)

    def _validate_witness(
        self,
        state: RecoveryState,
        witness: Sequence[RecoveryAction],
        *,
        action_function=legal_recovery_actions_optimized,
    ) -> tuple[RecoveryState, ...]:
        current = timing_erased_state(state)
        states = [current]
        for action in witness:
            if action not in action_function(current):
                raise ReuseError("stored witness contains a noncanonical action")
            current = apply_enumerated_action(current, action)
            states.append(current)
        if current.blocks:
            raise ReuseError("stored positive witness does not reach completion")
        return tuple(states)

    def insert_certificate(
        self,
        state: RecoveryState,
        certificate: RecoverabilityCertificate,
        *,
        action_function=legal_recovery_actions_optimized,
    ) -> int:
        if certificate.status is not ViabilityStatus.SAFE:
            return 0
        witness = tuple(certificate.witness)
        states = self._validate_witness(
            state, witness, action_function=action_function
        )
        added = 0
        for index, suffix_state in enumerate(states):
            suffix = witness[index:]
            proof = StoredPositiveProof(suffix_state, suffix)
            previous = self._proofs.get(suffix_state)
            if previous is None:
                self._proofs[suffix_state] = proof
                self.insertions += 1
                added += 1
            elif (
                proof.primitive_steps,
                proof.macro_count,
                tuple(action_to_dict(action)["kind"] for action in proof.witness),
            ) < (
                previous.primitive_steps,
                previous.macro_count,
                tuple(action_to_dict(action)["kind"] for action in previous.witness),
            ):
                self._proofs[suffix_state] = proof
                self.replacements += 1
        return added

    def lookup(
        self, state: RecoveryState, config: ViabilitySearchConfig
    ) -> Optional[RecoverabilityCertificate]:
        self.lookups += 1
        physical = timing_erased_state(state)
        proof = self._proofs.get(physical)
        if proof is None:
            return None
        if config.search_order != "goal_directed":
            # The present certificate type would interpret a nonempty BFS
            # witness as a shortest rank. E14 therefore refuses that claim.
            return None
        if config.max_depth is not None and proof.macro_count > config.max_depth:
            return None
        if (
            config.max_primitive_steps is not None
            and proof.primitive_steps > config.max_primitive_steps
        ):
            return None
        self.hits += 1
        return RecoverabilityCertificate(
            status=ViabilityStatus.SAFE,
            witness=proof.witness,
            witness_primitive_steps=proof.primitive_steps,
            explored_nodes=0,
            generated_states=1,
            max_depth_reached=proof.macro_count,
            max_primitive_steps_reached=proof.primitive_steps,
            frontier_states=0,
            exhaustive=not physical.blocks,
            reason=(
                "safe:stored_constructive_witness_suffix;"
                "timing_erased_physical_state;goal_directed_upper_bound"
            ),
            fixed_obstacles=physical.fixed_obstacles,
            reserved_cells=physical.reserved_cells,
            primitive_step_budget=config.max_primitive_steps,
            search_order="goal_directed",
        )


def key_to_config(key) -> ViabilitySearchConfig:
    if not isinstance(key, tuple) or len(key) != 5:
        raise ReuseError("unexpected certificate cache key")
    return ViabilitySearchConfig(
        max_depth=key[1],
        max_nodes=key[2],
        max_primitive_steps=key[3],
        search_order=key[4],
    )


class WitnessSuffixCertificateCache(TimingInvariantCertificateCache):
    """Separate ordinary outcomes from constructive positive proofs."""

    def __init__(self) -> None:
        super().__init__()
        self.proofs = PositiveProofStore()
        self.outcome_hits = 0
        self.proof_hits = 0

    def get(self, key, default=None):
        canonical = canonical_cache_key(key)
        outcome = self._entries.get(canonical)
        if outcome is not None and outcome.status is not ViabilityStatus.UNKNOWN:
            self.outcome_hits += 1
            return outcome
        proof = self.proofs.lookup(key[0], key_to_config(key))
        if proof is not None:
            self.proof_hits += 1
            return proof
        if outcome is not None:
            self.outcome_hits += 1
            return outcome
        return default

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if value.status is ViabilityStatus.SAFE:
            self.proofs.insert_certificate(key[0], value)


def replay_witness(
    state: RecoveryState,
    certificate: RecoverabilityCertificate,
    *,
    action_function=legal_recovery_actions_optimized,
) -> RecoveryState:
    if certificate.status is not ViabilityStatus.SAFE:
        raise ReuseError("only SAFE certificates carry completion witnesses")
    current = timing_erased_state(state)
    for action in certificate.witness:
        if action not in action_function(current):
            raise ReuseError("certificate witness is not legal at replay")
        current = apply_enumerated_action(current, action)
    if current.blocks:
        raise ReuseError("certificate witness did not reach completion")
    return current


def validate_timing_abstraction(state: RecoveryState) -> dict:
    """Check action equivariance and successor homomorphism for one state."""

    physical = timing_erased_state(state)
    original_actions = viability.legal_recovery_actions(state)
    physical_actions = viability.legal_recovery_actions(physical)
    optimized_actions = legal_recovery_actions_optimized(physical)
    if original_actions != physical_actions:
        raise ReuseError("clock erasure changed the legal action tuple")
    if physical_actions != optimized_actions:
        raise ReuseError("path cleanup changed the legal action tuple")
    for action in original_actions:
        original_successor = viability._apply_legal_action(state, action)
        physical_successor = apply_enumerated_action(physical, action)
        if timing_erased_state(original_successor) != physical_successor:
            raise ReuseError("timing projection is not transition-homomorphic")
    return {
        "physical_state_id": physical_state_id(state),
        "legal_action_count": len(original_actions),
        "terminal_equal": (not state.blocks) == (not physical.blocks),
        "actions_equal": True,
        "optimized_actions_equal": True,
        "successors_equivariant": True,
    }


__all__ = [
    "PositiveProofStore",
    "ReuseError",
    "STAGES",
    "STAGE_CURRENT",
    "STAGE_PATH_CLEANUP",
    "STAGE_TIMING_KEY",
    "STAGE_WITNESS_SUFFIX",
    "TimingInvariantCertificateCache",
    "WitnessSuffixCertificateCache",
    "action_from_dict",
    "action_to_dict",
    "apply_enumerated_action",
    "certificate_from_dict",
    "certificate_to_dict",
    "config_from_dict",
    "config_to_dict",
    "full_state_id",
    "legal_recovery_actions_optimized",
    "path_cleanup_active",
    "physical_state_id",
    "replay_witness",
    "timing_erased_state",
    "validate_timing_abstraction",
]
