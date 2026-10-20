"""Exact closed-admission recovery constrained by conditional pickup access.

No core verifier files are edited. Recovery search uses the same physical
macros, restricted to successors satisfying the additional access property.
The installed context is single-threaded and must surround enumeration and
execution. Original relocation-family certificates are deliberately disabled.
"""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from time import perf_counter
from unittest.mock import patch

import PSLAP.viability as recovery
import PSLAP.viability_candidates as frontier
import PSLAP.viability_filter as admission
from experiments.conditioned_vcg.E14_certification_scalability_95k.reuse import TimingInvariantCertificateCache

JOINT_CONTRACT = "e122_closed_admission_recovery_with_conditional_pickup_access_every_boundary_v1"
NATIVE_ACTIONS = recovery.legal_recovery_actions
NATIVE_ANALYZE = recovery.analyze_recoverability


@dataclass(frozen=True)
class AccessResult:
    allowed: bool
    reason: str
    free_cells: int
    reachable_free_cells: int
    agent_can_approach_pickup: bool


def neighbors(cell):
    row, col = cell
    return ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))


@lru_cache(maxsize=50_000)
def access(state):
    """Check actual capacity and both empty-hand/laden ingress connectivity.

    Pickup is a route endpoint for the arriving block, not a permanently
    impassable queue obstacle for THIS check. All other fixed obstacles,
    including reserved waiting cells, remain blocked. Recovery itself retains
    its original worst-case fixed-obstacle contract.
    """
    occupied = frozenset(block.position for block in state.blocks)
    free = state.storage_cells - occupied - state.reserved_cells - state.fixed_obstacles
    if not free:
        return AccessResult(True, "full_admissions_closed", 0, 0, False)
    if not state.pickup_cells:
        return AccessResult(False, "pickup_geometry_missing", len(free), 0, False)
    reachable_free = set()
    approached = False
    for pickup in sorted(state.pickup_cells):
        blocked = occupied | (state.fixed_obstacles - {pickup})
        if pickup in occupied:
            continue
        seen = {pickup}
        queue = deque([pickup])
        while queue:
            cell = queue.popleft()
            for other in neighbors(cell):
                if other in state.traversable and other not in blocked and other not in seen:
                    seen.add(other)
                    queue.append(other)
        # The transporter can start on a stored block after PUTDOWN, but that
        # exception must not make its cell traversable for the laden inlet path.
        agent = state.agent_position
        approach = agent in seen or (agent in occupied and any(n in seen for n in neighbors(agent)))
        if approach:
            approached = True
            reachable_free.update(free.intersection(seen))
    return AccessResult(bool(reachable_free),
                        "pickup_access_preserved" if reachable_free else
                        ("pickup_to_free_storage_blocked" if approached else "agent_or_pickup_access_blocked"),
                        len(free), len(reachable_free), approached)


def physical_key(state):
    return replace(state, blocks=tuple(replace(block, remaining_time=0.) for block in state.blocks))


class JointCertificateCache(TimingInvariantCertificateCache):
    """Never accept a certificate for the unaugmented transition system."""
    def __setitem__(self, key, value):
        if value.contract != JOINT_CONTRACT:
            raise RuntimeError("E122 joint cache rejected a native/foreign certificate")
        super().__setitem__(key, value)


class JointVerifier:
    def __init__(self):
        self.search_calls = 0
        self.access_rejected_states = 0
        self.rejected_recovery_edges = 0
        self.verified_witnesses = 0
        self.search_seconds = 0.
        self._actions = lru_cache(maxsize=20_000)(self._allowed_actions)

    def _allowed_actions(self, state):
        if not access(state).allowed:
            return ()
        original = NATIVE_ACTIONS(state)
        kept = tuple(action for action in original
                     if access(physical_key(recovery._apply_legal_action(state, action))).allowed)
        self.rejected_recovery_edges += len(original) - len(kept)
        return kept

    def actions(self, state):
        return self._actions(physical_key(state))

    def verify_witness(self, state, certificate):
        if certificate.status is not recovery.ViabilityStatus.SAFE:
            return
        if not access(physical_key(state)).allowed:
            raise RuntimeError("E122 witness starts outside joint access set")
        current = state
        for action in certificate.witness:
            if action not in NATIVE_ACTIONS(current):
                raise RuntimeError("E122 witness contains a physically illegal macro")
            current = recovery._apply_legal_action(current, action)
            if not access(physical_key(current)).allowed:
                raise RuntimeError("E122 recovery witness violates pickup access")
        if current.blocks:
            raise RuntimeError("E122 SAFE witness did not recover all accepted blocks")
        self.verified_witnesses += 1

    def analyze(self, state, **kwargs):
        started = perf_counter()
        self.search_calls += 1
        if not access(physical_key(state)).allowed:
            self.access_rejected_states += 1
            value = recovery._certificate(
                state, status=recovery.ViabilityStatus.UNSAFE,
                explored_nodes=0, generated_states=1, max_depth_reached=0,
                max_primitive_steps_reached=0, frontier_states=0, exhaustive=True,
                reason="unsafe:joint_access_invariant_false_at_initial_boundary",
                primitive_step_budget=kwargs.get("max_primitive_steps"),
                search_order=kwargs.get("search_order", "breadth_first"))
        else:
            with patch.object(recovery, "legal_recovery_actions", self.actions):
                value = NATIVE_ANALYZE(state, **kwargs)
        value = replace(value, contract=JOINT_CONTRACT, reason=f"joint_access;{value.reason}")
        self.verify_witness(state, value)
        self.search_seconds += perf_counter() - started
        return value

    @contextmanager
    def enabled(self):
        with patch.object(frontier, "analyze_recoverability", self.analyze), \
             patch.object(admission, "analyze_recoverability", self.analyze), \
             patch.object(frontier, "legal_recovery_actions", self.actions):
            yield self
        self._actions.cache_clear()
        access.cache_clear()

    def enumerate(self, env, **kwargs):
        if not isinstance(kwargs.get("cache"), JointCertificateCache):
            raise RuntimeError("E122 requires an isolated, joint-contract certificate cache")
        kwargs["recovery_certification_strategy"] = frontier.EXACT_ONLY_RECOVERY_CERTIFICATION
        snapshot = frontier.enumerate_viability_candidates(env, **kwargs)
        if snapshot.current_certificate.status is not recovery.ViabilityStatus.SAFE:
            raise RuntimeError("E122 current state has no joint SAFE recovery witness")
        for candidate in snapshot.candidates:
            if candidate.certificate.contract != JOINT_CONTRACT:
                raise RuntimeError("E122 joint frontier contains a native certificate")
            if not access(physical_key(candidate.successor_state)).allowed:
                raise RuntimeError("E122 candidate violates joint access property")
        return replace(snapshot, audit=replace(snapshot.audit,
            certification_contract=JOINT_CONTRACT,
            post_accept_contract=JOINT_CONTRACT,
            verifier_authority="e122_exact_joint_recovery_transition_search_v1"))

    def stats(self):
        return {key: getattr(self, key) for key in (
            "search_calls", "access_rejected_states", "rejected_recovery_edges",
            "verified_witnesses", "search_seconds")}
