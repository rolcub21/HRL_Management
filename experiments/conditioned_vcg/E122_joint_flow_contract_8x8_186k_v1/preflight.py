"""Deterministic contract fixtures; a gate, not a full general liveness theorem."""

from dataclasses import replace
from types import SimpleNamespace

from PSLAP.dynamic_yard import BlockView, YardSnapshot
from PSLAP.viability import RecoveryState, ViabilityStatus
from PSLAP.viability_candidates import ViabilityMode
from viability_graph_hierarchy import RecoveryWitnessGuard
from .flow import JointVerifier, NATIVE_ANALYZE, access, physical_key
from .policy import ProgressOnlyDeferRule


def yard_state(blocks=()):
    traversable = frozenset({(r, c) for r in range(1, 7) for c in range(1, 7)}
                            | {(0, 6), (7, 4), (7, 5), (7, 6)})
    storage = frozenset({(r, c) for r in range(1, 7) for c in range(1, 7)} - {(1, 6)})
    yard = YardSnapshot(8, 8, traversable, storage, ((7, 4), (7, 5), (7, 6)), tuple(blocks))
    return RecoveryState.from_yard_snapshot(yard, agent_position=(1, 1),
        fixed_obstacles=((1, 6), (0, 6)), pickup_cells=((1, 6),), wait_cells=((0, 6),))


def checks():
    results = {}
    verifier = JointVerifier()
    empty = yard_state()
    results["empty_yard_access"] = access(empty).allowed and access(empty).reachable_free_cells == 35
    trap = yard_state((BlockView("A", (1, 5), 30.), BlockView("B", (2, 6), 50.)))
    native = NATIVE_ANALYZE(trap, search_order="goal_directed")
    joint = verifier.analyze(trap, search_order="goal_directed")
    results["two_block_trap_native_safe_but_joint_rejected"] = (
        native.status is ViabilityStatus.SAFE and joint.status is ViabilityStatus.UNSAFE
        and access(trap).free_cells == 33 and access(trap).reachable_free_cells == 0)
    open_state = yard_state((BlockView("A", (1, 5), -100.), BlockView("B", (3, 3), 50.)))
    safe = verifier.analyze(open_state, search_order="goal_directed")
    results["open_state_has_prefix_preserving_recovery_witness"] = safe.status is ViabilityStatus.SAFE
    closing = [a for a in verifier.actions(open_state) if a.block_label == "B" and a.destination == (2, 6)]
    results["reconfiguration_cannot_close_second_entrance"] = not closing
    # Conditional inflow: no free slot permits recovery-only, but the first
    # freeing move must restore access. This tiny full yard has such a move.
    full = RecoveryState(3, 3, frozenset(((1, 0), (1, 1), (1, 2))),
        frozenset(((1, 1),)), ((1, 2),), (BlockView("X", (1, 1), 0.),), (1, 2),
        fixed_obstacles=frozenset(((1, 0),)), pickup_cells=frozenset(((1, 0),)))
    full_certificate = verifier.analyze(full)
    results["full_yard_admissions_closed_with_joint_recovery"] = (
        access(full).reason == "full_admissions_closed" and full_certificate.status is ViabilityStatus.SAFE)
    results["clock_invariance"] = access(open_state) == access(replace(open_state,
        blocks=tuple(replace(b, remaining_time=10000.) for b in open_state.blocks)))
    # Replay the actual guard interface, not a substitute progress counter.
    guard = RecoveryWitnessGuard(max_nonprogress_recovery_decisions=2, force_when_due=False)
    rule = ProgressOnlyDeferRule()
    defer = SimpleNamespace(liveness_forced=False, candidate=SimpleNamespace(mode=ViabilityMode.DEFER))
    reconfigure = SimpleNamespace(liveness_forced=False, exact_rank_progress=False,
                                 candidate=SimpleNamespace(mode=ViabilityMode.RECOVER,
                                                           action_type="reconfigure"))
    for _ in range(2):
        for streak in range(16):
            decision = rule.evaluate(SimpleNamespace(blocks=()), open_state.to_yard_snapshot(),
                inbound_present=True, current_certificate=safe,
                consecutive_defer_decisions=streak, terminal=False)
            if not decision.allowed:
                raise RuntimeError("E122 preflight unexpectedly forbids bounded wait")
            guard.observe(defer, next_snapshot=None, done=False)
        bounded = rule.evaluate(SimpleNamespace(blocks=()), open_state.to_yard_snapshot(),
            inbound_present=True, current_certificate=safe, consecutive_defer_decisions=16, terminal=False)
        if bounded.allowed:
            raise RuntimeError("E122 preflight Defer streak exceeded")
        guard.observe(reconfigure, next_snapshot=None, done=False)
    first = safe.witness[0]
    snapshot = SimpleNamespace(current_certificate=safe, recovery_state=open_state,
        candidates=(SimpleNamespace(mode=ViabilityMode.RECOVER, recovery_action=first),))
    prepared = SimpleNamespace(source_indices=(0,))
    forced = guard.forced_prepared_index(snapshot, prepared)
    results["alternating_defers_and_reconfigurations_cannot_reset_guard"] = forced == 0 and guard.active
    results["guard_uses_joint_prefix_preserving_witness"] = guard.active_witness == safe.witness
    results["native_verifier_unmodified_after_joint_search"] = NATIVE_ANALYZE(trap).status is ViabilityStatus.SAFE
    return {"passed": all(results.values()), "checks": results,
            "joint_certificate_scope": "conditional admission access at macro boundaries, including recovery witness prefixes",
            "not_claimed": "arbitrary future arrival/service viability, primitive-time ingress availability, or an unbounded-capacity guarantee",
            "training_runs": 0, "episode_runs": 0}
