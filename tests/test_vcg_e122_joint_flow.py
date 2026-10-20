from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import PSLAP.viability as recovery
import PSLAP.viability_candidates as frontier
import PSLAP.viability_filter as admission
from PSLAP.dynamic_yard import BlockView
from PSLAP.viability import RecoveryState, ViabilityStatus
from experiments.conditioned_vcg.E122_joint_flow_contract_8x8_186k_v1 import flow, policy, preflight, protocol


def open_state():
    return preflight.yard_state((BlockView("A", (1, 5), 30.), BlockView("B", (3, 3), 50.)))


def test_preflight_contract_and_actual_guard_fixtures():
    result = preflight.checks()
    assert result["passed"] and all(result["checks"].values())


def test_access_is_not_a_two_cell_hardcoded_veto():
    # Different layout: the one central block separates pickup from free space.
    state = RecoveryState(3, 5, frozenset((1, c) for c in range(5)),
        frozenset(((1, 1), (1, 2), (1, 3))), ((1, 4),),
        (BlockView("X", (1, 1), 0.),), (1, 4), pickup_cells=frozenset(((1, 0),)))
    result = flow.access(state)
    assert not result.allowed and result.free_cells == 2
    assert flow.access(replace(state, blocks=())).allowed


def test_reachable_free_cell_alone_is_not_an_executable_inflow_guarantee():
    state = RecoveryState(3, 5, frozenset((1, c) for c in range(5)),
        frozenset(((1, 1), (1, 2), (1, 3))), ((1, 4),),
        (BlockView("X", (1, 2), 0.),), (1, 4), pickup_cells=frozenset(((1, 0),)))
    # Free slot (1,1) is reachable from pickup, but the transporter cannot get there.
    assert not flow.access(state).allowed
    assert not flow.access(state).agent_can_approach_pickup


def test_pickup_endpoint_exception_does_not_unreserve_waiting_cells():
    state = open_state()
    assert (1, 6) in state.fixed_obstacles and (0, 6) in state.fixed_obstacles
    assert flow.access(state).allowed
    before = state.fixed_obstacles
    flow.access(state)
    assert state.fixed_obstacles == before


def test_agent_on_putdown_cell_may_leave_but_laden_path_cannot_cross_it():
    trap = preflight.yard_state((BlockView("A", (1, 5), 30.), BlockView("B", (2, 6), 50.)))
    state = replace(trap, agent_position=(2, 6))
    assert not flow.access(state).allowed


def test_joint_verifier_does_not_relabel_a_native_witness_without_checking_it():
    state = open_state()
    closing = next(a for a in flow.NATIVE_ACTIONS(state) if a.block_label == "B" and a.destination == (2, 6))
    native = flow.NATIVE_ANALYZE(state, search_order="goal_directed")
    invalid = replace(native, witness=(closing,))
    with pytest.raises(RuntimeError, match="violates pickup access"):
        flow.JointVerifier().verify_witness(state, invalid)


def test_every_returned_joint_recovery_action_preserves_access():
    state = open_state()
    verifier = flow.JointVerifier()
    for action in verifier.actions(state):
        assert action in flow.NATIVE_ACTIONS(state)
        assert flow.access(recovery._apply_legal_action(state, action)).allowed
    assert verifier.rejected_recovery_edges > 0


def test_joint_certificate_is_distinct_and_search_cutoff_stays_unknown():
    verifier = flow.JointVerifier()
    certificate = verifier.analyze(open_state(), max_nodes=1)
    assert certificate.status is ViabilityStatus.UNKNOWN
    assert certificate.contract == flow.JOINT_CONTRACT


def test_native_certificates_cannot_enter_joint_cache():
    state = open_state()
    native = flow.NATIVE_ANALYZE(state, search_order="goal_directed")
    key = (state, None, 100000, None, "goal_directed")
    cache = flow.JointCertificateCache()
    with pytest.raises(RuntimeError, match="native/foreign"):
        cache[key] = native
    joint = flow.JointVerifier().analyze(state, search_order="goal_directed")
    cache[key] = joint
    assert cache[key].contract == flow.JOINT_CONTRACT


def test_joint_context_restores_native_functions_even_on_failure():
    original = (frontier.analyze_recoverability, admission.analyze_recoverability,
                frontier.legal_recovery_actions, recovery.legal_recovery_actions)
    with pytest.raises(ValueError, match="fixture"):
        with flow.JointVerifier().enabled():
            raise ValueError("fixture")
    assert original == (frontier.analyze_recoverability, admission.analyze_recoverability,
                        frontier.legal_recovery_actions, recovery.legal_recovery_actions)


@pytest.mark.parametrize("remaining", [-100, 0, 20, 100])
@pytest.mark.parametrize("inbound", [False, True])
def test_progress_rule_has_no_due_or_inbound_preference(remaining, inbound):
    value = policy.ProgressOnlyDeferRule().evaluate(SimpleNamespace(blocks=()),
        SimpleNamespace(blocks=(SimpleNamespace(remaining_time=remaining),)),
        inbound_present=inbound, current_certificate=SimpleNamespace(status=ViabilityStatus.SAFE),
        consecutive_defer_decisions=0, terminal=False)
    assert value.allowed and value.horizon_steps == 10


@pytest.mark.parametrize("status", [ViabilityStatus.UNSAFE, ViabilityStatus.UNKNOWN])
def test_progress_rule_does_not_relax_safe_certificate_requirement(status):
    value = policy.ProgressOnlyDeferRule().evaluate(SimpleNamespace(blocks=()), SimpleNamespace(blocks=(1,)),
        inbound_present=True, current_certificate=SimpleNamespace(status=status),
        consecutive_defer_decisions=0, terminal=False)
    assert not value.allowed


def test_progress_rule_respects_streak_terminal_and_no_work():
    rule = policy.ProgressOnlyDeferRule()
    options = dict(inbound_present=False, current_certificate=SimpleNamespace(status=ViabilityStatus.SAFE),
                   consecutive_defer_decisions=0, terminal=False)
    env, yard = SimpleNamespace(blocks=[]), SimpleNamespace(blocks=())
    assert not rule.evaluate(env, yard, **options).allowed
    yard.blocks = (1,)
    assert not rule.evaluate(env, yard, **{**options, "terminal": True}).allowed
    assert not rule.evaluate(env, yard, **{**options, "consecutive_defer_decisions": 16}).allowed


def test_fresh_panels_are_disjoint_and_source_overlap_is_rejected():
    assert protocol.verify_freshness({"train_instance_seeds": [1], "validation_instance_seeds": [2]})
    with pytest.raises(RuntimeError, match="overlaps"):
        protocol.verify_freshness({"train_instance_seeds": [protocol.PANELS["pilot"][0]]})
    with pytest.raises(RuntimeError, match="overlaps"):
        protocol.verify_freshness({"sealed_stress_test_seeds": {"8x8": [protocol.PANELS["mechanism"][0]]}})


def test_incomplete_trajectories_cannot_enter_quality_means():
    from experiments.conditioned_vcg.E122_joint_flow_contract_8x8_186k_v1.run import aggregate
    assert aggregate([]) is None
    assert aggregate([{"strict_complete": False, "raw": {"operator_cost": -1000}}]) is None
