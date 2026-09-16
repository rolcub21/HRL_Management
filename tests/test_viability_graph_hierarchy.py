from dataclasses import replace
import math
import unittest

import torch

from PSLAP.dynamic_yard import BlockView
from PSLAP.viability import (
    RecoverabilityCertificate,
    RecoveryAction,
    RecoveryActionKind,
    RecoveryState,
    ViabilityStatus,
)
from PSLAP.viability_candidates import (
    ViabilityActionCandidate,
    ViabilityActionType,
    ViabilityCandidateAudit,
    ViabilityCandidateSnapshot,
    ViabilityMode,
)
from viability_graph_hierarchy import (
    ACTION_FEATURE_DIM,
    MODE_TO_ID,
    NoCertifiedViableAction,
    RecoveryWitnessMismatch,
    ViabilityGraphConfig,
    ViabilityGraphHierarchyAgent,
    cardinality_normalized_continuation,
    common_smdp_target,
    double_dqn_regularized_continuation,
    prepare_viability_snapshot,
)


class DummyOption:
    pass


def state_with_block(position=(0, 1), *, remaining=-2.0, agent=(0, 0)):
    return RecoveryState(
        rows=2,
        cols=4,
        traversable=frozenset(
            (row, column) for row in range(2) for column in range(4)
        ),
        storage_cells=frozenset({(0, 1), (0, 2), (1, 1), (1, 2)}),
        exits=((0, 3),),
        blocks=(BlockView("B", position, remaining),),
        agent_position=agent,
        pickup_cells=frozenset({(1, 0)}),
        wait_cells=frozenset({(1, 3)}),
    )


def empty_state(*, agent=(0, 3)):
    return replace(state_with_block(), blocks=(), agent_position=agent)


def safe_certificate(state, witness=(), *, search_order="breadth_first"):
    witness = tuple(witness)
    return RecoverabilityCertificate(
        status=ViabilityStatus.SAFE,
        witness=witness,
        witness_primitive_steps=sum(action.steps for action in witness),
        explored_nodes=1,
        generated_states=1,
        max_depth_reached=len(witness),
        max_primitive_steps_reached=sum(action.steps for action in witness),
        frontier_states=0,
        exhaustive=False,
        reason="test_safe",
        fixed_obstacles=state.fixed_obstacles,
        reserved_cells=state.reserved_cells,
        primitive_step_budget=None,
        search_order=search_order,
    )


def unsafe_certificate(state):
    return RecoverabilityCertificate(
        status=ViabilityStatus.UNSAFE,
        witness=(),
        witness_primitive_steps=None,
        explored_nodes=2,
        generated_states=2,
        max_depth_reached=1,
        max_primitive_steps_reached=1,
        frontier_states=0,
        exhaustive=True,
        reason="test_unsafe",
        fixed_obstacles=state.fixed_obstacles,
        reserved_cells=state.reserved_cells,
        primitive_step_budget=None,
        search_order="breadth_first",
    )


def delivery_action(source=(0, 1)):
    return RecoveryAction(
        kind=RecoveryActionKind.DELIVERY,
        block_label="B",
        source=source,
        destination=(0, 3),
        approach_path=((0, 0), source),
        transport_path=(source, (0, 2), (0, 3)),
    )


def reconfigure_action():
    return RecoveryAction(
        kind=RecoveryActionKind.RELOCATION,
        block_label="B",
        source=(0, 1),
        destination=(0, 2),
        approach_path=((0, 0), (0, 1)),
        transport_path=((0, 1), (0, 2)),
    )


def make_candidates(current):
    deliver = delivery_action()
    delivered = empty_state()
    relocated = state_with_block((0, 2), agent=(0, 2))
    accepted = replace(
        current,
        blocks=(
            BlockView("B", (0, 1), -2.0),
            BlockView("A", (1, 1), 8.0),
        ),
        agent_position=(1, 1),
    )
    return (
        ViabilityActionCandidate(
            key="accept:A:1:1",
            mode=ViabilityMode.ACCEPT,
            action_type=ViabilityActionType.ACCEPT,
            target_label="A",
            source=(1, 0),
            destination=(1, 1),
            successor_state=accepted,
            certificate=safe_certificate(accepted, (deliver, deliver)),
            option=DummyOption(),
            recovery_rank_before=1,
            recovery_rank_after=2,
        ),
        ViabilityActionCandidate(
            key="deliver:B:0:3",
            mode=ViabilityMode.RECOVER,
            action_type=ViabilityActionType.DELIVER,
            target_label="B",
            source=(0, 1),
            destination=(0, 3),
            successor_state=delivered,
            certificate=safe_certificate(delivered),
            option=DummyOption(),
            recovery_action=deliver,
            recovery_rank_before=1,
            recovery_rank_after=0,
        ),
        ViabilityActionCandidate(
            key="reconfigure:B:0:2",
            mode=ViabilityMode.RECOVER,
            action_type=ViabilityActionType.RECONFIGURE,
            target_label="B",
            source=(0, 1),
            destination=(0, 2),
            successor_state=relocated,
            certificate=safe_certificate(relocated, (delivery_action((0, 2)),)),
            option=DummyOption(),
            recovery_action=reconfigure_action(),
            recovery_rank_before=1,
            recovery_rank_after=1,
        ),
        ViabilityActionCandidate(
            key="defer:3",
            mode=ViabilityMode.DEFER,
            action_type=ViabilityActionType.DEFER,
            target_label=None,
            source=(0, 0),
            destination=None,
            successor_state=current,
            certificate=safe_certificate(current, (deliver,)),
            option=DummyOption(),
            horizon_steps=3,
            recovery_rank_before=1,
            recovery_rank_after=1,
        ),
    )


def make_audit(candidates, *, unsafe_count=0, search_order="breadth_first"):
    action_counts = {
        action_type: sum(
            candidate.action_type is action_type for candidate in candidates
        )
        for action_type in ViabilityActionType
    }
    return ViabilityCandidateAudit(
        interface="test_interface",
        boundary_contract="test_boundary",
        certification_contract="test_certification",
        verifier_authority="test_exact_verifier",
        search_order=search_order,
        post_accept_contract="test_post_accept",
        fixed_obstacle_contract="test_obstacles",
        liveness_rule="test_liveness",
        baseline_viability_teacher=False,
        terminal=False,
        current_recovery_status=ViabilityStatus.SAFE,
        current_recovery_rank=(1 if search_order == "breadth_first" else None),
        recovery_rank_exact=search_order == "breadth_first",
        physical_accept_count=action_counts[ViabilityActionType.ACCEPT]
        + unsafe_count,
        executable_accept_count=action_counts[ViabilityActionType.ACCEPT]
        + unsafe_count,
        accept_executor_rejections=0,
        safe_accept_count=action_counts[ViabilityActionType.ACCEPT],
        unsafe_accept_count=unsafe_count,
        unknown_accept_count=0,
        legal_recovery_count=action_counts[ViabilityActionType.DELIVER]
        + action_counts[ViabilityActionType.RECONFIGURE],
        recovery_executor_rejections=0,
        safe_recovery_count=action_counts[ViabilityActionType.DELIVER]
        + action_counts[ViabilityActionType.RECONFIGURE],
        unsafe_recovery_count=0,
        unknown_recovery_count=0,
        defer_allowed=bool(action_counts[ViabilityActionType.DEFER]),
        defer_horizon_steps=(
            3 if action_counts[ViabilityActionType.DEFER] else None
        ),
        defer_reason="test",
        consecutive_defer_decisions=0,
        candidate_count=len(candidates),
        accept_candidate_count=action_counts[ViabilityActionType.ACCEPT],
        deliver_candidate_count=action_counts[ViabilityActionType.DELIVER],
        reconfigure_candidate_count=action_counts[
            ViabilityActionType.RECONFIGURE
        ],
        defer_candidate_count=action_counts[ViabilityActionType.DEFER],
        rank_reducing_recovery_count=action_counts[
            ViabilityActionType.DELIVER
        ],
        cache_hits=0,
        cache_misses=0,
        cache_entries=0,
        analysis_seconds=0.0,
    )


def make_snapshot(*, search_order="breadth_first", unsafe_count=0):
    current = state_with_block()
    candidates = make_candidates(current)
    witness = (delivery_action(),)
    return ViabilityCandidateSnapshot(
        episode_instance_id="test-instance",
        decision_epoch=7,
        agent_position=current.agent_position,
        yard=current.to_yard_snapshot(),
        recovery_state=current,
        current_certificate=safe_certificate(
            current, witness, search_order=search_order
        ),
        inbound_label="A",
        candidates=candidates,
        audit=make_audit(
            candidates, unsafe_count=unsafe_count, search_order=search_order
        ),
    )


def make_relocated_snapshot():
    current = state_with_block((0, 2), agent=(0, 2))
    action = RecoveryAction(
        kind=RecoveryActionKind.DELIVERY,
        block_label="B",
        source=(0, 2),
        destination=(0, 3),
        approach_path=((0, 2),),
        transport_path=((0, 2), (0, 3)),
    )
    successor = empty_state()
    candidate = ViabilityActionCandidate(
        key="deliver:B:0:3",
        mode=ViabilityMode.RECOVER,
        action_type=ViabilityActionType.DELIVER,
        target_label="B",
        source=(0, 2),
        destination=(0, 3),
        successor_state=successor,
        certificate=safe_certificate(successor),
        option=DummyOption(),
        recovery_action=action,
        recovery_rank_before=1,
        recovery_rank_after=0,
    )
    candidates = (candidate,)
    return ViabilityCandidateSnapshot(
        episode_instance_id="test-instance",
        decision_epoch=12,
        agent_position=current.agent_position,
        yard=current.to_yard_snapshot(),
        recovery_state=current,
        current_certificate=safe_certificate(current, (action,)),
        inbound_label=None,
        candidates=candidates,
        audit=make_audit(candidates),
    )


def tiny_config(**overrides):
    values = dict(
        graph_hidden_dim=8,
        graph_embedding_dim=8,
        message_passing_steps=1,
        action_embedding_dim=8,
        head_hidden_dim=16,
        batch_size=2,
        replay_capacity=16,
        target_update_every=1,
        update_every=1,
    )
    values.update(overrides)
    return ViabilityGraphConfig(**values)


class RegularizedOperatorTests(unittest.TestCase):
    def test_nested_value_is_invariant_to_complete_within_mode_replication(self):
        temperatures = (0.4, 0.7, 0.9)
        original = cardinality_normalized_continuation(
            torch.tensor([2.0, -1.0, 0.5]),
            torch.tensor([0, 0, 1]),
            temperatures,
            0.8,
        )
        duplicated = cardinality_normalized_continuation(
            torch.tensor([2.0, -1.0, 2.0, -1.0, 0.5]),
            torch.tensor([0, 0, 0, 0, 1]),
            temperatures,
            0.8,
        )
        torch.testing.assert_close(original, duplicated)

        double = double_dqn_regularized_continuation(
            torch.tensor([2.0, -1.0, 0.5]),
            torch.tensor([2.0, -1.0, 0.5]),
            torch.tensor([0, 0, 1]),
            temperatures,
            0.8,
        )
        torch.testing.assert_close(original, double)

    def test_double_q_uses_online_policy_and_target_evaluation(self):
        value = double_dqn_regularized_continuation(
            torch.tensor([20.0, 0.0]),
            torch.tensor([3.0, 100.0]),
            torch.tensor([0, 0]),
            (0.01, 0.2, 0.2),
            0.5,
        )
        self.assertAlmostEqual(
            float(value), 3.0 - 0.01 * math.log(2.0), places=4
        )

    def test_common_smdp_target_uses_duration_for_every_mode(self):
        targets = common_smdp_target(
            torch.tensor([5.0, 5.0, 5.0]),
            torch.tensor([1, 2, 3]),
            torch.tensor([False, False, True]),
            torch.tensor([10.0, 10.0, 10.0]),
            gamma=0.5,
            reward_scale=1.0,
        )
        torch.testing.assert_close(targets, torch.tensor([10.0, 7.5, 5.0]))


class ViabilityGraphAgentTests(unittest.TestCase):
    def test_scores_every_counterfactual_graph_with_one_shared_head(self):
        snapshot = make_snapshot()
        agent = ViabilityGraphHierarchyAgent(
            config=tiny_config(), seed=3, epsilon=0.0
        )
        observed_batch_sizes = []
        handle = agent.Q_local.graph_encoder.register_forward_pre_hook(
            lambda _module, arguments: observed_batch_sizes.append(
                arguments[0].batch_size
            )
        )
        try:
            values = agent.score_snapshot(snapshot)
        finally:
            handle.remove()

        prepared = prepare_viability_snapshot(snapshot)
        self.assertEqual(values.shape, (4,))
        self.assertEqual(observed_batch_sizes, [8])
        self.assertTrue(
            all(
                record.current_state == snapshot.recovery_state
                for record in prepared.records
            )
        )
        self.assertEqual(
            [record.successor_state for record in prepared.records],
            [candidate.successor_state for candidate in snapshot.candidates],
        )
        self.assertEqual(
            len(prepared.records[0].action_features), ACTION_FEATURE_DIM
        )
        self.assertEqual(
            prepared.records[0].mode_id, MODE_TO_ID["accept"]
        )

    def test_non_safe_foreign_candidate_is_never_scored_or_selected(self):
        snapshot = make_snapshot(unsafe_count=1)
        poisoned = snapshot.candidates[0]
        object.__setattr__(
            poisoned,
            "certificate",
            unsafe_certificate(poisoned.successor_state),
        )
        prepared = prepare_viability_snapshot(snapshot)
        self.assertEqual(prepared.interface_rejection_count, 1)
        self.assertNotIn(poisoned.key, [record.key for record in prepared.records])

        agent = ViabilityGraphHierarchyAgent(
            config=tiny_config(), seed=5, epsilon=1.0
        )
        selected = {
            agent.select(snapshot, training=True).candidate.key
            for _ in range(30)
        }
        self.assertNotIn(poisoned.key, selected)
        self.assertGreater(agent.audit()["interface_non_safe_rejections"], 0)

    def test_exact_witness_guard_forces_and_bellman_frontier_matches(self):
        snapshot = make_snapshot()
        agent = ViabilityGraphHierarchyAgent(
            config=tiny_config(max_nonprogress_recovery_decisions=1),
            seed=9,
            epsilon=1.0,
        )
        agent.recovery_witness_guard.nonprogress_recovery_decisions = 1

        decision = agent.select(snapshot, training=True)
        self.assertTrue(decision.liveness_forced)
        self.assertEqual(decision.candidate.action_type, ViabilityActionType.DELIVER)
        self.assertEqual(len(decision.prepared_snapshot.records), 1)
        self.assertEqual(decision.prepared_snapshot.exact_safe_candidate_count, 4)
        self.assertTrue(decision.prepared_snapshot.liveness_restricted)
        self.assertEqual(decision.selection_source, "exact_recovery_witness_guard")

        # A nonterminal physical successor with remaining witness work would
        # be singleton-restricted in replay.  This one-step witness terminates.
        agent.remember(
            decision,
            reward=2.0,
            duration=delivery_action().steps,
            next_snapshot=None,
            done=True,
        )
        guard_audit = agent.audit()["recovery_witness_guard_audit"]
        self.assertEqual(guard_audit["forced_decisions"], 1)
        self.assertEqual(guard_audit["completed_witnesses"], 1)

    def test_retained_multistep_witness_restricts_replayed_next_frontier(self):
        snapshot = make_snapshot()
        relocated_snapshot = make_relocated_snapshot()
        second_action = relocated_snapshot.current_certificate.witness[0]
        first_action = reconfigure_action()
        candidates = tuple(
            replace(
                candidate,
                recovery_rank_before=None,
                recovery_rank_after=None,
            )
            for candidate in snapshot.candidates
        )
        snapshot = replace(
            snapshot,
            current_certificate=safe_certificate(
                snapshot.recovery_state,
                (first_action, second_action),
                search_order="goal_directed",
            ),
            candidates=candidates,
            audit=replace(
                snapshot.audit,
                search_order="goal_directed",
                current_recovery_rank=None,
                recovery_rank_exact=False,
            ),
        )
        agent = ViabilityGraphHierarchyAgent(
            config=tiny_config(max_nonprogress_recovery_decisions=1),
            seed=2,
            epsilon=1.0,
        )
        agent.recovery_witness_guard.nonprogress_recovery_decisions = 1
        first = agent.select(snapshot, training=True)
        self.assertEqual(
            first.candidate.action_type, ViabilityActionType.RECONFIGURE
        )

        observed = agent.observe_outcome(
            first, next_snapshot=relocated_snapshot, done=False
        )
        self.assertIsNotNone(observed)
        self.assertTrue(observed.liveness_restricted)
        self.assertEqual(len(observed.records), 1)
        self.assertEqual(observed.records[0].key, "deliver:B:0:3")
        agent.remember(
            first,
            reward=-0.5,
            duration=first_action.steps,
            next_snapshot=relocated_snapshot,
            done=False,
            outcome_already_observed=True,
        )
        replayed = agent.replay.memory[-1]
        self.assertEqual(len(replayed.next_candidates), 1)
        self.assertEqual(replayed.next_candidates[0].key, "deliver:B:0:3")
        self.assertEqual(
            replayed.next_candidates[0].action_features[-1], 1.0
        )

        second = agent.select(relocated_snapshot, training=True)
        self.assertTrue(second.liveness_forced)
        self.assertEqual(second.candidate.action_type, ViabilityActionType.DELIVER)

    def test_goal_directed_witness_length_is_not_treated_as_exact_rank(self):
        snapshot = make_snapshot(search_order="goal_directed")
        agent = ViabilityGraphHierarchyAgent(
            config=tiny_config(max_nonprogress_recovery_decisions=10),
            seed=1,
            epsilon=0.0,
        )
        # Force a deterministic selected record without relying on learned Q.
        with torch.no_grad():
            for parameter in agent.Q_local.parameters():
                parameter.zero_()
        decision = agent.select(snapshot, training=False)
        self.assertFalse(decision.exact_rank_progress)
        self.assertFalse(
            agent.checkpoint_metadata()[
                "goal_directed_witness_length_used_as_exact_rank"
            ]
        )

    def test_learning_and_resumable_checkpoint_round_trip(self):
        snapshot = make_snapshot()
        agent = ViabilityGraphHierarchyAgent(
            config=tiny_config(), seed=13, epsilon=0.0
        )
        first = agent.select(snapshot, training=False)
        agent.remember(
            first,
            reward=4.0,
            duration=2,
            next_snapshot=snapshot,
            done=False,
        )
        second = agent.select(snapshot, training=False)
        agent.remember(
            second,
            reward=-1.0,
            duration=1,
            next_snapshot=None,
            done=True,
        )
        loss = agent.learn()
        self.assertIsNotNone(loss)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(agent.gradient_steps, 1)
        self.assertEqual(agent.target_updates, 1)

        payload = agent.checkpoint(include_replay=True, run="unit-test")
        metadata = agent.checkpoint_metadata()
        self.assertTrue(metadata["exact_safe_mask_authoritative"])
        self.assertTrue(metadata["common_smdp_continuation"])
        self.assertTrue(metadata["double_dqn"])
        self.assertFalse(metadata["baseline_teacher"])
        self.assertFalse(metadata["behavior_cloning"])
        restored = ViabilityGraphHierarchyAgent.from_checkpoint(
            payload, resumable=True, seed=99
        )
        self.assertEqual(len(restored.replay), 2)
        torch.testing.assert_close(
            agent.score_snapshot(snapshot), restored.score_snapshot(snapshot)
        )

    def test_eval_outcome_api_and_episode_reset_do_not_write_replay(self):
        snapshot = make_snapshot()
        agent = ViabilityGraphHierarchyAgent(
            config=tiny_config(max_nonprogress_recovery_decisions=1),
            seed=4,
            epsilon=0.0,
        )
        agent.recovery_witness_guard.nonprogress_recovery_decisions = 1
        decision = agent.select(snapshot, training=False)
        agent.observe_outcome(decision, next_snapshot=None, done=True)
        self.assertEqual(len(agent.replay), 0)
        agent.recovery_witness_guard.nonprogress_recovery_decisions = 1
        agent.reset_episode_state()
        self.assertEqual(
            agent.recovery_witness_guard.nonprogress_recovery_decisions, 0
        )


if __name__ == "__main__":
    unittest.main()
