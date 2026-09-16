from dataclasses import fields, replace
import math
import random
import unittest

import torch

from PSLAP.dynamic_yard import BlockView
from PSLAP.viability import RecoveryState
from PSLAP.viability_candidates import ViabilityActionType
from PSLAP.viability_candidates_hold_v2 import (
    CertifiedHoldCandidateSnapshotV2,
    CertifiedHoldRuleV2,
    PrimitiveIdleBudgetStateV2,
    enumerate_viability_candidates_hold_v2,
)
from PSLAP.viability_filter import ViabilitySearchConfig
import viability_graph_constrained_v2 as constrained_v2
from viability_graph_constrained_v2 import (
    ACTION_TYPE_TO_GROUP,
    CONSTRAINED_V2_CHECKPOINT_FAMILY,
    CONSTRAINED_V2_CONTROLLER_ARCHITECTURE,
    CONSTRAINED_V2_NETWORK_ARCHITECTURE,
    CONTROL_GROUP_NAMES,
    ConstrainedV2CandidateRecord,
    ConstrainedV2Config,
    ConstrainedV2HierarchyAgent,
    ConstrainedV2QNetwork,
    ConstrainedV2ReplayBuffer,
    ConstrainedV2Transition,
    V2_ACTION_FEATURE_DIM,
    constrained_component_targets,
    prepare_constrained_snapshot,
    shared_lagrangian_policy_values,
)
from viability_graph_hierarchy import (
    VIABILITY_GRAPH_CONTROLLER_ARCHITECTURE,
    VIABILITY_GRAPH_NETWORK_ARCHITECTURE,
    ViabilityGraphHierarchyAgent,
)


def _state(*, position=(0, 1), remaining_time=5.0):
    return RecoveryState(
        rows=2,
        cols=3,
        traversable=frozenset(
            (row, column) for row in range(2) for column in range(3)
        ),
        storage_cells=frozenset({(0, 1), (1, 1)}),
        exits=((0, 2),),
        blocks=(BlockView("B", position, remaining_time),),
        agent_position=(0, 0),
        pickup_cells=frozenset({(1, 0)}),
        wait_cells=frozenset({(1, 2)}),
    )


_ACTION_BY_GROUP = (
    ViabilityActionType.ACCEPT.value,
    ViabilityActionType.DELIVER.value,
    ViabilityActionType.RECONFIGURE.value,
    ViabilityActionType.DEFER.value,
)


def _record(group_id, suffix="0"):
    action_type = _ACTION_BY_GROUP[group_id]
    return ConstrainedV2CandidateRecord(
        key=f"{action_type}:{suffix}",
        group_id=group_id,
        action_type=action_type,
        current_state=_state(),
        successor_state=_state(position=(1, 1), remaining_time=4.0),
        action_features=(0.0,) * V2_ACTION_FEATURE_DIM,
        elapsed_steps=0,
        remaining_episode_steps=2_000,
        idle_budget_remaining_steps=20,
        idle_budget_max_steps=20,
    )


def _transition(group_id, suffix="0", *, physical_cost=0.0):
    return ConstrainedV2Transition(
        chosen=_record(group_id, suffix),
        operational_reward=10.0 + group_id,
        physical_rehandle_cost=physical_cost,
        duration=group_id + 1,
        next_candidates=(),
        done=True,
        behavior_lambda=0.25,
    )


def _manual_nested_weights(
    online_op, online_cost, groups, taus, tau_group, lam, op_weight=1.0
):
    """Independent, explicit expansion of the two-stage policy weights."""

    merit = op_weight * online_op - lam * online_cost
    group_merits = []
    live_groups = []
    within = {}
    for group_id in range(len(CONTROL_GROUP_NAMES)):
        indices = torch.nonzero(groups == group_id, as_tuple=False).flatten()
        if not len(indices):
            continue
        group_tau = taus[group_id]
        group_values = merit[indices]
        within[group_id] = (indices, torch.softmax(group_values / group_tau, 0))
        group_merits.append(
            group_tau
            * (
                torch.logsumexp(group_values / group_tau, 0)
                - math.log(len(indices))
            )
        )
        live_groups.append(group_id)
    group_weights = torch.softmax(torch.stack(group_merits) / tau_group, 0)
    flat = torch.zeros_like(online_op)
    for outer_weight, group_id in zip(group_weights, live_groups):
        indices, inner_weights = within[group_id]
        flat[indices] = outer_weight * inner_weights
    return flat, torch.stack(group_merits), torch.tensor(live_groups)


class SharedVectorBackupTests(unittest.TestCase):
    def test_elapsed_weight_is_required_for_the_mixed_discount_lagrangian(self):
        operational = torch.tensor([10.0, 0.0])
        physical = torch.tensor([1.0, 0.0])
        groups = torch.tensor([0, 1])
        early = shared_lagrangian_policy_values(
            operational,
            physical,
            operational,
            physical,
            groups,
            (0.1, 0.1, 0.1, 0.1),
            0.1,
            6.0,
            operational_weight=1.0,
        )
        late = shared_lagrangian_policy_values(
            operational,
            physical,
            operational,
            physical,
            groups,
            (0.1, 0.1, 0.1, 0.1),
            0.1,
            6.0,
            operational_weight=0.5,
        )
        self.assertEqual(int(early[3][torch.argmax(early[2])]), 0)
        self.assertEqual(int(late[3][torch.argmax(late[2])]), 1)

    def test_both_target_heads_are_evaluated_under_one_online_policy(self):
        online_op = torch.tensor([0.0, 2.0, 1.0, 3.0, 2.5, 2.5, -2.0, -1.0])
        online_cost = torch.tensor([0.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 0.0])
        target_op = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        target_cost = torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        groups = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        taus = torch.tensor([0.5, 0.7, 0.9, 1.1])
        tau_group = 0.8
        dual_lambda = 0.5

        op_value, cost_value, group_merits, live_groups = (
            shared_lagrangian_policy_values(
                online_op,
                online_cost,
                target_op,
                target_cost,
                groups,
                taus,
                tau_group,
                dual_lambda,
                operational_weight=1.0,
            )
        )
        weights, expected_merits, expected_live = _manual_nested_weights(
            online_op,
            online_cost,
            groups,
            taus,
            tau_group,
            dual_lambda,
            1.0,
        )

        torch.testing.assert_close(op_value, torch.sum(weights * target_op))
        torch.testing.assert_close(cost_value, torch.sum(weights * target_cost))
        torch.testing.assert_close(group_merits, expected_merits)
        torch.testing.assert_close(live_groups.cpu(), expected_live)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

        # Target-network estimates evaluate the policy; they must not choose it.
        changed = shared_lagrangian_policy_values(
            online_op,
            online_cost,
            target_op.flip(0),
            target_cost.square(),
            groups,
            taus,
            tau_group,
            dual_lambda,
            operational_weight=1.0,
        )
        torch.testing.assert_close(changed[2], group_merits)
        torch.testing.assert_close(changed[3], live_groups)

    def test_four_live_groups_and_cardinality_normalization(self):
        self.assertEqual(
            tuple(ACTION_TYPE_TO_GROUP[action] for action in _ACTION_BY_GROUP),
            (0, 1, 2, 3),
        )
        online_op = torch.tensor([1.0, 2.0, 3.0, 4.0])
        online_cost = torch.zeros(4)
        target_op = torch.tensor([10.0, 20.0, 30.0, 40.0])
        target_cost = torch.tensor([4.0, 3.0, 2.0, 1.0])
        groups = torch.arange(4)
        temperatures = (0.2, 0.3, 0.4, 0.5)

        base = shared_lagrangian_policy_values(
            online_op,
            online_cost,
            target_op,
            target_cost,
            groups,
            temperatures,
            0.7,
            0.0,
            operational_weight=1.0,
        )
        self.assertEqual(tuple(base[3].tolist()), (0, 1, 2, 3))
        self.assertEqual(int(base[3][torch.argmax(base[2])]), 3)

        # Replicating an identical candidate inside one group must not give that
        # temporal group extra probability merely because its set is larger.
        duplicate_indices = torch.tensor([0, 1, 2, 2, 2, 3])
        duplicated = shared_lagrangian_policy_values(
            online_op[duplicate_indices],
            online_cost[duplicate_indices],
            target_op[duplicate_indices],
            target_cost[duplicate_indices],
            groups[duplicate_indices],
            temperatures,
            0.7,
            0.0,
            operational_weight=1.0,
        )
        for actual, expected in zip(duplicated[:3], base[:3]):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(duplicated[3], base[3])

    def test_vector_targets_use_separate_discounts_without_scalarization(self):
        operational, physical = constrained_component_targets(
            operational_rewards=torch.tensor([10.0, 20.0]),
            rehandle_costs=torch.tensor([2.0, 3.0]),
            durations=torch.tensor([1, 3]),
            dones=torch.tensor([False, True]),
            next_operational_values=torch.tensor([4.0, 5.0]),
            next_rehandle_values=torch.tensor([7.0, 11.0]),
            gamma_operational=0.9,
            gamma_rehandle=1.0,
            operational_reward_scale=0.01,
        )
        torch.testing.assert_close(operational, torch.tensor([3.7, 0.2]))
        torch.testing.assert_close(physical, torch.tensor([9.0, 3.0]))


class VectorReplayAndCostTests(unittest.TestCase):
    def test_transition_fails_closed_on_negative_or_nonfinite_physical_cost(self):
        for invalid in (-1.0, 0.5, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "physical cost"):
                    _transition(2, physical_cost=invalid)

    def test_replay_preserves_raw_vector_fields_and_balances_four_groups(self):
        replay = ConstrainedV2ReplayBuffer(capacity=16)
        for group_id in range(4):
            replay.add(
                _transition(
                    group_id,
                    suffix="a",
                    physical_cost=float(group_id),
                )
            )
            replay.add(
                _transition(
                    group_id,
                    suffix="b",
                    physical_cost=float(group_id + 1),
                )
            )

        self.assertEqual(replay.group_counts(), {0: 2, 1: 2, 2: 2, 3: 2})
        sample = replay.sample_group_balanced(4, random.Random(7))
        self.assertEqual({item.chosen.group_id for item in sample}, {0, 1, 2, 3})
        self.assertTrue(
            all(math.isfinite(item.operational_reward) for item in sample)
        )
        self.assertTrue(all(item.physical_rehandle_cost >= 0.0 for item in sample))
        self.assertNotIn(
            "scalarized_reward",
            {item.name for item in fields(ConstrainedV2Transition)},
        )

        restored = ConstrainedV2ReplayBuffer(capacity=16)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(restored.group_counts(), replay.group_counts())
        self.assertEqual(
            [item.physical_rehandle_cost for item in restored.memory],
            [item.physical_rehandle_cost for item in replay.memory],
        )
        with self.assertRaisesRegex(ValueError, "capacity mismatch"):
            ConstrainedV2ReplayBuffer(capacity=15).load_state_dict(
                replay.state_dict()
            )

    def test_network_rehandle_head_is_structurally_nonnegative(self):
        config = ConstrainedV2Config(
            graph_hidden_dim=8,
            graph_embedding_dim=8,
            message_passing_steps=1,
            action_embedding_dim=4,
            head_hidden_dim=8,
            batch_size=2,
            replay_capacity=8,
        )
        network = ConstrainedV2QNetwork(config)
        current = (_state(), _state(remaining_time=-3.0))
        successor = (
            _state(position=(1, 1)),
            _state(position=(1, 1), remaining_time=-2.0),
        )
        features = torch.zeros((2, V2_ACTION_FEATURE_DIM))

        with torch.no_grad():
            network.operational_head.weight.zero_()
            network.operational_head.bias.fill_(-3.0)
            network.rehandle_head.weight.zero_()
            network.rehandle_head.bias.fill_(-100.0)
        operational, rehandle = network(current, successor, features)
        self.assertTrue(bool(torch.isfinite(rehandle).all()))
        self.assertTrue(bool((rehandle >= 0.0).all()))
        torch.testing.assert_close(operational, torch.full((2,), -3.0))

        with torch.no_grad():
            network.rehandle_head.bias.fill_(100.0)
        _, large_rehandle = network(current, successor, features)
        self.assertTrue(bool((large_rehandle >= 0.0).all()))


class CheckpointIsolationContractTests(unittest.TestCase):
    def test_v2_identity_is_distinct_and_v1_loader_rejects_it(self):
        self.assertNotEqual(
            CONSTRAINED_V2_CONTROLLER_ARCHITECTURE,
            VIABILITY_GRAPH_CONTROLLER_ARCHITECTURE,
        )
        self.assertNotEqual(
            CONSTRAINED_V2_NETWORK_ARCHITECTURE,
            VIABILITY_GRAPH_NETWORK_ARCHITECTURE,
        )
        self.assertEqual(
            CONSTRAINED_V2_CHECKPOINT_FAMILY,
            "vcg_constrained_vector_smdp_v2",
        )
        with self.assertRaisesRegex(
            ValueError, "incompatible viability graph checkpoint"
        ):
            ViabilityGraphHierarchyAgent.from_checkpoint(
                {
                    "checkpoint_schema_version": 1,
                    "checkpoint_family": CONSTRAINED_V2_CHECKPOINT_FAMILY,
                    "controller_architecture": CONSTRAINED_V2_CONTROLLER_ARCHITECTURE,
                    "network_architecture": CONSTRAINED_V2_NETWORK_ARCHITECTURE,
                }
            )

    @unittest.skipUnless(
        hasattr(constrained_v2, "ConstrainedV2HierarchyAgent"),
        "V2 agent/checkpoint loader has not landed in the core module",
    )
    def test_v2_loader_rejects_v1_checkpoint_identity(self):
        agent_class = constrained_v2.ConstrainedV2HierarchyAgent
        with self.assertRaisesRegex(ValueError, "incompatible"):
            agent_class.from_checkpoint(
                {
                    "checkpoint_schema_version": 1,
                    "controller_architecture": VIABILITY_GRAPH_CONTROLLER_ARCHITECTURE,
                    "network_architecture": VIABILITY_GRAPH_NETWORK_ARCHITECTURE,
                },
                device="cpu",
                resumable=False,
                seed=0,
            )

    def test_v2_vector_learning_and_checkpoint_round_trip(self):
        config = ConstrainedV2Config(
            graph_hidden_dim=8,
            graph_embedding_dim=8,
            message_passing_steps=1,
            action_embedding_dim=4,
            head_hidden_dim=8,
            batch_size=4,
            replay_capacity=8,
            target_update_every=1,
        )
        agent = ConstrainedV2HierarchyAgent(
            config=config, seed=13, device="cpu", epsilon=0.0
        )
        for group_id in range(4):
            agent.replay.add(_transition(group_id, physical_cost=float(group_id)))
        agent.transition_count = 4
        result = agent.learn()
        self.assertIsNotNone(result)
        self.assertEqual(agent.gradient_steps, 1)
        self.assertEqual(agent.target_updates, 1)
        payload = agent.checkpoint(include_replay=True)
        restored = ConstrainedV2HierarchyAgent.from_checkpoint(
            payload, device="cpu", resumable=True, seed=13
        )
        self.assertEqual(restored.replay.group_counts(), agent.replay.group_counts())
        self.assertEqual(restored.dual_lambda, agent.dual_lambda)

        incompatible_values = {
            "checkpoint_schema_version": 999,
            "checkpoint_family": "vcg_v1",
            "controller_architecture": "wrong_controller",
            "network_architecture": "wrong_network",
            "replay_version": "scalar_replay",
            "backup_version": "independent_head_backups",
            "policy_version": "flat_union_argmax",
            "candidate_interface": "base_v1_frontier",
            "cost_definition": "named_action_count",
            "single_shared_lagrangian_policy_for_both_target_heads": False,
            "entropy_or_kl_added_to_raw_component_targets": True,
            "scalarized_reward_stored_in_replay": True,
        }
        for field, incompatible in incompatible_values.items():
            with self.subTest(field=field):
                corrupted = dict(payload)
                corrupted[field] = incompatible
                with self.assertRaisesRegex(
                    ValueError, "incompatible constrained V2 checkpoint"
                ):
                    ConstrainedV2HierarchyAgent.from_checkpoint(
                        corrupted,
                        device="cpu",
                        resumable=False,
                        seed=13,
                    )


class HoldContextFailClosedTests(unittest.TestCase):
    def test_prepare_requires_authenticated_hold_and_exact_context(self):
        from tests.test_viability_candidates_hold_v2 import compact_env, make_stored

        env = compact_env(1, seed=7811)
        make_stored(env.blocks[0], (2, 2), duration=30)
        env.current_state = (1, 1)
        env.time_steps = 5
        snapshot = enumerate_viability_candidates_hold_v2(
            env,
            idle_budget=PrimitiveIdleBudgetStateV2(20, 3),
            remaining_episode_steps=7,
            recovery_witness_forced=False,
            search_config=ViabilitySearchConfig(
                max_nodes=None, search_order="breadth_first"
            ),
            hold_rule=CertifiedHoldRuleV2(10, 20),
        )
        config = ConstrainedV2Config(episode_horizon_steps=12)
        prepared = prepare_constrained_snapshot(snapshot, config=config)
        self.assertTrue(prepared.records)
        self.assertTrue(all(record.elapsed_steps == 5 for record in prepared.records))
        with self.assertRaisesRegex(TypeError, "certified-Hold"):
            prepare_constrained_snapshot(snapshot.frontier, config=config)

        bad = CertifiedHoldCandidateSnapshotV2(
            frontier=snapshot.frontier,
            hold_audit=replace(snapshot.hold_audit, remaining_episode_steps=6),
        )
        with self.assertRaisesRegex(ValueError, "remaining horizon"):
            prepare_constrained_snapshot(bad, config=config)


if __name__ == "__main__":
    unittest.main()
