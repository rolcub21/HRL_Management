import math
import unittest

import torch

from tests.test_viability_graph_hierarchy import make_snapshot
from viability_graph_episodic_audit import (
    EPISODIC_TIME_CONTEXT_CONTRACT,
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION,
    EpisodicViabilityGraphConfig,
    EpisodicViabilityGraphHierarchyAgent,
    episodic_common_smdp_target,
)
from viability_graph_hierarchy import (
    VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION,
    ViabilityGraphConfig,
    ViabilityGraphHierarchyAgent,
    common_smdp_target,
)


def tiny_episodic_config(**overrides):
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
        episode_horizon_steps=100,
    )
    values.update(overrides)
    return EpisodicViabilityGraphConfig(**values)


class EpisodicTargetTests(unittest.TestCase):
    def test_gamma_one_uses_undiscounted_nonterminal_continuation(self):
        target = episodic_common_smdp_target(
            torch.tensor([5.0, 5.0, -2.0]),
            torch.tensor([1, 7, 11]),
            torch.tensor([False, False, True]),
            torch.tensor([10.0, 10.0, 10.0]),
            gamma=1.0,
            reward_scale=1.0,
        )
        torch.testing.assert_close(target, torch.tensor([15.0, 15.0, -2.0]))

    def test_discounted_path_is_numerically_identical_to_v1(self):
        rewards = torch.tensor([4.5, -0.25, 8.0], dtype=torch.float64)
        durations = torch.tensor([1, 6, 13])
        dones = torch.tensor([False, True, False])
        continuation = torch.tensor([2.0, 20.0, -3.0], dtype=torch.float64)
        expected = common_smdp_target(
            rewards,
            durations,
            dones,
            continuation,
            gamma=0.99,
            reward_scale=0.01,
        )
        observed = episodic_common_smdp_target(
            rewards,
            durations,
            dones,
            continuation,
            gamma=0.99,
            reward_scale=0.01,
        )
        self.assertTrue(torch.equal(expected, observed))

    def test_gamma_validation_distinguishes_v1_from_episodic_audit(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
            ViabilityGraphConfig(gamma=1.0)
        config = EpisodicViabilityGraphConfig(gamma=1.0)
        self.assertEqual(config.gamma, 1.0)
        for invalid in (-0.1, 1.01, math.inf, math.nan):
            with self.subTest(gamma=invalid):
                with self.assertRaises(ValueError):
                    EpisodicViabilityGraphConfig(gamma=invalid)


class EpisodicAgentIsolationTests(unittest.TestCase):
    def test_network_initialization_and_scores_match_v1_when_gamma_matches(self):
        episodic_config = tiny_episodic_config(gamma=0.99)
        v1_values = {
            name: getattr(episodic_config, name)
            for name in ViabilityGraphConfig.__dataclass_fields__
        }
        original = ViabilityGraphHierarchyAgent(
            config=ViabilityGraphConfig(**v1_values),
            seed=17,
            epsilon=0.0,
        )
        audit = EpisodicViabilityGraphHierarchyAgent(
            config=episodic_config,
            seed=17,
            epsilon=0.0,
        )
        original_state = original.Q_local.state_dict()
        audit_state = audit.Q_local.state_dict()
        self.assertEqual(original_state.keys(), audit_state.keys())
        for name in original_state:
            torch.testing.assert_close(original_state[name], audit_state[name])
        torch.testing.assert_close(
            original.score_snapshot(make_snapshot()),
            audit.score_snapshot(make_snapshot()),
        )

    def test_checkpoint_families_are_mutually_incompatible(self):
        original = ViabilityGraphHierarchyAgent(
            config=ViabilityGraphConfig(
                graph_hidden_dim=8,
                graph_embedding_dim=8,
                message_passing_steps=1,
                action_embedding_dim=8,
                head_hidden_dim=16,
                batch_size=2,
                replay_capacity=16,
            ),
            seed=2,
            epsilon=0.0,
        )
        audit = EpisodicViabilityGraphHierarchyAgent(
            config=tiny_episodic_config(gamma=1.0),
            seed=2,
            epsilon=0.0,
        )
        original_payload = original.checkpoint(include_replay=False)
        audit_payload = audit.checkpoint(include_replay=False)
        self.assertEqual(
            original_payload["checkpoint_schema_version"],
            VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION,
        )
        self.assertEqual(
            audit_payload["checkpoint_schema_version"],
            EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION,
        )
        self.assertNotEqual(
            original_payload["controller_architecture"],
            audit_payload["controller_architecture"],
        )
        with self.assertRaisesRegex(
            ValueError, "incompatible viability graph checkpoint"
        ):
            ViabilityGraphHierarchyAgent.from_checkpoint(audit_payload)
        with self.assertRaisesRegex(
            ValueError, "incompatible episodic viability graph checkpoint"
        ):
            EpisodicViabilityGraphHierarchyAgent.from_checkpoint(
                original_payload
            )

    def test_gamma_one_checkpoint_round_trip_and_learning(self):
        snapshot = make_snapshot()
        agent = EpisodicViabilityGraphHierarchyAgent(
            config=tiny_episodic_config(gamma=1.0),
            seed=31,
            epsilon=0.0,
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

        payload = agent.checkpoint(include_replay=True, run="unit-test")
        self.assertTrue(payload["finite_episode_objective_audit"])
        self.assertEqual(payload["gamma"], 1.0)
        self.assertFalse(payload["gamma_one_is_infinite_horizon_contraction"])
        self.assertFalse(payload["remaining_budget_observed_by_q"])
        self.assertEqual(
            payload["time_context_contract"], EPISODIC_TIME_CONTEXT_CONTRACT
        )
        restored = EpisodicViabilityGraphHierarchyAgent.from_checkpoint(
            payload, resumable=True, seed=99
        )
        self.assertEqual(restored.config.gamma, 1.0)
        self.assertEqual(len(restored.replay), 2)
        torch.testing.assert_close(
            agent.score_snapshot(snapshot), restored.score_snapshot(snapshot)
        )

    def test_nonterminal_bootstrap_at_declared_horizon_is_rejected(self):
        snapshot = make_snapshot()
        agent = EpisodicViabilityGraphHierarchyAgent(
            config=tiny_episodic_config(
                gamma=1.0,
                episode_horizon_steps=snapshot.decision_epoch + 2,
            ),
            seed=8,
            epsilon=0.0,
        )
        decision = agent.select(snapshot, training=False)
        with self.assertRaisesRegex(ValueError, "must be terminal"):
            agent.remember(
                decision,
                reward=0.0,
                duration=2,
                next_snapshot=snapshot,
                done=False,
            )
        self.assertEqual(len(agent.replay), 0)

        # Marking the same horizon transition terminal is valid and creates no
        # continuation, as required by the episodic return definition.
        agent.remember(
            decision,
            reward=0.0,
            duration=2,
            next_snapshot=None,
            done=True,
        )
        self.assertTrue(agent.replay.memory[-1].done)
        self.assertFalse(agent.replay.memory[-1].next_candidates)

    def test_checkpoint_metadata_authenticates_no_representation_change(self):
        agent = EpisodicViabilityGraphHierarchyAgent(
            config=tiny_episodic_config(gamma=1.0),
            seed=3,
            epsilon=0.0,
        )
        metadata = agent.checkpoint_metadata()
        self.assertEqual(
            metadata["representation_difference_from_vcg_v1"], "none"
        )
        self.assertEqual(metadata["safe_frontier_difference_from_vcg_v1"], "none")
        self.assertEqual(metadata["policy_difference_from_vcg_v1"], "none")
        self.assertTrue(metadata["comparison_information_parity_required"])
        self.assertTrue(
            metadata["horizon_nonbinding_must_be_reported_empirically"]
        )


if __name__ == "__main__":
    unittest.main()
