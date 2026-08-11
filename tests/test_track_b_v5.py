import unittest

import numpy as np
import torch

from example.Options.selector_v5 import StorageSelectOptionV5
from example.small_rooms_env import SmallRoomsEnv
from gated_agent import (
    GatedModeOnlyAgent,
    GatedModeOnlyFoundationAgent,
    masked_max,
    mode_only_temporal_value,
    normalized_logsumexp,
    temporal_mode_value,
)
from options_agent import (
    LAYER_NORM,
    QNetwork,
    controller_ids,
    resolve_q_network_config,
)
from PSLAP.dynamic_yard import BlockView, YardSnapshot
from PSLAP.reg_selector_v5 import (
    REGV5AssignmentSource,
    REGV5Config,
    build_assignment_observation,
)
from PSLAP.track_a import shared_candidate_mask


class TrackBV5AdapterTests(unittest.TestCase):
    def make_env(self):
        return SmallRoomsEnv(
            choose_storage=False,
            arrival_rate=0.5,
            proc_mean=50,
        )

    @staticmethod
    def context(env):
        block = next(
            item
            for item in env.blocks
            if item.position == env.pickup_cell and not item.stored
        )
        yard = YardSnapshot.from_env(env)
        view = BlockView(
            block.label,
            block.position,
            float(block.get_remaining_storage_time()),
        )
        candidates = shared_candidate_mask(yard, view, block.position)
        return block, yard, view, candidates

    def test_assignment_phase_is_identical_before_and_after_pickup(self):
        track_a = self.make_env()
        instance = track_a.sample_episode_instance(123)
        track_a.reset(instance=instance)
        block_a, yard_a, view_a, candidates_a = self.context(track_a)
        observation_a = build_assignment_observation(
            track_a, yard_a, view_a, block_a.position, candidates_a
        )

        track_b = self.make_env()
        track_b.reset(instance=instance)
        block_b, yard_b, _, candidates_b = self.context(track_b)
        block_b.carrying = True
        track_b.current_state = block_b.position
        view_b = BlockView(
            block_b.label,
            block_b.position,
            float(block_b.get_remaining_storage_time()),
        )
        observation_b = build_assignment_observation(
            track_b, yard_b, view_b, block_b.position, candidates_b
        )

        np.testing.assert_array_equal(
            observation_a.block_features, observation_b.block_features
        )
        np.testing.assert_array_equal(
            observation_a.current_block_features,
            observation_b.current_block_features,
        )
        np.testing.assert_array_equal(
            observation_a.global_features, observation_b.global_features
        )
        np.testing.assert_array_equal(
            observation_a.candidate_features,
            observation_b.candidate_features,
        )

    def test_adapter_preserves_controller_id_and_commits_masked_cell(self):
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(9))
        block, _, _, _ = self.context(env)
        block.carrying = True
        env.current_state = block.position
        config = REGV5Config(
            block_embedding_dim=8,
            candidate_embedding_dim=8,
            context_dim=16,
            batch_size=2,
            min_replay_size=2,
        )
        source = REGV5AssignmentSource(
            env, config, learning_enabled=False, seed=0
        )
        option = StorageSelectOptionV5(env, source)
        self.assertEqual(controller_ids([option]), ["option:StorageSelectOption"])
        self.assertTrue(option.initiation(env.get_current_state()))
        action = option.policy(env.get_current_state())
        self.assertEqual(action, env.ACTION_IDS["WAIT"])
        self.assertIsNotNone(block.storage_location)
        self.assertEqual(option.invalid_assignment_count, 0)
        self.assertEqual(option.infeasible_epoch_count, 0)
        self.assertEqual(option.decision_count, 1)

    def test_primitive_mask_excludes_illegal_pickup_and_putdown(self):
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(17))
        pickup = env.ACTION_IDS["PICKUP"]
        putdown = env.ACTION_IDS["PUTDOWN"]
        self.assertNotIn(pickup, env.get_available_actions(env.get_current_state()))
        self.assertNotIn(putdown, env.get_available_actions(env.get_current_state()))

        block, _, _, _ = self.context(env)
        env.current_state = env.pickup_cell
        self.assertIn(pickup, env.get_available_actions(env.get_current_state()))
        block.carrying = True
        self.assertNotIn(pickup, env.get_available_actions(env.get_current_state()))
        self.assertNotIn(putdown, env.get_available_actions(env.get_current_state()))

        block.storage_location = env.storage_positions[0]
        block.position = block.storage_location
        env.current_state = block.storage_location
        self.assertIn(putdown, env.get_available_actions(env.get_current_state()))


class TemporalModeOperatorTests(unittest.TestCase):
    def test_uniform_replication_is_cardinality_invariant(self):
        values = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
        mask = torch.ones_like(values, dtype=torch.bool)
        base = normalized_logsumexp(values, mask, 0.7)
        repeated = normalized_logsumexp(
            values.repeat(1, 4),
            mask.repeat(1, 4),
            0.7,
        )
        torch.testing.assert_close(base, repeated, atol=1e-12, rtol=0)

    def test_unavailable_option_mode_does_not_change_primitive_value(self):
        option = torch.tensor([[8.0, 9.0]])
        primitive = torch.tensor([[1.0, 3.0]])
        option_mask = torch.zeros_like(option, dtype=torch.bool)
        primitive_mask = torch.ones_like(primitive, dtype=torch.bool)
        value, primitive_mode, _, _ = temporal_mode_value(
            option,
            primitive,
            option_mask,
            primitive_mask,
            0.1,
            0.1,
            1.0,
        )
        torch.testing.assert_close(value, primitive_mode)


class ModeOnlyTemporalOperatorTests(unittest.TestCase):
    def test_masked_max_ignores_duplicates_and_lower_valued_controls(self):
        base_values = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
        base_mask = torch.ones_like(base_values, dtype=torch.bool)
        expanded_values = torch.tensor(
            [[1.0, 3.0, 3.0, -100.0]], dtype=torch.float64
        )
        expanded_mask = torch.ones_like(expanded_values, dtype=torch.bool)
        torch.testing.assert_close(
            masked_max(base_values, base_mask),
            masked_max(expanded_values, expanded_mask),
        )

    def test_mode_only_value_matches_max_within_soft_across_modes(self):
        option = torch.tensor([[2.0, 5.0]], dtype=torch.float64)
        primitive = torch.tensor([[1.0, 3.0, 4.0]], dtype=torch.float64)
        option_mask = torch.tensor([[True, True]])
        primitive_mask = torch.tensor([[True, False, True]])
        value, primitive_mode, option_mode, mode_mask = (
            mode_only_temporal_value(
                option,
                primitive,
                option_mask,
                primitive_mask,
                tau_mode=0.7,
            )
        )
        torch.testing.assert_close(
            option_mode, torch.tensor([[5.0]], dtype=torch.float64)
        )
        torch.testing.assert_close(
            primitive_mode, torch.tensor([[4.0]], dtype=torch.float64)
        )
        expected = normalized_logsumexp(
            torch.tensor([[4.0, 5.0]], dtype=torch.float64),
            torch.tensor([[True, True]]),
            0.7,
        )
        torch.testing.assert_close(value, expected)
        self.assertTrue(bool(mode_mask.all()))

    def test_unavailable_mode_reduces_to_available_mode_max(self):
        option = torch.tensor([[50.0, 60.0]])
        primitive = torch.tensor([[1.0, 4.0]])
        value, primitive_mode, option_mode, _ = mode_only_temporal_value(
            option,
            primitive,
            torch.zeros_like(option, dtype=torch.bool),
            torch.ones_like(primitive, dtype=torch.bool),
            tau_mode=1.0,
        )
        torch.testing.assert_close(value, primitive_mode)
        torch.testing.assert_close(option_mode, torch.zeros_like(option_mode))

    def test_mode_only_policy_rejects_within_mode_sampling(self):
        agent = object.__new__(GatedModeOnlyAgent)
        agent.set_policy_realization("mode_regularized")
        self.assertTrue(agent.sample_regularized_mode)
        self.assertFalse(agent.sample_regularized_controls)
        with self.assertRaisesRegex(ValueError, "map or mode_regularized"):
            agent.set_policy_realization("regularized")


class ControllerNetworkContractTests(unittest.TestCase):
    def test_layer_norm_is_rank_and_batch_companion_invariant(self):
        network = QNetwork(12, 5, seed=7, normalization=LAYER_NORM)
        x = torch.linspace(-1.0, 1.0, 12)
        companion = torch.linspace(3.0, -2.0, 12)
        for training in (True, False):
            network.train(training)
            single = network(x)
            batch_one = network(x.unsqueeze(0))[0]
            with_companion = network(torch.stack((x, companion)))[0]
            torch.testing.assert_close(single, batch_one)
            torch.testing.assert_close(single, with_companion)

    def test_explicit_initialization_is_seeded_and_rng_isolated(self):
        torch.manual_seed(1234)
        before = torch.get_rng_state().clone()
        first = QNetwork(8, 3, seed=11, normalization=LAYER_NORM)
        after = torch.get_rng_state().clone()
        second = QNetwork(8, 3, seed=11, normalization=LAYER_NORM)
        different = QNetwork(8, 3, seed=12, normalization=LAYER_NORM)
        self.assertTrue(torch.equal(before, after))
        for left, right in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(left, right)
        self.assertTrue(
            any(
                not torch.equal(left, right)
                for left, right in zip(first.parameters(), different.parameters())
            )
        )

    def test_foundation_defaults_are_recorded(self):
        agent = object.__new__(GatedModeOnlyFoundationAgent)
        agent.q_network_normalization = LAYER_NORM
        agent.initialization_seed = 4
        agent.wait_training_penalty = 0.0
        agent.terminal_on_truncation = True
        agent.close_options_on_episode_end = True
        agent.tau_option = 0.1
        agent.tau_primitive = 0.1
        agent.tau_mode = 1.0
        agent.policy_realization = "mode_regularized"
        metadata = agent.checkpoint_metadata()
        self.assertEqual(metadata["q_network_normalization"], LAYER_NORM)
        self.assertEqual(metadata["manager_initialization_seed"], 4)
        self.assertEqual(metadata["worker_initialization_seed"], 5)
        self.assertEqual(metadata["wait_training_penalty"], 0.0)
        self.assertTrue(metadata["terminal_on_truncation"])

    def test_missing_network_metadata_resolves_only_to_legacy_topology(self):
        config = resolve_q_network_config({})
        self.assertTrue(config["legacy"])
        with self.assertRaisesRegex(ValueError, "Incomplete Q-network"):
            resolve_q_network_config({"q_network_normalization": LAYER_NORM})


if __name__ == "__main__":
    unittest.main()
