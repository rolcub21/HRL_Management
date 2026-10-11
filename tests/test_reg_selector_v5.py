import unittest

import numpy as np
import torch

from example.small_rooms_env import SmallRoomsEnv
from PSLAP.dynamic_yard import BlockView, YardSnapshot
from PSLAP.reg_selector_v5 import REGV5AssignmentSource, REGV5Config
from PSLAP.track_a import (
    TRACK_A_REG_SELECTOR_V5,
    run_track_a_episode,
    shared_candidate_mask,
)


class REGV5Tests(unittest.TestCase):
    def make_env(self):
        return SmallRoomsEnv(
            choose_storage=False,
            arrival_rate=0.5,
            proc_mean=50,
        )

    @staticmethod
    def inbound_context(env):
        inbound = next(
            block
            for block in env.blocks
            if block.position == env.pickup_cell and not block.stored
        )
        view = BlockView(
            inbound.label,
            inbound.position,
            float(inbound.get_remaining_storage_time()),
        )
        yard = YardSnapshot.from_env(env)
        candidates = shared_candidate_mask(yard, view, inbound.position)
        return yard, view, inbound.position, candidates

    def test_decision_intervals_do_not_overlap_rewards(self):
        torch.manual_seed(31)
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(301))
        config = REGV5Config(
            batch_size=8,
            min_replay_size=8,
            updates_per_episode=1,
            gamma=1.0,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_warmup_assignments=0,
            epsilon_decay_assignments=1,
        )
        source = REGV5AssignmentSource(env, config, seed=31)
        yard, block, pickup, candidates = self.inbound_context(env)

        source.propose(yard, block, pickup, candidates)
        source.on_step(1.0, {})
        source.on_step(2.0, {})
        source.propose(yard, block, pickup, candidates)
        source.on_step(4.0, {})
        source.on_episode_end(success=False, truncated=True)

        self.assertEqual(len(source.replay), 2)
        first, second = source.replay
        self.assertEqual(first.interval_return, 3.0)
        self.assertEqual(second.interval_return, 4.0)
        self.assertEqual((first.duration, second.duration), (2, 1))
        self.assertFalse(first.terminal)
        self.assertIsNotNone(first.next_observation)
        self.assertTrue(second.terminal)
        self.assertTrue(second.truncated)
        self.assertIsNone(second.next_observation)
        self.assertEqual(
            sum(item.interval_return for item in source.replay), 7.0
        )

    def test_variable_duration_double_dqn_target(self):
        torch.manual_seed(32)
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(302))
        config = REGV5Config(
            batch_size=3,
            min_replay_size=3,
            updates_per_episode=1,
            gamma=0.5,
            reward_scale=1.0,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_warmup_assignments=0,
            epsilon_decay_assignments=1,
        )
        source = REGV5AssignmentSource(env, config, seed=32)
        yard, block, pickup, candidates = self.inbound_context(env)
        source.propose(yard, block, pickup, candidates)
        source.on_step(1.0, {})
        source.on_step(2.0, {})
        source.propose(yard, block, pickup, candidates)
        source.on_step(3.0, {})
        source.on_episode_end(success=True, truncated=False)

        with torch.no_grad():
            for parameter in source.network.parameters():
                parameter.zero_()
            for parameter in source.target_network.parameters():
                parameter.zero_()
            source.target_network.scorer[-1].bias.fill_(2.0)
        targets = source._td_targets(list(source.replay)).cpu().numpy()

        # First interval: 1 + 0.5*2 = 2, then gamma^2 * next Q (2).
        np.testing.assert_allclose(targets, [2.5, 3.0], rtol=0, atol=1e-6)

    def test_full_episode_rewards_partition_and_checkpoint_round_trip(self):
        torch.manual_seed(33)
        env = self.make_env()
        instance = env.sample_episode_instance(303)
        config = REGV5Config(
            batch_size=8,
            min_replay_size=8,
            updates_per_episode=1,
            gamma=1.0,
            target_update_every=1,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_warmup_assignments=0,
            epsilon_decay_assignments=1,
        )
        source = REGV5AssignmentSource(env, config, seed=33)

        result = run_track_a_episode(
            env,
            TRACK_A_REG_SELECTOR_V5,
            max_steps=2000,
            episode_instance=instance,
            assignment_source=source,
        )

        self.assertEqual(result["success"], 1.0)
        self.assertEqual(result["invalid_assignment_count"], 0)
        self.assertEqual(len(source.replay), result["assignment_decision_count"])
        self.assertEqual(source.completed_intervals, len(source.replay))
        self.assertEqual(source.terminal_intervals, 1)
        self.assertEqual(source.truncated_intervals, 0)
        self.assertEqual(sum(item.duration for item in source.replay), result["steps"])
        self.assertAlmostEqual(
            sum(item.interval_return for item in source.replay),
            result["return"],
            places=8,
        )
        self.assertEqual(source.gradient_steps, 1)
        self.assertEqual(source.target_updates, 1)
        self.assertTrue(np.isfinite(source.loss_history[-1]))

        restored = REGV5AssignmentSource.from_checkpoint(
            env,
            source.checkpoint(),
            learning_enabled=False,
        )
        for key, value in source.ema_network.state_dict().items():
            torch.testing.assert_close(
                value.cpu(), restored.network.state_dict()[key]
            )


if __name__ == "__main__":
    unittest.main()
