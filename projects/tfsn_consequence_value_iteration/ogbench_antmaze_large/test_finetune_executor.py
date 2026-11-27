from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .finetune_executor import (
    EmpiricalCommandSampler,
    OnlineCommandReplay,
    empirical_command_pairs,
    load_command_diagnostics,
    mix_update_batches,
    save_command_diagnostics,
)


class FineTuneExecutorTest(unittest.TestCase):
    def test_empirical_pairs_never_cross_trajectory_end(self):
        terminals = np.asarray([False, False, True, False, False, True])
        source, target = empirical_command_pairs(terminals, command_steps=1)
        np.testing.assert_array_equal(source, [0, 1, 3, 4])
        np.testing.assert_array_equal(target, [1, 2, 4, 5])

    def test_target_offsets_are_snapped_to_recorded_support(self):
        observations = np.zeros((6, 3), np.float32)
        observations[:, 0] = [0, 1, 2, 10, 11, 12]
        terminals = np.asarray([False, False, True, False, False, True])
        sampler = EmpiricalCommandSampler(observations, terminals, command_steps=1)
        target, details = sampler.sample_target(
            np.asarray([10.0, 0.0]), np.random.default_rng(0)
        )
        self.assertIn(float(target[0]), {10.0, 11.0, 12.0})
        self.assertGreaterEqual(details["empirical_offset_distance_m"], 1.0)

    def test_online_replay_relabels_only_within_command(self):
        replay = OnlineCommandReplay(capacity=20, observation_dim=2, action_dim=1)
        replay.add_command(
            np.asarray([[0, 0], [1, 0], [2, 0]], np.float32),
            np.asarray([[0.1], [0.2]], np.float32),
        )
        replay.add_command(
            np.asarray([[10, 0], [11, 0], [12, 0]], np.float32),
            np.asarray([[0.3], [0.4]], np.float32),
        )
        config = {
            "value_p_curgoal": 0.0,
            "value_p_trajgoal": 1.0,
            "value_p_randomgoal": 0.0,
            "value_geom_sample": True,
            "actor_p_curgoal": 0.0,
            "actor_p_trajgoal": 1.0,
            "actor_p_randomgoal": 0.0,
            "actor_geom_sample": False,
            "discount": 0.99,
        }
        batch = replay.sample(100, config, np.random.default_rng(1))
        source_side = batch["observations"][:, 0] < 5
        self.assertTrue(np.all((batch["actor_goals"][:, 0] < 5) == source_side))
        self.assertTrue(np.all((batch["value_goals"][:, 0] < 5) == source_side))

    def test_physical_episode_replay_extends_goals_across_commands_only(self):
        replay = OnlineCommandReplay(
            capacity=30, observation_dim=2, action_dim=1,
            goal_boundary="physical_episode",
        )
        replay.start_episode()
        replay.add_command(
            np.asarray([[0, 0], [1, 0], [2, 0]], np.float32),
            np.asarray([[0.1], [0.2]], np.float32),
        )
        replay.add_command(
            np.asarray([[2, 0], [3, 0], [4, 0]], np.float32),
            np.asarray([[0.3], [0.4]], np.float32),
        )
        first_episode_end = replay.size - 1
        self.assertTrue(np.all(
            replay.ends[replay.valid_rows[: replay.valid_size]]
            == first_episode_end
        ))
        replay.start_episode()
        replay.add_command(
            np.asarray([[10, 0], [11, 0], [12, 0]], np.float32),
            np.asarray([[0.5], [0.6]], np.float32),
        )
        np.testing.assert_array_equal(
            replay.ends[replay.valid_rows[:4]],
            np.full(4, first_episode_end),
        )
        self.assertTrue(np.all(replay.ends[replay.valid_rows[4:6]] > 4))

    def test_mixed_batch_has_declared_offline_then_online_rows(self):
        keys = (
            "observations", "actions", "next_observations", "value_goals",
            "actor_goals", "masks", "rewards",
        )
        offline = {key: np.zeros((2, 1), np.float32) for key in keys}
        online = {key: np.ones((3, 1), np.float32) for key in keys}
        mixed = mix_update_batches(offline, online)
        for key in keys:
            self.assertEqual(mixed[key].shape, (5, 1))
            np.testing.assert_array_equal(mixed[key][:2], 0.0)
            np.testing.assert_array_equal(mixed[key][2:], 1.0)

    def test_command_diagnostics_round_trip_without_json_rows(self):
        rows = [
            {"command": 0, "distance": 1.5, "inverted": False},
            {"command": 1, "distance": 2.5, "inverted": True},
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "commands.npz"
            save_command_diagnostics(path, rows)
            self.assertEqual(load_command_diagnostics(path), rows)


if __name__ == "__main__":
    unittest.main()
