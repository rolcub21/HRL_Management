from __future__ import annotations

import unittest

import numpy as np

from .audit_executor import (
    chain_inversion_summary,
    derive_bank,
    fixed_contract_env_kwargs,
    trajectory_ranges,
)
from .audit_executor_adaptive_reach import (
    derive_transition_bank,
    nearest_transition,
    standardized_sources,
    standardize_query,
)


class ExecutorAuditBankTest(unittest.TestCase):
    def test_complete_trajectories_are_partitioned(self):
        terminals = np.asarray([False, False, True, False, False, True])
        self.assertEqual(trajectory_ranges(terminals), [(0, 2), (3, 5)])

    def test_targets_follow_the_command_contract(self):
        observations = np.arange(24, dtype=np.float32).reshape(6, 4)
        terminals = np.asarray([False, False, True, False, False, True])
        bank = derive_bank(observations, terminals, command_steps=1, episode_commands=2)
        np.testing.assert_array_equal(bank["start_index"], [0, 3])
        np.testing.assert_array_equal(bank["target_index"], [[1, 2], [4, 5]])
        np.testing.assert_array_equal(bank["source_xy"], observations[[0, 3], :2])
        np.testing.assert_array_equal(
            bank["target_xy"], observations[[[1, 2], [4, 5]], :2]
        )

    def test_incomplete_trajectory_is_rejected(self):
        with self.assertRaises(RuntimeError):
            trajectory_ranges(np.asarray([False, False]))

    def test_wrong_trajectory_horizon_is_rejected(self):
        observations = np.zeros((4, 3), dtype=np.float32)
        terminals = np.asarray([False, False, False, True])
        with self.assertRaises(RuntimeError):
            derive_bank(observations, terminals, command_steps=1, episode_commands=2)

    def test_fixed_contract_disables_only_goal_termination(self):
        self.assertEqual(
            fixed_contract_env_kwargs(command_steps=100, episode_commands=10),
            {
                "terminate_at_goal": False,
                "success_timing": "post",
                "max_episode_steps": 1000,
            },
        )

    def test_fixed_contract_rejects_nonpositive_horizon(self):
        with self.assertRaises(ValueError):
            fixed_contract_env_kwargs(command_steps=0, episode_commands=10)

    def test_chain_inversion_counts_first_event_and_recovery(self):
        rows = [
            {
                "chain": 0,
                "command_index": 0,
                "initial_distance_m": 2.0,
                "inverted_during_command": False,
                "inverted_at_endpoint": False,
            },
            {
                "chain": 0,
                "command_index": 1,
                "initial_distance_m": 3.0,
                "inverted_during_command": True,
                "inverted_at_endpoint": True,
            },
            {
                "chain": 1,
                "command_index": 0,
                "initial_distance_m": 4.0,
                "inverted_during_command": False,
                "inverted_at_endpoint": False,
            },
        ]
        summary = chain_inversion_summary(rows)
        self.assertEqual(summary["chains_with_inversion"], 1)
        self.assertEqual(summary["chain_inversion_rate"], 0.5)
        self.assertEqual(summary["inverted_chains_recovered_by_chain_end"], 0)
        self.assertEqual(summary["first_inversion_events"][0]["command_index"], 1)


class AdaptiveReachAuditBankTest(unittest.TestCase):
    def test_bank_contains_every_exact_command_step_transition(self):
        observations = np.arange(24, dtype=np.float32).reshape(6, 4)
        terminals = np.asarray([False, False, True, False, False, True])
        bank = derive_transition_bank(observations, terminals, command_steps=1)
        np.testing.assert_array_equal(bank["source_index"], [0, 1, 3, 4])
        np.testing.assert_array_equal(bank["target_index"], [1, 2, 4, 5])
        np.testing.assert_array_equal(bank["reset_index"], [0, 3])
        np.testing.assert_array_equal(
            bank["target_xy"], observations[[1, 2, 4, 5], :2]
        )

    def test_standardized_nearest_transition_uses_full_active_state(self):
        observations = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [2.0, 0.0, 3.0],
                [0.0, 0.0, 9.0],
                [1.0, 0.0, 8.0],
                [2.0, 0.0, 7.0],
            ],
            dtype=np.float32,
        )
        terminals = np.asarray([False, False, True, False, False, True])
        bank = derive_transition_bank(observations, terminals, command_steps=1)
        source = standardized_sources(bank)
        query = standardize_query(np.asarray([0.0, 0.0, 8.9]), bank)
        selected, distance = nearest_transition(source, query)
        self.assertEqual(int(bank["source_index"][selected]), 3)
        self.assertGreaterEqual(distance, 0.0)

    def test_constant_coordinates_are_removed_without_a_scale_floor(self):
        observations = np.asarray(
            [[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]], dtype=np.float32
        )
        bank = derive_transition_bank(
            observations, np.asarray([False, False, True]), command_steps=1
        )
        np.testing.assert_array_equal(bank["active_coordinate"], [True, False])
        self.assertEqual(standardized_sources(bank).shape[1], 1)


if __name__ == "__main__":
    unittest.main()
