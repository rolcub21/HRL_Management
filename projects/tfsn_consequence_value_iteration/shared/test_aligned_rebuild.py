"""Contract tests for the frozen-R74 aligned multiseed rebuild."""

from __future__ import annotations

import unittest

from .aligned_rebuild import (
    PROTOCOL_VALUE,
    dimensions,
    result_root,
    seed_values,
    task_seed,
)


class AlignedRebuildContractTest(unittest.TestCase):
    def test_full_allocation_is_exactly_36000(self):
        bootstrap, aligned, commands = dimensions("full")
        self.assertEqual((bootstrap, aligned, commands), (320, 80, 75))
        self.assertEqual((bootstrap + 2 * aligned) * commands, 36_000)
        self.assertEqual(PROTOCOL_VALUE["total_commands_per_seed"], 36_000)

    def test_preparation_seeds_are_independent(self):
        rows = [seed_values(seed) for seed in (0, 1, 2)]
        for name in ("collection_seed_base", "split_seed", "training_seed"):
            self.assertEqual(len({row[name] for row in rows}), 3)
        for phase in range(3):
            self.assertEqual(
                len({task_seed(seed, phase, 0) for seed in (0, 1, 2)}), 3
            )

    def test_domains_and_profiles_do_not_share_output(self):
        roots = {
            result_root(domain, seed, profile)
            for domain in ("pointmaze", "antmaze")
            for seed in (0, 1, 2)
            for profile in ("smoke", "full")
        }
        self.assertEqual(len(roots), 12)

    def test_no_task_oracle_inputs(self):
        self.assertEqual(PROTOCOL_VALUE["oracle_inputs"], [])
        self.assertFalse(PROTOCOL_VALUE["hidden_goals_accessed"])
        self.assertFalse(PROTOCOL_VALUE["task_reward_used"])
        self.assertFalse(PROTOCOL_VALUE["environment_reward_consumed"])
        self.assertFalse(
            PROTOCOL_VALUE["success_radius_read_by_planner_or_preparation"]
        )


if __name__ == "__main__":
    unittest.main()
