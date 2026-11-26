import unittest

from projects.tfsn_consequence_value_iteration.antmaze.expanded_development import (
    EXTENSION_INDICES,
    EXTENSION_TASK_IDS,
    RESERVED_CONFIRMATION_INDICES,
    build_extension_panel,
)


class TestExpandedDevelopment(unittest.TestCase):
    def test_extension_uses_only_unused_goal_indices(self):
        original = {
            "name": "development",
            "goal_partition_seed": 37700,
            "goal_offset": 2,
            "tasks": [
                {"task_id": index, "goal": [index, index], "reset_seed": index}
                for index in range(9)
            ],
        }
        goals = [[index, -index] for index in range(6)]
        panel = build_extension_panel(original, goals)
        self.assertEqual(
            panel["extension_goal_indices"], list(EXTENSION_INDICES)
        )
        self.assertEqual(
            [row["task_id"] for row in panel["tasks"]],
            list(EXTENSION_TASK_IDS),
        )
        self.assertTrue(
            set(panel["extension_goal_indices"]).isdisjoint(
                RESERVED_CONFIRMATION_INDICES
            )
        )
        self.assertFalse(panel["confirmation_panel_accessed"])


if __name__ == "__main__":
    unittest.main()
