from argparse import Namespace
import unittest

from enumerate_viability_kernel import PROTOCOL, enumerate_kernel


def kernel_args(**overrides):
    values = {
        "lam": 0.8,
        "mu": 100.0,
        "grid_rows": 4,
        "grid_cols": 4,
        "exit_width": 1,
        "number_blocks": 3,
        "max_occupancy": 3,
        "min_occupancy": 0,
        "max_depth": None,
        "max_nodes": 20_000,
        "max_primitive_steps": None,
        "search_order": "goal_directed",
        "include_states": True,
    }
    values.update(overrides)
    return Namespace(**values)


class CompactViabilityKernelTests(unittest.TestCase):
    def test_compact_kernel_is_deterministic_dynamics_only_supervision(self):
        first = enumerate_kernel(kernel_args())
        second = enumerate_kernel(kernel_args())

        self.assertEqual(first, second)
        self.assertEqual(first["protocol"], PROTOCOL)
        self.assertFalse(first["baseline_viability_teacher"])
        self.assertEqual(
            first["information_regime"], "online_no_future_schedule"
        )
        self.assertEqual(first["state_count"], 20)
        self.assertEqual(first["status_counts"], {"SAFE": 19, "UNSAFE": 1})
        self.assertEqual(len(first["states"]), first["state_count"])

    def test_bfs_and_goal_directed_agree_on_status_but_only_bfs_labels_rank(self):
        goal_directed = enumerate_kernel(
            kernel_args(search_order="goal_directed")
        )
        breadth_first = enumerate_kernel(
            kernel_args(search_order="breadth_first")
        )

        goal_keys = [
            (
                row["occupancy_mask"],
                tuple(row["agent_position"]),
                row["status"],
            )
            for row in goal_directed["states"]
        ]
        bfs_keys = [
            (
                row["occupancy_mask"],
                tuple(row["agent_position"]),
                row["status"],
            )
            for row in breadth_first["states"]
        ]
        self.assertEqual(goal_keys, bfs_keys)

        goal_safe = [
            row for row in goal_directed["states"] if row["status"] == "SAFE"
        ]
        bfs_safe = [
            row for row in breadth_first["states"] if row["status"] == "SAFE"
        ]
        self.assertTrue(
            all(
                (
                    row["exact_recovery_rank"] == 0
                    and row["recovery_rank_exact"]
                    and row["witness_primitive_steps"] == 0
                )
                if row["occupancy_count"] == 0
                else (
                    row["exact_recovery_rank"] is None
                    and not row["recovery_rank_exact"]
                    and row["witness_primitive_steps"] is not None
                )
                for row in goal_safe
            )
        )
        self.assertTrue(
            all(
                row["exact_recovery_rank"] == row["witness_macro_count"]
                and row["recovery_rank_exact"]
                for row in bfs_safe
            )
        )

    def test_budget_unknowns_remain_unknown_and_agent_position_changes_label(self):
        budgeted = enumerate_kernel(kernel_args(max_nodes=1))

        self.assertEqual(budgeted["state_count"], 20)
        self.assertGreater(budgeted["status_counts"].get("UNKNOWN", 0), 0)
        unknown = [
            row for row in budgeted["states"] if row["status"] == "UNKNOWN"
        ]
        self.assertTrue(
            all(
                row["exact_recovery_rank"] is None
                and row["witness_primitive_steps"] is None
                for row in unknown
            )
        )

        exhaustive = enumerate_kernel(kernel_args())
        fully_occupied = [
            row
            for row in exhaustive["states"]
            if row["occupancy_count"]
            == exhaustive["geometry"]["storage_cell_count"]
        ]
        self.assertEqual(
            {row["status"] for row in fully_occupied}, {"SAFE", "UNSAFE"}
        )
        self.assertEqual(
            {row["occupancy_mask"] for row in fully_occupied}, {7}
        )


if __name__ == "__main__":
    unittest.main()
