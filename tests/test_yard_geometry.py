import unittest

from example.yard_geometry import (
    make_shipyard_env,
    right_aligned_exit_cells,
)


class RightAlignedExitCellsTests(unittest.TestCase):
    def test_builds_contiguous_gate_ending_at_right_interior_column(self):
        self.assertEqual(
            right_aligned_exit_cells(10, 10, 1),
            ((9, 8),),
        )
        self.assertEqual(
            right_aligned_exit_cells(10, 10, 3),
            ((9, 6), (9, 7), (9, 8)),
        )
        self.assertEqual(
            right_aligned_exit_cells(10, 10, 8),
            tuple((9, col) for col in range(1, 9)),
        )

    def test_rejects_invalid_gate_widths_and_grid_dimensions(self):
        for exit_width in (-1, 0, 9):
            with self.subTest(exit_width=exit_width):
                with self.assertRaises(ValueError):
                    right_aligned_exit_cells(10, 10, exit_width)

        for grid_rows, grid_cols in ((3, 10), (10, 3), (3, 3)):
            with self.subTest(grid_rows=grid_rows, grid_cols=grid_cols):
                with self.assertRaises(ValueError):
                    right_aligned_exit_cells(grid_rows, grid_cols, 1)

    def test_environment_factory_uses_requested_right_aligned_gate(self):
        env = make_shipyard_env(
            arrival_rate=0.5,
            proc_mean=50,
            exit_width=3,
        )

        self.assertEqual(tuple(env.exit_cells), ((9, 6), (9, 7), (9, 8)))


class EpisodeScheduleIdentityTests(unittest.TestCase):
    def test_same_seed_has_same_schedule_but_distinct_geometry_instance(self):
        default_env = make_shipyard_env(
            arrival_rate=0.5,
            proc_mean=50,
        )
        narrow_env = make_shipyard_env(
            arrival_rate=0.5,
            proc_mean=50,
            exit_width=1,
        )

        default = default_env.sample_episode_instance(seed=46_000)
        narrow = narrow_env.sample_episode_instance(seed=46_000)

        self.assertEqual(default.arrival_steps, narrow.arrival_steps)
        self.assertEqual(
            default.storage_steps_needed,
            narrow.storage_steps_needed,
        )
        self.assertEqual(default.schedule_id, narrow.schedule_id)
        self.assertNotEqual(default.instance_id, narrow.instance_id)
        self.assertNotEqual(default.exit_cells, narrow.exit_cells)


if __name__ == "__main__":
    unittest.main()
