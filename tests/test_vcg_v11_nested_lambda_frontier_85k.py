from __future__ import annotations

import unittest

import run_vcg_v11_nested_lambda_frontier_85k as subject


class FrontierTests(unittest.TestCase):
    def test_grid_and_new_row_count(self):
        self.assertEqual(subject.LAMBDA_GRID, (0.0, 0.025, 0.05, 0.1, 0.2))
        self.assertEqual(subject.EXPECTED_ROWS, 180)
        self.assertEqual(subject.EXPECTED_NEW_ROWS, 72)

    def test_nondominated(self):
        points = {
            0.0: {"mean_absolute_error": 10.0, "physical_rehandles_per_100": 20.0},
            0.1: {"mean_absolute_error": 11.0, "physical_rehandles_per_100": 10.0},
            0.2: {"mean_absolute_error": 12.0, "physical_rehandles_per_100": 11.0},
        }
        self.assertEqual(subject._nondominated(points), [0.0, 0.1])

    def test_monotonicity_reports_inversion(self):
        points = {
            value: {"physical_rehandles_per_100": rehandles}
            for value, rehandles in zip(subject.LAMBDA_GRID, (20, 15, 10, 11, 5))
        }
        result = subject._monotonicity(points)
        self.assertEqual(result["inversion_count"], 1)
        self.assertEqual(result["total_inversion_magnitude"], 1)


if __name__ == "__main__":
    unittest.main()
