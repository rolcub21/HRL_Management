from __future__ import annotations

import math
import unittest

import run_vcg_v11_nested_handling_confirmation_88k as subject


class ConfirmationMathTests(unittest.TestCase):
    def test_frozen_grid(self):
        self.assertEqual(subject.INSTANCE_SEEDS, tuple(range(88000, 88030)))
        self.assertEqual(subject.MODEL_SEEDS, (0, 1, 2))
        self.assertEqual(subject.ARMS, ("lambda0", "lambda0025"))
        self.assertEqual(subject.ARM_LAMBDAS, {"lambda0": 0.0, "lambda0025": 0.025})
        self.assertEqual(subject.EXPECTED_ROWS, 180)

    def test_paired_interval_constants_and_arithmetic(self):
        values = [float(index) for index in range(30)]
        observed = subject._paired_interval(values)
        mean = 14.5
        se = math.sqrt(sum((value - mean) ** 2 for value in values) / 29) / math.sqrt(30)
        self.assertAlmostEqual(observed["mean"], mean)
        self.assertAlmostEqual(
            observed["simultaneous_one_sided_upper_bound"],
            mean + subject.SIMULTANEOUS_ONE_SIDED_T * se,
        )
        self.assertAlmostEqual(
            observed["simultaneous_one_sided_lower_bound"],
            mean - subject.SIMULTANEOUS_ONE_SIDED_T * se,
        )

    def test_summary_rehandle_denominator(self):
        rows = [
            {
                "dense_return": 1.0,
                "mean_absolute_error": 2.0,
                "steps": 3,
                "physical_rehandles": index % 2,
            }
            for index in range(6)
        ]
        summary = subject._summary(rows)
        self.assertEqual(summary["total_physical_rehandles"], 3)
        self.assertAlmostEqual(summary["physical_rehandles_per_100"], 6.25)

    def test_strict_gate_constants(self):
        self.assertAlmostEqual(subject.SIMULTANEOUS_ONE_SIDED_T, 2.2343299587959504)
        self.assertAlmostEqual(subject.NOMINAL_TWO_SIDED_T, 2.0452296421)


if __name__ == "__main__":
    unittest.main()
