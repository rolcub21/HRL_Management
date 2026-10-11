from __future__ import annotations

import unittest

import plot_vcg_v11_nested_all_baselines_88k as subject


class ReductionTests(unittest.TestCase):
    def test_kim_dense_return_alias(self):
        self.assertEqual(
            subject._metric({"dense_objective_return": 12.5}, "dense_return"),
            12.5,
        )

    def _rows(self, failed_seed=None):
        rows = []
        for seed in subject.INSTANCE_SEEDS:
            for model in (0, 1, 2):
                rows.append(
                    {
                        "instance_seed": seed,
                        "model_seed": model,
                        "strict_safe_complete": seed != failed_seed,
                        "dense_return": float(seed - 88000 + model),
                        "mean_absolute_error": 10.0,
                        "physical_rehandles": 1,
                        "required_deliveries": 8,
                    }
                )
        return rows

    def test_complete_curve_has_30_points(self):
        result = subject._reduce(
            self._rows(),
            replicate_key=lambda row: row["model_seed"],
            expected_replicates=(0, 1, 2),
            nuisance_count=1,
        )
        self.assertTrue(result["whole_method_eligible"])
        self.assertEqual(result["numeric_prefix_length"], 30)
        self.assertAlmostEqual(
            result["curves"]["physical_rehandles_per_100_required_deliveries"]["mean"][-1],
            12.5,
        )

    def test_failure_suppresses_endpoint_and_stops_prefix(self):
        result = subject._reduce(
            self._rows(failed_seed=88007),
            replicate_key=lambda row: row["model_seed"],
            expected_replicates=(0, 1, 2),
            nuisance_count=1,
        )
        self.assertFalse(result["whole_method_eligible"])
        self.assertEqual(result["first_failure_k"], 8)
        self.assertEqual(result["numeric_prefix_length"], 7)


if __name__ == "__main__":
    unittest.main()
