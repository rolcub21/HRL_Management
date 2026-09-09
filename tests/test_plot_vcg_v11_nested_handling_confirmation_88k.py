from __future__ import annotations

import unittest

import plot_vcg_v11_nested_handling_confirmation_88k as plotter
import run_vcg_v11_nested_handling_confirmation_88k as subject


def _report():
    points = []
    for index, instance_seed in enumerate(subject.INSTANCE_SEEDS, start=1):
        points.append(
            {
                "instance_seed": instance_seed,
                "lambda0": {
                    "dense_return": float(index),
                    "mean_absolute_error": 10.0,
                    "physical_rehandles_per_100": 20.0,
                },
                "lambda0025": {
                    "dense_return": float(index + 1),
                    "mean_absolute_error": 11.0,
                    "physical_rehandles_per_100": 15.0,
                },
            }
        )
    payload = {
        "protocol": subject.PROTOCOL,
        "row_count": subject.EXPECTED_ROWS,
        "strict_safe_row_count": subject.EXPECTED_ROWS,
        "instance_points": points,
        "arm_summaries": {
            "lambda0": {
                "dense_return": 15.5,
                "mean_absolute_error": 10.0,
                "physical_rehandles_per_100": 20.0,
            },
            "lambda0025": {
                "dense_return": 16.5,
                "mean_absolute_error": 11.0,
                "physical_rehandles_per_100": 15.0,
            },
        },
    }
    return subject._with_hash(payload, "report_sha256")


class PlotDataTests(unittest.TestCase):
    def test_only_two_nested_arms_and_endpoints(self):
        curves = plotter.build_curve_data(_report())
        self.assertEqual(set(curves), {"lambda0", "lambda0025"})
        self.assertAlmostEqual(curves["lambda0"]["dense_return"]["mean"][-1], 15.5)
        self.assertAlmostEqual(
            curves["lambda0025"]["physical_rehandles_per_100"]["mean"][-1],
            15.0,
        )

    def test_wrong_panel_order_rejected(self):
        report = _report()
        report["instance_points"][0]["instance_seed"] = 999
        report = subject._with_hash(
            {key: value for key, value in report.items() if key != "report_sha256"},
            "report_sha256",
        )
        with self.assertRaises(plotter.ConfirmationPlotError):
            plotter.build_curve_data(report)


if __name__ == "__main__":
    unittest.main()
