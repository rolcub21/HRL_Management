from __future__ import annotations

import unittest

import plot_vcg_v11_nested_lambda_frontier_confirmation_89k as subject


class Confirmation89PlotTests(unittest.TestCase):
    def test_cumulative_mean(self):
        self.assertEqual(subject._cumulative([2, 4, 9]), [2.0, 3.0, 5.0])

    def test_endpoint_check_rejects_report_mismatch(self):
        curves = {}
        aggregate = {}
        for value in subject.confirmation.LAMBDA_GRID:
            curves[value] = {}
            aggregate[subject.confirmation._lambda_key(value)] = {}
            for metric, _ in subject.METRICS:
                curves[value][metric] = {"mean": [1.0]}
                report_metric = subject.REPORT_METRIC_NAMES[metric]
                aggregate[subject.confirmation._lambda_key(value)][report_metric] = 1.0
        report = {"aggregate_metrics": aggregate}
        subject._check_endpoints(curves, report)
        report["aggregate_metrics"]["0.1"]["mean_absolute_error"] = 2.0
        with self.assertRaises(subject.PlotError):
            subject._check_endpoints(curves, report)

    def test_dense_return_uses_aggregate_report_name(self):
        report = {
            "aggregate_metrics": {
                "0.0": {"mean_dense_return": 123.5},
            }
        }
        self.assertEqual(subject._report_metric(report, 0.0, "dense_return"), 123.5)


if __name__ == "__main__":
    unittest.main()
