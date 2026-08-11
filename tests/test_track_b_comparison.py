import unittest

from compare_track_b import (
    LEARNED_METHOD,
    OFFLINE_REFERENCE,
    ONLINE_PRIMARY,
    URGENCY_METHOD,
    method_summary,
    paired_summary,
)
from PSLAP.track_a import TRACK_A_GA_OFFLINE, TRACK_A_NEAREST_FREE


def row(
    method,
    instance_id,
    episode_return,
    steps,
    success,
    absolute,
    tardiness,
    *,
    flow_mean=100.0,
    flow_completion=1.0,
):
    flow_observed = flow_completion == 1.0
    return {
        "method": method,
        "instance_id": instance_id,
        "controller_architecture": (
            "test_controller" if method == LEARNED_METHOD else "not_applicable"
        ),
        "comparison_group": (
            OFFLINE_REFERENCE
            if method == TRACK_A_GA_OFFLINE
            else ONLINE_PRIMARY
        ),
        "information_regime": (
            "offline_full_schedule"
            if method == TRACK_A_GA_OFFLINE
            else "online"
        ),
        "return": episode_return,
        "steps": steps,
        "success": success,
        "strict_method_success": success,
        "mean_absolute_error": absolute,
        "mean_tardiness": tardiness,
        "within_target_window_rate": None if absolute is None else 0.5,
        "storage_flow_completion_rate": flow_completion,
        "storage_flow_fully_observed": flow_observed,
        "storage_flow_manifest_count": 1,
        "storage_flow_arrived_count": 1,
        "storage_flow_completed_count": int(flow_observed),
        "storage_flow_unfinished_count": int(not flow_observed),
        "storage_flow_right_censored_count": int(not flow_observed),
        "storage_flow_not_yet_arrived_count": 0,
        "completed_storage_flow_times": (
            [flow_mean] if flow_observed else []
        ),
        "mean_storage_flow_time": flow_mean if flow_observed else None,
        "median_storage_flow_time": flow_mean if flow_observed else None,
        "p90_storage_flow_time": (
            flow_mean + 10.0 if flow_observed else None
        ),
        "p95_storage_flow_time": (
            flow_mean + 15.0 if flow_observed else None
        ),
        "max_storage_flow_time": (
            flow_mean + 20.0 if flow_observed else None
        ),
        "invalid_assignment_count": 0,
        "fallback_count": 0,
        "total_method_seconds": 1.0,
    }


class TrackBComparisonTests(unittest.TestCase):
    def test_paired_metric_orientation_always_favors_learned(self):
        rows = [
            row(
                LEARNED_METHOD, "a", 10.0, 8, 1.0, 2.0, 1.0,
                flow_mean=80.0,
            ),
            row(
                LEARNED_METHOD, "b", 12.0, 9, 1.0, 3.0, 2.0,
                flow_mean=90.0,
            ),
            row(
                TRACK_A_NEAREST_FREE, "a", 7.0, 12, 0.0, 5.0, 4.0,
                flow_mean=100.0,
            ),
            row(
                TRACK_A_NEAREST_FREE, "b", 8.0, 11, 1.0, 4.0, 3.0,
                flow_mean=110.0,
            ),
        ]
        result = paired_summary(
            rows, (LEARNED_METHOD, TRACK_A_NEAREST_FREE), samples=50
        )[0]
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["instance_ids"], ["a", "b"])
        self.assertEqual(result["metrics"]["return_advantage"]["mean"], 3.5)
        self.assertEqual(result["metrics"]["step_reduction"]["mean"], 3.0)
        self.assertEqual(
            result["metrics"]["absolute_error_reduction"]["mean"], 2.0
        )
        self.assertEqual(result["metrics"]["tardiness_reduction"]["mean"], 2.0)
        self.assertEqual(result["metrics"]["success_advantage"]["mean"], 0.5)
        self.assertEqual(
            result["metrics"]["mean_storage_flow_time_reduction"]["mean"],
            20.0,
        )
        self.assertEqual(
            result["metrics"]["storage_flow_completion_advantage"]["mean"],
            0.0,
        )

    def test_missing_delivery_metrics_are_excluded_not_crashed(self):
        rows = [
            row(LEARNED_METHOD, "a", 1.0, 10, 0.0, None, None),
            row(TRACK_A_NEAREST_FREE, "a", 2.0, 10, 0.0, None, None),
        ]
        result = paired_summary(
            rows, (LEARNED_METHOD, TRACK_A_NEAREST_FREE), samples=10
        )[0]
        metric = result["metrics"]["absolute_error_reduction"]
        self.assertEqual(metric["n"], 0)
        self.assertIsNone(metric["mean"])
        self.assertIsNone(metric["bootstrap_95_ci"])

        summary = method_summary(LEARNED_METHOD, rows)
        self.assertIsNone(summary["mean_absolute_error"])
        self.assertIsNone(summary["within_target_window_rate"])

    def test_incomplete_storage_flow_is_counted_but_excluded_from_times(self):
        rows = [
            row(
                LEARNED_METHOD,
                "a",
                1.0,
                10,
                0.0,
                None,
                None,
                flow_mean=20.0,
                flow_completion=0.5,
            ),
            row(
                TRACK_A_NEAREST_FREE,
                "a",
                2.0,
                10,
                1.0,
                2.0,
                1.0,
                flow_mean=100.0,
            ),
        ]

        metrics = paired_summary(
            rows, (LEARNED_METHOD, TRACK_A_NEAREST_FREE), samples=10
        )[0]["metrics"]

        self.assertEqual(
            metrics["storage_flow_completion_advantage"]["mean"], -0.5
        )
        self.assertEqual(metrics["mean_storage_flow_time_reduction"]["n"], 0)
        self.assertIsNone(
            metrics["mean_storage_flow_time_reduction"]["mean"]
        )

    def test_offline_ga_is_not_grouped_with_online_primary_methods(self):
        rows = [
            row(LEARNED_METHOD, "a", 1.0, 10, 1.0, 1.0, 1.0),
            row(TRACK_A_GA_OFFLINE, "a", 2.0, 9, 1.0, 1.0, 1.0),
        ]
        result = paired_summary(
            rows, (LEARNED_METHOD, TRACK_A_GA_OFFLINE), samples=10
        )[0]
        self.assertEqual(result["comparison_group"], OFFLINE_REFERENCE)

    def test_repaired_candidate_can_anchor_paired_baseline_differences(self):
        rows = [
            row(URGENCY_METHOD, "a", 10.0, 8, 1.0, 2.0, 1.0),
            row(TRACK_A_NEAREST_FREE, "a", 7.0, 12, 0.0, 5.0, 4.0),
        ]

        result = paired_summary(
            rows,
            (URGENCY_METHOD, TRACK_A_NEAREST_FREE),
            samples=10,
            reference_method=URGENCY_METHOD,
        )[0]

        self.assertEqual(result["reference_method"], URGENCY_METHOD)
        self.assertNotIn("learned_method", result)
        self.assertEqual(result["metrics"]["return_advantage"]["mean"], 3.0)
        self.assertEqual(result["metrics"]["step_reduction"]["mean"], 4.0)


if __name__ == "__main__":
    unittest.main()
