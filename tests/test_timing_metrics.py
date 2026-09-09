import math
from types import SimpleNamespace
import unittest

import numpy as np

from example.helper.timing_metrics import (
    STORAGE_FLOW_METRIC_CONTRACT,
    summarize_block_storage_flow,
    summarize_delivery_timing,
    summarize_storage_flow,
    summarize_storage_flow_runs,
)
from example.small_rooms_env import SmallRoomsEnv


class DeliveryTimingMetricTests(unittest.TestCase):
    def test_metrics_separate_bias_accuracy_earliness_and_tardiness(self):
        deviations = [-10.0, 0.0, 5.0, 30.0]
        result = summarize_delivery_timing(deviations, target_window=20.0)

        self.assertEqual(result["delivery_count"], 4)
        self.assertAlmostEqual(result["mean_signed_deviation"], 6.25)
        self.assertAlmostEqual(result["mean_absolute_error"], 11.25)
        self.assertAlmostEqual(result["mean_tardiness"], 8.75)
        self.assertAlmostEqual(result["mean_earliness"], 2.5)
        self.assertAlmostEqual(result["within_target_window_rate"], 0.75)
        self.assertAlmostEqual(result["tardy_delivery_rate"], 0.5)
        self.assertAlmostEqual(result["mean_tardiness_when_tardy"], 17.5)
        self.assertAlmostEqual(result["p90_tardiness"], 22.5)
        self.assertAlmostEqual(result["p90_absolute_error"], 24.0)
        self.assertAlmostEqual(
            result["signed_deviation_std"], np.std(deviations)
        )
        self.assertAlmostEqual(
            result["absolute_error_std"], np.std(np.abs(deviations))
        )

    def test_empty_delivery_set_is_explicit(self):
        result = summarize_delivery_timing([], target_window=20.0)
        self.assertEqual(result["delivery_count"], 0)
        for key, value in result.items():
            if key != "delivery_count":
                self.assertTrue(math.isnan(value), key)

    def test_negative_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            summarize_delivery_timing([0.0], target_window=-1.0)

    def test_active_delivery_reward_is_symmetric_in_timing_deviation(self):
        env = SmallRoomsEnv(choose_storage=False)
        storage = env.storage_positions[0]
        for magnitude in (0.0, 7.0, env.DELIVERY_TARGET_WINDOW, 35.0):
            self.assertAlmostEqual(
                env._delivery_reward(storage, -magnitude),
                env._delivery_reward(storage, magnitude),
            )


class StorageFlowMetricTests(unittest.TestCase):
    PRIMARY_FIELDS = (
        "mean_storage_flow_time",
        "median_storage_flow_time",
        "p90_storage_flow_time",
        "p95_storage_flow_time",
        "max_storage_flow_time",
    )

    def test_complete_manifest_has_exact_linear_quantiles(self):
        result = summarize_storage_flow(
            [0, 2, 5, 5],
            [4, 6, 10, 12],
            12,
            labels=["B1", "B2", "B3", "B4"],
        )

        self.assertEqual(
            result["storage_flow_metric_contract"],
            STORAGE_FLOW_METRIC_CONTRACT,
        )
        self.assertEqual(result["storage_flow_manifest_count"], 4)
        self.assertEqual(result["storage_flow_arrived_count"], 4)
        self.assertEqual(result["storage_flow_completed_count"], 4)
        self.assertEqual(result["storage_flow_right_censored_count"], 0)
        self.assertEqual(result["storage_flow_not_yet_arrived_count"], 0)
        self.assertEqual(result["storage_flow_unfinished_count"], 0)
        self.assertEqual(result["storage_flow_completion_rate"], 1.0)
        self.assertEqual(
            result["storage_flow_arrived_completion_rate"], 1.0
        )
        self.assertTrue(result["storage_flow_fully_observed"])
        self.assertEqual(result["completed_storage_flow_times"], [4, 4, 5, 7])
        self.assertEqual(result["right_censored_storage_flow_ages"], [])
        self.assertAlmostEqual(result["mean_storage_flow_time"], 5.0)
        self.assertAlmostEqual(result["median_storage_flow_time"], 4.5)
        self.assertAlmostEqual(result["p90_storage_flow_time"], 6.4)
        self.assertAlmostEqual(result["p95_storage_flow_time"], 6.7)
        self.assertAlmostEqual(result["max_storage_flow_time"], 7.0)
        self.assertEqual(
            [record["block_label"] for record in result["storage_flow_records"]],
            ["B1", "B2", "B3", "B4"],
        )
        self.assertTrue(
            all(
                record["status"] == "completed"
                for record in result["storage_flow_records"]
            )
        )

    def test_truncation_is_explicit_and_not_imputed_into_primary_stats(self):
        result = summarize_storage_flow(
            [0, 3, 8, 15],
            [5, None, None, None],
            10,
            labels=["B1", "B2", "B3", "B4"],
        )

        self.assertEqual(result["storage_flow_manifest_count"], 4)
        self.assertEqual(result["storage_flow_arrived_count"], 3)
        self.assertEqual(result["storage_flow_completed_count"], 1)
        self.assertEqual(result["storage_flow_right_censored_count"], 2)
        self.assertEqual(result["storage_flow_not_yet_arrived_count"], 1)
        self.assertEqual(result["storage_flow_unfinished_count"], 3)
        self.assertEqual(result["storage_flow_completion_rate"], 0.25)
        self.assertAlmostEqual(
            result["storage_flow_arrived_completion_rate"], 1.0 / 3.0
        )
        self.assertFalse(result["storage_flow_fully_observed"])
        self.assertEqual(result["completed_storage_flow_times"], [5])
        self.assertEqual(result["right_censored_storage_flow_ages"], [7, 2])
        for field in self.PRIMARY_FIELDS:
            self.assertTrue(math.isnan(result[field]), field)

        records = result["storage_flow_records"]
        self.assertEqual(
            [record["manifest_index"] for record in records],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            [record["status"] for record in records],
            [
                "completed",
                "right_censored",
                "right_censored",
                "not_yet_arrived",
            ],
        )
        self.assertEqual(records[0]["storage_flow_time"], 5)
        self.assertEqual(records[1]["censor_step"], 10)
        self.assertEqual(records[1]["censor_age"], 7)
        self.assertEqual(records[2]["censor_age"], 2)
        self.assertIsNone(records[3]["censor_step"])
        self.assertIsNone(records[3]["censor_age"])

    def test_arrival_at_observation_end_is_right_censored_at_zero(self):
        result = summarize_storage_flow([0, 10], [4, None], 10)

        self.assertEqual(result["storage_flow_arrived_count"], 2)
        self.assertEqual(result["storage_flow_right_censored_count"], 1)
        self.assertEqual(result["storage_flow_not_yet_arrived_count"], 0)
        self.assertEqual(result["right_censored_storage_flow_ages"], [0])
        self.assertEqual(result["storage_flow_records"][1]["censor_age"], 0)

    def test_no_completed_storage_events_has_counts_and_nan_stats(self):
        result = summarize_storage_flow([0, 5], [None, None], 3)

        self.assertEqual(result["storage_flow_arrived_count"], 1)
        self.assertEqual(result["storage_flow_completed_count"], 0)
        self.assertEqual(result["storage_flow_right_censored_count"], 1)
        self.assertEqual(result["storage_flow_not_yet_arrived_count"], 1)
        self.assertEqual(result["storage_flow_completion_rate"], 0.0)
        self.assertEqual(result["storage_flow_arrived_completion_rate"], 0.0)
        for field in self.PRIMARY_FIELDS:
            self.assertTrue(math.isnan(result[field]), field)

    def test_block_adapter_uses_first_storage_timestamp_and_manifest_order(self):
        blocks = [
            SimpleNamespace(label="B1", arrival_step=0, stored_time_step=4),
            SimpleNamespace(label="B2", arrival_step=3, stored_time_step=9),
        ]

        result = summarize_block_storage_flow(blocks, 12)

        self.assertEqual(result["completed_storage_flow_times"], [4, 6])
        self.assertEqual(
            [record["block_label"] for record in result["storage_flow_records"]],
            ["B1", "B2"],
        )

    def test_invalid_flow_inputs_are_rejected(self):
        cases = (
            (([0], [1, 2], 2), "equal length"),
            (([0], [1], -1), "observation_end_step"),
            (([-1], [None], 2), "arrival_steps"),
            (([2, 1], [None, None], 2), "nondecreasing"),
            (([3], [2], 4), "precede arrival"),
            (([0], [5], 4), "exceed observation_end_step"),
            (([0], [True], 4), "stored_steps"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, message):
                    summarize_storage_flow(*arguments)

        with self.assertRaisesRegex(ValueError, "labels"):
            summarize_storage_flow([0], [1], 1, labels=[])

    def test_run_summary_pools_blocks_only_when_every_manifest_is_complete(self):
        first = summarize_storage_flow([0, 2], [4, 8], 8)
        second = summarize_storage_flow([0, 1], [5, 8], 8)

        complete = summarize_storage_flow_runs([first, second])

        self.assertEqual(complete["storage_flow_episode_count"], 2)
        self.assertEqual(complete["storage_flow_manifest_count"], 4)
        self.assertEqual(complete["storage_flow_completed_count"], 4)
        self.assertTrue(complete["storage_flow_fully_observed"])
        self.assertEqual(complete["storage_flow_completion_rate"], 1.0)
        self.assertAlmostEqual(complete["mean_storage_flow_time"], 5.5)
        self.assertAlmostEqual(complete["median_storage_flow_time"], 5.5)
        self.assertAlmostEqual(complete["p90_storage_flow_time"], 6.7)
        self.assertEqual(complete["max_storage_flow_time"], 7.0)

        partial = summarize_storage_flow([0, 3], [4, None], 6)
        censored = summarize_storage_flow_runs([first, partial])
        self.assertFalse(censored["storage_flow_fully_observed"])
        self.assertEqual(censored["storage_flow_right_censored_count"], 1)
        self.assertEqual(censored["storage_flow_completion_rate"], 0.75)
        for field in self.PRIMARY_FIELDS:
            self.assertTrue(math.isnan(censored[field]), field)


if __name__ == "__main__":
    unittest.main()
