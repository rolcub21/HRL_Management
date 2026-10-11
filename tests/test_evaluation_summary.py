import json
import unittest

from summarize_evaluations import summarize_group


class EvaluationSummaryTests(unittest.TestCase):
    def test_summary_separates_block_and_episode_variability(self):
        rows = [
            {
                "method": "hrl",
                "lambda": "0.2",
                "mu": "20",
                "target_window": "20",
                "return": "10",
                "success": "1",
                "mean_signed_deviation": "0",
                "mean_absolute_error": "10",
                "delivery_deviations": json.dumps([-10, 10]),
            },
            {
                "method": "hrl",
                "lambda": "0.2",
                "mu": "20",
                "target_window": "20",
                "return": "0",
                "success": "0",
                "mean_signed_deviation": "30",
                "mean_absolute_error": "30",
                "delivery_deviations": json.dumps([30]),
            },
        ]

        result = summarize_group(rows)
        self.assertEqual(result["episode_count"], 2)
        self.assertEqual(result["delivery_count"], 3)
        self.assertAlmostEqual(result["mean_return"], 5.0)
        self.assertAlmostEqual(result["success_rate"], 0.5)
        self.assertAlmostEqual(result["mean_episode_signed_deviation"], 15.0)
        self.assertAlmostEqual(
            result["signed_deviation_std_across_episodes"], 15.0
        )
        self.assertAlmostEqual(result["mean_absolute_error"], 50.0 / 3.0)
        self.assertAlmostEqual(result["tardy_delivery_rate"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
