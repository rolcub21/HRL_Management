"""Static protocol checks for the isolated R50 miss extension."""

import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


class R50Latent8MissContractTest(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(
            (HERE / "r50_latent8_miss_protocol.json").read_text()
        )

    def test_exact_arm_tasks_and_episode_count(self):
        self.assertEqual(
            self.protocol["factorial_arms"],
            ["latent_radius__nearest_recorded"],
        )
        self.assertEqual(self.protocol["task_ids"], [0, 2])
        self.assertEqual(self.protocol["excluded_task_ids"], [1, 4])
        self.assertEqual(self.protocol["episodes_per_task"], 8)
        self.assertEqual(self.protocol["horizon_commands"], 10)

    def test_no_oracle_or_failed_conformance_item(self):
        self.assertEqual(self.protocol["oracle_inputs"], [])
        self.assertTrue(all(
            row["status"] == "pass"
            for row in self.protocol["conformance_audit"]
        ))


if __name__ == "__main__":
    unittest.main()
