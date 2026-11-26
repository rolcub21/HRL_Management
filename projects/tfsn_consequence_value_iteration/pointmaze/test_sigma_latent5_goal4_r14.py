"""Static contract checks for the bounded R14 comparison."""

import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


class SigmaLatentFiveCommandContractTest(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(
            (HERE / "sigma_latent5_goal4_protocol_r14.json").read_text()
        )

    def test_exact_two_arm_memory_comparison(self):
        self.assertEqual(self.protocol["factorial_arms"], [
            "sigma_kernel__nearest_recorded",
            "latent_radius__nearest_recorded",
        ])
        self.assertTrue(all(
            arm.endswith("__nearest_recorded")
            for arm in self.protocol["factorial_arms"]
        ))

    def test_exact_four_goal_eight_episode_five_command_panel(self):
        self.assertEqual(len(self.protocol["corner_goal_names"]), 2)
        self.assertEqual(len(self.protocol["ordinary_goal_names"]), 2)
        self.assertEqual(len(set(self.protocol["goal_names"])), 4)
        self.assertEqual(self.protocol["episodes_per_goal"], 8)
        self.assertEqual(self.protocol["evaluation_command_cap"], 5)
        self.assertEqual(self.protocol["benchmark_horizon_commands"], 20)

    def test_no_oracle_region_or_failed_conformance_item(self):
        self.assertEqual(self.protocol["oracle_inputs"], [])
        self.assertTrue(all(
            row["status"] == "pass"
            for row in self.protocol["conformance_audit"]
        ))
        reference = (
            HERE / "R14_SIGMA_LATENT5_GOAL4_REFERENCE.md"
        ).read_text().lower()
        self.assertIn("command regions", reference)
        self.assertIn("not loaded", reference)

    def test_not_misrepresented_as_bitwise_r11(self):
        deviations = " ".join(self.protocol["declared_deviations"])
        self.assertIn("not_a_bitwise_R11", deviations)


if __name__ == "__main__":
    unittest.main()
