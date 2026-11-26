"""Static checks for the PointMaze R13 command-region screen."""

import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


class PointCommandRegionProtocolTest(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(
            (HERE / "command_region_protocol_r13.json").read_text()
        )

    def test_exact_arm_panel_and_horizon(self):
        self.assertEqual(
            self.protocol["factorial_arms"],
            ["latent_radius__command_region"],
        )
        self.assertEqual(self.protocol["episodes_per_goal"], 8)
        self.assertEqual(self.protocol["evaluation_command_cap"], 20)
        self.assertEqual(
            self.protocol["evaluation_command_cap"],
            self.protocol["benchmark_horizon_commands"],
        )

    def test_no_oracle_or_failed_audit(self):
        self.assertEqual(self.protocol["oracle_inputs"], [])
        self.assertTrue(all(
            row["status"] == "pass"
            for row in self.protocol["conformance_audit"]
        ))

    def test_directed_assignment_and_node_local_consumption_are_declared(self):
        manifest = json.loads(
            (HERE.parent / "shared" /
             "method_manifest_command_regions_r13.json").read_text()
        )["global_decision_principles"]
        self.assertIn("directed_outward", manifest["assignment"])
        self.assertIn("one_over", manifest["link_cost"])
        self.assertIn("selected_target_node", manifest["imagined_consumption"])
        self.assertEqual(manifest["fixed_distance_thresholds"], [])


if __name__ == "__main__":
    unittest.main()
