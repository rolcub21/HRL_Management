import json
import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r31_aligned_hidden_goal as r31,
)


class R31AlignedHiddenGoalTest(unittest.TestCase):
    def test_contract_binds_full_r30_checkpoint(self):
        protocol, report, sigma = r31.load_contract()
        self.assertEqual(protocol["gate_task_ids"], [3, 5])
        self.assertEqual(protocol["development_task_ids"], list(range(15)))
        self.assertEqual(report["profile"], "full")
        self.assertAlmostEqual(sigma, r31.SIGMA)
        self.assertEqual(
            report["checkpoint_sha256"], r31.sha256(r31.R30_CHECKPOINT)
        )

    def test_artifacts_are_isolated_from_r28(self):
        self.assertIn("r30_deployment_r31", str(r31.graph_path()))
        self.assertIn("r30_deployment_r31", str(r31.artifact_paths()["bridges"]))
        self.assertIn("r30_deployment_r31", str(r31.passage_paths()["metadata"]))

    def test_runtime_bindings_use_aligned_scale_and_model(self):
        r31.bind_runtime()
        self.assertEqual(r31.r22.SIGMA, r31.SIGMA)
        self.assertEqual(r31.mobility.LANDING_SIGMA, r31.SIGMA)
        self.assertEqual(r31.recorded.LANDING_SIGMA, r31.SIGMA)
        self.assertEqual(
            r31.mobility.factorized_checkpoint(None), r31.R30_CHECKPOINT
        )
        value = r31.aligned_alias_kernel(np.asarray([0.0, r31.SIGMA]))
        self.assertAlmostEqual(float(value[0]), 1.0)
        self.assertAlmostEqual(float(value[1]), float(np.exp(-0.5)), places=6)

    def test_protocol_has_no_confirmation_or_oracle_input(self):
        protocol = json.loads(r31.PROTOCOL.read_text())
        self.assertEqual(protocol["oracle_inputs"], [])
        self.assertFalse(protocol["confirmation_panel_accessed"])
        self.assertFalse(protocol["hidden_goal_visible_to_planner_executor_or_models"])

    def test_development_sets_remain_separate_from_confirmation(self):
        self.assertEqual(r31.task_ids("gate2"), (3, 5))
        self.assertEqual(r31.task_ids("all15"), tuple(range(15)))
        self.assertNotEqual(
            r31.evaluation_directory("gate2"),
            r31.evaluation_directory("all15"),
        )


if __name__ == "__main__":
    unittest.main()

