import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r67_e549_hub_routed_exploration as r67,
)


class R67HubRoutingTest(unittest.TestCase):
    def test_protocol_is_explicitly_development_only(self):
        self.assertEqual(r67.PROTOCOL_VALUE["task_ids"], [1, 2, 8])
        self.assertEqual(
            r67.PROTOCOL_VALUE["hub_candidate_count_computational"], 20
        )
        self.assertIsNone(r67.PROTOCOL_VALUE["reachability_cutoff"])
        self.assertFalse(r67.PROTOCOL_VALUE["confirmation_panel_accessed"])

    def test_area_posterior_is_normalized(self):
        nodes = np.zeros((4, 29), np.float32)
        nodes[:, :2] = [[0.0, 0.0], [0.01, 0.01], [1.0, 0.0], [2.0, 0.0]]
        posterior = r67._area_posterior(nodes, 0.1)
        self.assertAlmostEqual(float(posterior.goal_probability.sum()), 1.0)
        # The two points in the first occupied cell share that cell's mass.
        self.assertAlmostEqual(
            float(posterior.goal_probability[0]),
            float(posterior.goal_probability[1]),
        )
        self.assertGreater(
            float(posterior.goal_probability[2]),
            float(posterior.goal_probability[0]),
        )

    def test_reachability_backup_uses_failure_self_loop(self):
        layers = r67._one_target_layers(
            target=1,
            horizon=2,
            node_count=2,
            source=np.asarray([0], np.int32),
            endpoint=np.asarray([1], np.int32),
            edge_weight=np.asarray([0.5], np.float32),
            bridge_target=np.asarray([[0], [1]], np.int32),
            bridge_weight=np.asarray([[0.0], [1.0]], np.float32),
        )
        self.assertAlmostEqual(float(layers[0, 0]), 0.0)
        self.assertAlmostEqual(float(layers[1, 0]), 0.5)
        self.assertAlmostEqual(float(layers[2, 0]), 0.75)
        self.assertAlmostEqual(float(layers[:, 1].min()), 1.0)


if __name__ == "__main__":
    unittest.main()
