import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze.hidden_goal_r22 import (
    dynamic_stitched_values,
    frozen_search_sigma,
)


class DummyGraph:
    nodes = np.asarray([
        [0.0, 0.0], [1.0, 0.0], [3.0, 0.0]
    ], np.float32)
    source_index = np.asarray([0, 1], np.int32)
    endpoint_index = np.asarray([1, 2], np.int32)
    edge_reliability = np.asarray([1.0, 1.0], np.float32)
    node_count = 3


class HiddenGoalR22Test(unittest.TestCase):
    def test_dynamic_field_uses_frozen_max_backup(self):
        graph = DummyGraph()
        targets = np.asarray([[2], [0], [1]], np.int32)
        weight = np.asarray([[0.1], [0.1], [0.1]], np.float32)
        values, coverage = dynamic_stitched_values(
            graph, targets, weight, np.asarray([0.0, 0.0, 1.0]), 2
        )
        self.assertAlmostEqual(float(values[1, 1]), 1.0)
        self.assertAlmostEqual(float(values[2, 0]), 1.0)
        self.assertEqual(len(coverage), 2)

    def test_search_sigma_is_exactly_frozen_r22_sigma(self):
        from projects.tfsn_consequence_value_iteration.antmaze import (
            reach_based_r14 as r22,
        )
        self.assertEqual(frozen_search_sigma(), r22.SIGMA)


if __name__ == "__main__":
    unittest.main()
