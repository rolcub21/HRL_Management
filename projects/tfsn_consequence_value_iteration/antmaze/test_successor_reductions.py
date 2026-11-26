import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze.successor_reductions import (
    finite_horizon_edge_values,
)


class DummyGraph:
    node_count = 3
    source_index = np.asarray([0, 1], np.int32)
    endpoint_index = np.asarray([1, 2], np.int32)
    edge_reliability = np.asarray([1.0, 0.5], np.float32)


class SuccessorReductionTest(unittest.TestCase):
    def test_max_and_mean_use_the_same_edges(self):
        graph = DummyGraph()
        targets = np.asarray([[1, 2], [0, 2], [0, 1]], np.int32)
        weight = np.asarray(
            [[0.5, 0.25], [1.0, 0.5], [0.25, 1.0]], np.float32
        )
        field = np.asarray([0.0, 1.0, 0.0], np.float32)
        maximum = finite_horizon_edge_values(
            graph, targets, weight, field, 1, "max"
        )
        average = finite_horizon_edge_values(
            graph, targets, weight, field, 1, "mean"
        )
        # Node 0 has one recorded and two model edges: 1, .5, and 0.
        self.assertAlmostEqual(float(maximum[1, 0]), 1.0)
        self.assertAlmostEqual(float(average[1, 0]), 0.5)
        # Node 2 has only its two model edges: 0 and 1.
        self.assertAlmostEqual(float(maximum[1, 2]), 1.0)
        self.assertAlmostEqual(float(average[1, 2]), 0.5)

    def test_mean_is_uniform_over_duplicate_edge_slots(self):
        graph = DummyGraph()
        targets = np.asarray([[1, 1], [2, 2], [1, 1]], np.int32)
        weight = np.ones((3, 2), np.float32)
        field = np.asarray([0.0, 1.0, 0.0], np.float32)
        average = finite_horizon_edge_values(
            graph, targets, weight, field, 1, "mean"
        )
        # The two bridge slots are kept as the two options R22 maximizes over.
        self.assertAlmostEqual(float(average[1, 2]), 1.0)


if __name__ == "__main__":
    unittest.main()
