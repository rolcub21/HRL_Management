import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze.directed_successor_gpi import (
    directed_policy_edges,
    gpi_successor_values,
    spread_anchor_indices,
)


class DummyGraph:
    nodes = np.asarray([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0]
    ], np.float32)
    source_index = np.asarray([0, 1], np.int32)
    endpoint_index = np.asarray([2, 3], np.int32)
    edge_reliability = np.asarray([1.0, 0.5], np.float32)
    node_count = 4


class DirectedSuccessorTest(unittest.TestCase):
    def test_spread_anchors_are_deterministic_exact_points(self):
        xy = np.asarray([
            [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [9.0, 0.0]
        ], np.float32)
        first = spread_anchor_indices(xy, 3)
        second = spread_anchor_indices(xy, 3)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(np.unique(first)), 3)
        self.assertTrue(np.all(np.isin(first, np.arange(len(xy)))))

    def test_policy_uses_model_weighted_directed_progress(self):
        graph = DummyGraph()
        targets = np.asarray([
            [1, 2], [3, 2], [0, 1], [1, 2]
        ], np.int32)
        weights = np.asarray([
            [0.9, 1.0], [0.5, 1.0], [1.0, 1.0], [1.0, 1.0]
        ], np.float32)
        target, weight, kind, _score = directed_policy_edges(
            graph, targets, weights, np.asarray([3], np.int32), node_chunk=2
        )
        # From node zero, the 0.9-reliable bridge toward x=1 beats the
        # observed edge toward y=1, which makes no progress to the anchor.
        self.assertEqual(int(target[0, 0]), 1)
        self.assertAlmostEqual(float(weight[0, 0]), 0.9)
        self.assertEqual(int(kind[0, 0]), 0)

    def test_gpi_is_max_over_fixed_policy_readouts(self):
        targets = np.asarray([[1, 2, 2], [2, 0, 1]], np.int32)
        weights = np.ones_like(targets, np.float32)
        field = np.asarray([0.0, 1.0, 0.0], np.float32)
        values, winner, _coverage = gpi_successor_values(
            targets, weights, field, horizon=2
        )
        self.assertAlmostEqual(float(values[1, 0]), 1.0)
        self.assertEqual(int(winner[1, 0]), 0)
        # Policy zero reaches field at its first step; policy one reaches it
        # from node zero only on its second step through node two.
        self.assertGreaterEqual(float(values[2, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
