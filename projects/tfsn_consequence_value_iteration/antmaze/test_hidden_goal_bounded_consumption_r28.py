import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze.hidden_goal_bounded_consumption_r28 import (
    BoundedPassageGeometry,
    bounded_passage_values,
)


class OneNodeGraph:
    node_count = 1
    source_index = np.asarray([0], np.int32)
    endpoint_index = np.asarray([0], np.int32)
    edge_reliability = np.asarray([1.0], np.float32)


class BoundedBackupTest(unittest.TestCase):
    def test_repeated_gain_is_bounded_by_one(self):
        graph = OneNodeGraph()
        values, _coverage = bounded_passage_values(
            graph,
            targets=np.asarray([[0]], np.int32),
            weight=np.asarray([[0.0]], np.float32),
            recorded_gain=np.asarray([0.2], np.float32),
            bridge_gain=np.asarray([[0.0]], np.float32),
            horizon=10,
        )
        expected = 1.0 - 0.8**10
        self.assertAlmostEqual(float(values[-1, 0]), expected, places=6)
        self.assertLessEqual(float(values.max()), 1.0)

    def test_failure_stays_at_source(self):
        graph = OneNodeGraph()
        graph.edge_reliability = np.asarray([0.5], np.float32)
        values, _coverage = bounded_passage_values(
            graph,
            targets=np.asarray([[0]], np.int32),
            weight=np.asarray([[0.0]], np.float32),
            recorded_gain=np.asarray([0.2], np.float32),
            bridge_gain=np.asarray([[0.0]], np.float32),
            horizon=2,
        )
        # V1 = .5*.2; V2 = .5*(.2 + .8*V1) + .5*V1.
        self.assertAlmostEqual(float(values[1, 0]), 0.1, places=6)
        self.assertAlmostEqual(float(values[2, 0]), 0.19, places=6)

    def test_passage_mass_normalization_is_dimensionless_and_clipped(self):
        geometry = object.__new__(BoundedPassageGeometry)
        geometry.residual_reachable_mass_m2 = 2.0
        geometry.transverse_kernel_mass_m = 0.25
        value = geometry._normalize(
            np.asarray([0.0, 4.0, 20.0], np.float32)
        )
        np.testing.assert_allclose(value, [0.0, 0.5, 1.0], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
