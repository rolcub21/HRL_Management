import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze.hidden_goal_passage_r26 import (
    PassageGeometry,
    passage_stitched_values,
)


class DummyGraph:
    nodes = np.zeros((3, 29), np.float32)
    source_index = np.asarray([0, 0], np.int32)
    endpoint_index = np.asarray([1, 2], np.int32)
    edge_reliability = np.ones(2, np.float32)
    node_count = 3


class ConstantField:
    def search(self, xy):
        return np.ones(len(np.asarray(xy).reshape(-1, 2)), np.float32)


class PassageGainTest(unittest.TestCase):
    def test_max_backup_prefers_larger_swept_mass(self):
        graph = DummyGraph()
        targets = np.asarray([[1, 2], [1, 1], [2, 2]], np.int32)
        weight = np.zeros_like(targets, np.float32)
        values, _coverage = passage_stitched_values(
            graph,
            targets,
            weight,
            recorded_gain=np.asarray([0.2, 2.0], np.float32),
            bridge_gain=np.zeros_like(targets, np.float32),
            horizon=1,
        )
        self.assertAlmostEqual(float(values[1, 0]), 2.0)

    def test_live_integral_of_constant_field_is_segment_length(self):
        geometry = object.__new__(PassageGeometry)
        value = geometry.live_segment_gain(
            np.asarray([0.0, 0.0], np.float32),
            np.asarray([[3.0, 4.0], [0.0, 0.0]], np.float32),
            ConstantField(),
        )
        np.testing.assert_allclose(value, [5.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
