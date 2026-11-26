import unittest

import numpy as np
from scipy.spatial import cKDTree

from projects.tfsn_consequence_value_iteration.antmaze.hidden_goal_empirical_passage_r27 import (
    EmpiricalPassageGeometry,
    EmpiricalPassagePlanner,
    PASSAGE_BINS,
    _bridge_mapping,
    _linear_cell_index,
    _local_passage_tree,
)
from projects.tfsn_consequence_value_iteration.antmaze.mobility_stitched_planning import (
    STATE_DIM,
)


class DummyGraph:
    def __init__(self):
        self.nodes = np.zeros((5, STATE_DIM), np.float32)
        self.nodes[:, :2] = np.asarray(
            [[0, 0], [1, 0], [0, 1], [3, 2], [4, 2]], np.float32
        )
        self.source_index = np.asarray([0, 0, 3], np.int32)
        self.endpoint_index = np.asarray([1, 2, 4], np.int32)
        self.edge_count = 3
        self.node_count = 5


class LocalPassageMatchingTest(unittest.TestCase):
    def test_mapping_uses_local_start_and_realized_displacement(self):
        graph = DummyGraph()
        tree, _start, _displacement = _local_passage_tree(graph, 1.0)
        targets = np.asarray([[1], [2], [2], [4], [4]], np.int32)
        edge, distance = _bridge_mapping(
            tree, graph, targets, begin=0, stop=5, sigma=1.0
        )
        self.assertEqual(edge.reshape(-1).tolist(), [0, 1, 0, 2, 2])
        self.assertTrue(np.isfinite(distance).all())

    def test_live_gain_follows_matched_observed_path(self):
        graph = DummyGraph()
        geometry = object.__new__(EmpiricalPassageGeometry)
        geometry.tree, _start, _displacement = _local_passage_tree(graph, 1.0)
        geometry.relative = np.zeros((3, PASSAGE_BINS, 2), np.float32)
        geometry.bin_length = np.zeros((3, PASSAGE_BINS), np.float32)
        # The observed unit-x transition follows y=1 before returning to y=0.
        geometry.relative[0, :, 0] = np.linspace(
            0.0, 1.0, PASSAGE_BINS, dtype=np.float32
        )
        geometry.relative[0, :, 1] = 1.0
        geometry.bin_length[0] = 1.0 / PASSAGE_BINS

        class UpperPassageField:
            @staticmethod
            def search(xy):
                return (np.asarray(xy)[:, 1] > 0.5).astype(np.float32)

        gain = geometry.live_segment_gain(
            np.asarray([0.0, 0.0], np.float32),
            np.asarray([[1.0, 0.0]], np.float32),
            UpperPassageField(),
        )
        self.assertAlmostEqual(float(gain[0]), 1.0, places=6)

    def test_linear_lattice_index_handles_multidimensional_cells(self):
        cells = np.asarray([[[0, 0], [1, 2]], [[2, 1], [0, 2]]])
        index = _linear_cell_index(cells, minimum=[0, 0], shape=[3, 3])
        np.testing.assert_array_equal(index, [[0, 5], [7, 2]])


class PositiveEvidenceScoringTest(unittest.TestCase):
    def test_exact_evidence_uses_same_model_and_continuation(self):
        planner = object.__new__(EmpiricalPassagePlanner)

        class Consequence:
            members = [object()]

            @staticmethod
            def sample_member(inputs, member, samples, rng):
                del member, rng
                command = inputs[:, STATE_DIM:STATE_DIM + 2]
                start = inputs[:, :STATE_DIM]
                delta = np.zeros(
                    (len(inputs), samples, STATE_DIM), np.float32
                )
                delta[..., :2] = command[:, None] - start[:, None, :2]
                return delta

        class Mobility:
            @staticmethod
            def predict(states):
                return np.ones(len(states), np.float32)

        class Fields:
            @staticmethod
            def reward_rate(commands):
                return np.ones(len(commands), np.float32)

        planner.consequence = Consequence()
        planner.mobility = Mobility()
        planner.node_xy_tree = cKDTree(np.asarray([[1.0, 0.0]]))
        planner.values = np.zeros((11, 1), np.float32)
        state = np.zeros(STATE_DIM, np.float32)
        q, connection = planner._positive_evidence_value(
            state,
            np.asarray([[1.0, 0.0]], np.float32),
            Fields(),
            remaining=10,
            seed=1,
        )
        np.testing.assert_allclose(connection, [1.0], atol=1e-6)
        np.testing.assert_allclose(q, [1.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
