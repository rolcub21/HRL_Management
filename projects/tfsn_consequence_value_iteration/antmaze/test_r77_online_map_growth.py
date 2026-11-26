import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r77_online_map_growth as r77,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    RecordedTransitionGraph,
)
from projects.tfsn_consequence_value_iteration.antmaze.successor_reductions import (
    finite_horizon_edge_values,
)


class OnlineMapGrowthTest(unittest.TestCase):
    @staticmethod
    def graph():
        nodes = np.zeros((3, 29), np.float32)
        nodes[:, 0] = [0.0, 1.0, 2.0]
        return RecordedTransitionGraph(
            nodes=nodes,
            source_index=[0, 1, 2],
            endpoint_index=[0, 1, 2],
            commands=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            edge_reliability=[1.0, 1.0, 1.0],
            state_scale=np.ones(29, np.float32),
        )

    def test_no_evidence_and_no_links_is_exactly_r75(self):
        graph = self.graph()
        targets = np.asarray([[1], [2], [2]], np.int32)
        weight = np.asarray([[0.8], [0.9], [1.0]], np.float32)
        field = np.asarray([0.0, 0.25, 1.0], np.float32)
        expected = finite_horizon_edge_values(
            graph, targets, weight, field, 3, reduction="max"
        )
        actual = r77.r75_values_with_online_links(
            graph, targets, weight, field, 3
        )
        np.testing.assert_array_equal(actual, expected)

    def test_online_link_creates_a_missing_route(self):
        graph = self.graph()
        targets = np.asarray([[0], [1], [2]], np.int32)
        weight = np.ones((3, 1), np.float32)
        field = np.asarray([0.0, 0.0, 1.0], np.float32)
        before = r77.r75_values_with_online_links(
            graph, targets, weight, field, 2
        )
        after = r77.r75_values_with_online_links(
            graph, targets, weight, field, 2,
            online_source=[0, 1], online_endpoint=[1, 2],
            online_reliability=[1.0, 1.0],
        )
        self.assertEqual(float(before[2, 0]), 0.0)
        self.assertGreater(float(after[2, 0]), 0.0)

    def test_failed_route_lowers_similar_value_but_keeps_it_positive(self):
        nodes = np.asarray([[0.0, 0.0], [3.0, 0.0]], np.float32)
        route, support, _arrival = r77.route_reliability_layers(
            nodes,
            [4.0, 0.0],
            [{
                "source_xy": [0.0, 0.0],
                "target_xy": [4.0, 0.0],
                "remaining_commands": 2,
                "arrived": False,
            }],
            np.eye(2),
            2,
        )
        self.assertAlmostEqual(float(support[2, 0]), 1.0)
        self.assertAlmostEqual(float(route[2, 0]), 0.5)
        self.assertGreater(float(route[2, 0]), 0.0)
        self.assertGreater(float(route[2, 1]), float(route[2, 0]))

    def test_projection_uses_continuous_sigma_association(self):
        source, endpoint, reliability, source_weight, endpoint_weight = (
            r77.project_online_links(
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.1, 0.0]],
                [[0.9, 0.0]],
                sigma=0.2,
            )
        )
        np.testing.assert_array_equal(source, [0])
        np.testing.assert_array_equal(endpoint, [1])
        self.assertAlmostEqual(float(source_weight[0]), np.exp(-0.125))
        self.assertAlmostEqual(float(endpoint_weight[0]), np.exp(-0.125))
        self.assertAlmostEqual(
            float(reliability[0]), np.exp(-0.25), places=6
        )


if __name__ == "__main__":
    unittest.main()
