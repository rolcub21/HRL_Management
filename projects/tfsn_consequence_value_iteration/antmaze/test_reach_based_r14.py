import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze.reach_bridge_candidates import (
    EmpiricalReachQuery,
    representative_reach_offsets,
)


class ReachBasedBridgeTest(unittest.TestCase):
    def test_representatives_are_deterministic_observed_displacements(self):
        angle = np.linspace(-np.pi, np.pi, 512, endpoint=False)
        radius = np.linspace(0.0, 8.0, 512)
        displacement = np.stack(
            (radius * np.cos(angle), radius * np.sin(angle)), axis=1
        ).astype(np.float32)
        first = representative_reach_offsets(displacement, 32)
        second = representative_reach_offsets(displacement, 32)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, (32, 2))
        for value in first:
            self.assertTrue(np.any(np.all(displacement == value, axis=1)))
        selected_radius = np.linalg.norm(first, axis=1)
        self.assertGreater(float(selected_radius[-1]), float(selected_radius[0]))

    def test_query_uses_fixed_reach_not_nearest_landmark_count(self):
        grid_x, grid_y = np.meshgrid(
            np.arange(-5.0, 5.01, 0.25),
            np.arange(-5.0, 5.01, 0.25),
        )
        endpoint = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
        offsets = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]],
            np.float32,
        )
        query = EmpiricalReachQuery(endpoint, offsets)
        distance, local = query.query(np.asarray([0.0, 0.0]), k=4)
        selected = endpoint[local]
        self.assertTrue(np.allclose(selected, offsets))
        self.assertTrue(np.allclose(distance, [0.0, 1.0, 2.0, 3.0]))

    def test_query_batch_contract(self):
        endpoint = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
            np.float32,
        )
        query = EmpiricalReachQuery(
            endpoint, np.asarray([[0.0, 0.0], [1.0, 0.0]], np.float32)
        )
        distance, local = query.query(
            np.asarray([[0.0, 0.0], [5.0, 0.0]], np.float32), k=2
        )
        self.assertEqual(distance.shape, (2, 2))
        self.assertTrue(np.array_equal(local, [[0, 1], [2, 3]]))


if __name__ == "__main__":
    unittest.main()
