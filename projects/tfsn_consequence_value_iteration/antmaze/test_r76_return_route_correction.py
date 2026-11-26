import math
import unittest
from types import SimpleNamespace

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r76_return_route_correction as r76,
)


class ReturnRouteCorrectionTest(unittest.TestCase):
    def test_return_phase_requires_observed_positive_evidence(self):
        supplied_hub = SimpleNamespace(
            contacted=True,
            coords=np.empty((0, 2), np.float32),
            rewards=np.empty(0, np.float32),
        )
        contacted_task = SimpleNamespace(
            contacted=True,
            coords=[[0.0, 0.0], [1.0, 2.0]],
            rewards=[0.0, 1.0],
        )
        self.assertFalse(r76.has_observed_positive_evidence(supplied_hub))
        self.assertTrue(r76.has_observed_positive_evidence(contacted_task))
        np.testing.assert_array_equal(
            r76.ReturnRouteCorrectedPlanner._first_positive_xy(contacted_task),
            np.asarray([1.0, 2.0], np.float32),
        )

    def test_protocol_has_no_similarity_cutoff(self):
        self.assertIsNone(
            r76.PROTOCOL_VALUE["return_route_similarity_cutoff"]
        )
        self.assertEqual(
            r76.PROTOCOL_VALUE["route_source_scale"],
            "R52_empirical_reach_covariance",
        )
        self.assertEqual(
            r76.PROTOCOL_VALUE["route_target_scale"],
            "R52_empirical_reach_covariance",
        )

    def test_failed_prefix_lowers_similar_route_only_at_matching_horizon(self):
        base = np.full((3, 2), 0.8, np.float32)
        records = [{
            "source_xy": [0.0, 0.0],
            "target_xy": [4.0, 0.0],
            "remaining_commands": 2,
            "arrived": False,
        }]
        corrected, support, _arrival = r76.corrected_route_layers(
            base,
            np.asarray([[0.0, 0.0], [3.0, 0.0]]),
            np.asarray([4.0, 0.0]),
            records,
            np.eye(2),
        )
        self.assertEqual(float(support[1, 0]), 0.0)
        self.assertAlmostEqual(float(support[2, 0]), 1.0, places=7)
        self.assertAlmostEqual(
            float(support[2, 1]), math.exp(-4.5), places=7
        )
        self.assertAlmostEqual(float(corrected[2, 0]), 0.4, places=7)
        np.testing.assert_array_equal(corrected[1], base[1])

    def test_successful_prefix_raises_route_probability(self):
        base = np.asarray([[0.0], [0.2]], np.float32)
        records = [{
            "source_xy": [0.0, 0.0],
            "target_xy": [1.0, 0.0],
            "remaining_commands": 1,
            "arrived": True,
        }]
        corrected, support, arrival = r76.corrected_route_layers(
            base,
            np.asarray([[0.0, 0.0]]),
            np.asarray([1.0, 0.0]),
            records,
            np.eye(2),
        )
        self.assertAlmostEqual(float(support[1, 0]), 1.0, places=7)
        self.assertAlmostEqual(float(arrival[1, 0]), 1.0, places=7)
        self.assertAlmostEqual(float(corrected[1, 0]), 0.6, places=7)


if __name__ == "__main__":
    unittest.main()
