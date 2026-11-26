import math
import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r74_reach_scale_route_correction as r74,
)


class ReachScaleRouteCorrectionTest(unittest.TestCase):
    def test_protocol_uses_one_reach_covariance_on_both_sides(self):
        self.assertEqual(
            r74.PROTOCOL_VALUE["route_source_scale"],
            "R52_empirical_reach_covariance",
        )
        self.assertEqual(
            r74.PROTOCOL_VALUE["route_hub_scale"],
            "R52_empirical_reach_covariance",
        )
        self.assertIsNone(
            r74.PROTOCOL_VALUE["hub_route_similarity_cutoff"]
        )
        self.assertIsNone(r74.PROTOCOL_VALUE["hub_evaluation_cutoff"])

    def test_weighted_covariance_is_central_second_moment(self):
        displacement = np.asarray([
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -2.0],
            [0.0, 2.0],
        ])
        weight = np.ones(4)
        mean, covariance = r74.weighted_central_covariance(
            displacement, weight
        )
        np.testing.assert_allclose(mean, np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(
            covariance, np.diag([0.5, 2.0]), atol=1e-12
        )

    def test_one_mahalanobis_unit_on_each_side_has_exp_minus_one(self):
        value = r74.reach_route_similarity(
            np.asarray([[2.0, 0.0]]),
            np.asarray([[0.0, 3.0]]),
            np.asarray([[0.0, 0.0]]),
            np.asarray([[0.0, 0.0]]),
            np.diag([0.25, 1.0 / 9.0]),
        )
        self.assertAlmostEqual(float(value[0]), math.exp(-1.0), places=7)

    def test_start_and_hub_each_transfer_route_evidence(self):
        records = [{
            "source_xy": [0.0, 0.0],
            "hub_xy": [4.0, 4.0],
            "remaining_commands": 20,
            "arrived": False,
            "censored": False,
        }]
        support, arrivals = r74.route_mass_from_records(
            records,
            source_xy=np.asarray([1.0, 0.0]),
            hub_xy=np.asarray([[4.0, 5.0]]),
            remaining=20,
            inverse_covariance=np.eye(2),
        )
        self.assertAlmostEqual(float(support[0]), math.exp(-1.0), places=7)
        self.assertEqual(float(arrivals[0]), 0.0)

    def test_route_evidence_does_not_cross_decision_horizons(self):
        records = [{
            "source_xy": [0.0, 0.0],
            "hub_xy": [1.0, 0.0],
            "remaining_commands": 10,
            "arrived": True,
            "censored": False,
        }]
        support, arrivals = r74.route_mass_from_records(
            records,
            source_xy=np.asarray([0.0, 0.0]),
            hub_xy=np.asarray([[1.0, 0.0]]),
            remaining=20,
            inverse_covariance=np.eye(2),
        )
        np.testing.assert_array_equal(support, np.zeros(1))
        np.testing.assert_array_equal(arrivals, np.zeros(1))


if __name__ == "__main__":
    unittest.main()
