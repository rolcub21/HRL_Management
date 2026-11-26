import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.shared.backups import (
    additive_field_backup,
    bounded_search_backup,
    ensemble_reduce,
)
from projects.tfsn_consequence_value_iteration.shared.factorized import (
    consequence_features,
    diagonal_gaussian_circle_probability,
)


class SharedMethodTest(unittest.TestCase):
    def test_pointmaze_feature_contract(self):
        states = np.asarray([[1.0, 2.0, 0.2, -0.1]], np.float32)
        commands = np.asarray([[4.0, 6.0]], np.float32)
        features = consequence_features(states, commands)
        self.assertEqual(features.shape, (1, 9))
        np.testing.assert_allclose(features[0, 6:8], [3.0, 4.0])
        self.assertAlmostEqual(float(features[0, 8]), 5.0)

    def test_bounded_search_never_exceeds_one(self):
        values = bounded_search_backup(
            1, [0], [0], [1.0], [0.2], horizon=20
        )
        self.assertAlmostEqual(float(values[-1, 0]), 1.0 - 0.8**20, places=6)
        self.assertLessEqual(float(values.max()), 1.0)

    def test_additive_field_is_not_bounded(self):
        values = additive_field_backup(
            1, [0], [0], [1.0], [0.5], horizon=20
        )
        self.assertAlmostEqual(float(values[-1, 0]), 10.0, places=6)

    def test_ensemble_modes(self):
        values = np.asarray([[0.2, 0.5, 0.8]], np.float32)
        self.assertAlmostEqual(float(ensemble_reduce(values, "mean")[0]), 0.5)
        self.assertAlmostEqual(float(ensemble_reduce(values, "pessimistic")[0]), 0.2)

    def test_gaussian_circle_probability_is_bounded_and_centered(self):
        centered = diagonal_gaussian_circle_probability(
            np.zeros((1, 1, 2)),
            np.ones((1, 1, 2)) * 0.1,
            np.zeros((1, 2)),
            0.4,
        )
        remote = diagonal_gaussian_circle_probability(
            np.ones((1, 1, 2)) * 4.0,
            np.ones((1, 1, 2)) * 0.1,
            np.zeros((1, 2)),
            0.4,
        )
        self.assertGreater(float(centered[0, 0]), 0.99)
        self.assertLess(float(remote[0, 0]), 1e-6)


if __name__ == "__main__":
    unittest.main()
