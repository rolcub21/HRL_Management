import math
import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r71_sigma_generalized_correction as r71,
)


class SigmaGeneralizedCorrectionTest(unittest.TestCase):
    def test_protocol_has_no_similarity_or_hub_cutoff(self):
        self.assertIsNone(r71.PROTOCOL_VALUE["similarity_cutoff"])
        self.assertIsNone(r71.PROTOCOL_VALUE["hub_evaluation_cutoff"])
        self.assertEqual(r71.PROTOCOL_VALUE["task_ids"], list(range(9)))
        self.assertEqual(r71.PROTOCOL_VALUE["episodes"], 10)
        self.assertEqual(r71.PROTOCOL_VALUE["commands_per_episode"], 20)

    def test_exact_link_has_unit_similarity(self):
        value = r71.sigma_link_similarity(
            np.asarray([1.0, 2.0]),
            np.asarray([3.0, 4.0]),
            np.asarray([5.0, 6.0]),
            np.asarray([1.0, 2.0]),
            np.asarray([3.0, 4.0]),
            np.asarray([5.0, 6.0]),
            0.3,
        )
        self.assertAlmostEqual(float(value), 1.0, places=7)

    def test_one_sigma_source_displacement_has_gaussian_weight(self):
        value = r71.sigma_link_similarity(
            np.asarray([1.0, 0.0]),
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            1.0,
        )
        self.assertAlmostEqual(float(value), math.exp(-0.5), places=7)

    def test_graph_update_ignores_posture_but_not_spatial_geometry(self):
        nodes = np.zeros((2, 29), np.float32)
        nodes[0, :2] = [0.0, 0.0]
        nodes[1, :2] = [0.0, 0.0]
        nodes[1, 2:] = 1000.0
        targets = np.asarray([
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ], np.float32)
        support = np.zeros((2, 1), np.float32)
        outcome = np.zeros((2, 1), np.float32)
        start = np.zeros(29, np.float32)
        r71.add_sigma_observation_mass(
            support,
            outcome,
            nodes,
            targets,
            start,
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 0.0]),
            0.0,
            0.3,
        )
        np.testing.assert_allclose(support[:, 0], [1.0, 1.0], atol=1e-7)
        np.testing.assert_array_equal(outcome, 0.0)

    def test_failed_observation_adds_support_not_success_mass(self):
        nodes = np.zeros((1, 29), np.float32)
        targets = np.asarray([[[0.0, 0.0]]], np.float32)
        support = np.zeros((1, 1), np.float32)
        outcome = np.zeros((1, 1), np.float32)
        r71.add_sigma_observation_mass(
            support,
            outcome,
            nodes,
            targets,
            np.zeros(29, np.float32),
            np.zeros(2, np.float32),
            np.zeros(2, np.float32),
            0.0,
            0.3,
        )
        self.assertAlmostEqual(float(support[0, 0]), 1.0)
        self.assertAlmostEqual(float(outcome[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
