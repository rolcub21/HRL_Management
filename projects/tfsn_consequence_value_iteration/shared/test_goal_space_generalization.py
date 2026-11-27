"""Regression tests for executor-goal-space dimensionality.

The maze deployments remain two-dimensional.  These tests guarantee that the
shared estimators do not encode that adapter choice and can accept, for
example, a three-dimensional FetchReach or PickAndPlace goal space.
"""

from __future__ import annotations

import unittest

import numpy as np

from .factorized import (
    consequence_features,
    diagonal_gaussian_kernel_expectation,
)
from .frozen_hub_method import (
    alias_hub_candidates,
    area_weights,
    empirical_reach_covariance,
    latent_hub_coverage,
    route_similarity,
)
from .latent_detection import LatentRadiusPosterior, exact_support_diameter
from .learned_passage import PassageMLP, passage_features, passage_targets
from .reach import EmpiricalReachQuery, representative_reach_offsets


class GoalSpaceGeneralizationTest(unittest.TestCase):
    def test_two_dimensional_features_are_bitwise_unchanged(self):
        states = np.asarray([[1.0, 2.0, 0.2, -0.1]], np.float32)
        commands = np.asarray([[4.0, 6.0]], np.float32)
        expected = np.asarray(
            [[1.0, 2.0, 0.2, -0.1, 4.0, 6.0, 3.0, 4.0, 5.0]],
            np.float32,
        )
        np.testing.assert_array_equal(consequence_features(states, commands), expected)

    def test_three_dimensional_features_follow_executor_goal_dimension(self):
        states = np.asarray([[1.0, 2.0, 3.0, 9.0, -4.0]], np.float32)
        commands = np.asarray([[2.0, 4.0, 5.0]], np.float32)
        features = consequence_features(states, commands)
        self.assertEqual(features.shape, (1, 12))
        np.testing.assert_array_equal(features[0, 5:8], commands[0])
        np.testing.assert_array_equal(features[0, 8:11], [1.0, 2.0, 2.0])
        self.assertEqual(float(features[0, 11]), 3.0)

    def test_three_dimensional_gaussian_kernel_is_exact_at_zero_variance_limit(self):
        means = np.zeros((2, 1, 3), np.float32)
        stds = np.full_like(means, 1e-4)
        centers = np.zeros((2, 3), np.float32)
        value = diagonal_gaussian_kernel_expectation(
            means, stds, centers, sigma=0.2
        )
        expected = (0.2**2 / (0.2**2 + 1e-8)) ** 1.5
        np.testing.assert_allclose(value, expected, rtol=2e-6)

    def test_three_dimensional_latent_posterior_and_diameter(self):
        support = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], np.float64
        )
        self.assertEqual(exact_support_diameter(support), np.sqrt(13.0))
        posterior = LatentRadiusPosterior(support, sigma=0.1)
        self.assertEqual(posterior.goal_dim, 3)
        probability = posterior.contact_probability(
            np.asarray([[0.0, 0.0, 0.0]], np.float64)
        )
        self.assertGreater(float(probability), 0.0)
        posterior.update(np.asarray([[0.5, 0.0, 0.0]]), np.asarray([0]))
        self.assertEqual(posterior.negative_observations, 1)

    def test_three_dimensional_hubs_reach_and_coverage(self):
        node = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            np.float64,
        )
        representative, inverse, origin = alias_hub_candidates(node, 0.5)
        self.assertEqual(origin.shape, (3,))
        self.assertEqual(len(representative), 4)
        self.assertEqual(inverse.shape, (5,))
        weights, occupied, area_origin = area_weights(node, 0.5)
        self.assertEqual(occupied, 4)
        self.assertEqual(area_origin.shape, (3,))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

        # Four non-coplanar, nonzero displacements give a full-rank reach
        # covariance without introducing any numerical regularizer.
        graph_nodes = np.vstack((np.zeros((4, 3)), node[1:5]))
        reach = empirical_reach_covariance(
            graph_nodes,
            source=np.arange(4),
            endpoint=np.arange(4, 8),
            commands=node[1:5],
            sigma=0.1,
        )
        self.assertEqual(reach["covariance_m2"].shape, (3, 3))
        similarity = route_similarity(
            source_xy=np.asarray([[0.0, 0.0, 0.0]]),
            hub_xy=node[1:3],
            observed_source_xy=np.asarray([[0.0, 0.0, 0.0]]),
            observed_hub_xy=np.asarray([[0.0, 0.0, 0.0]]),
            inverse_covariance=reach["inverse_covariance_per_m2"],
        )
        self.assertEqual(similarity.shape, (2,))

        posterior = LatentRadiusPosterior(node, sigma=0.05)
        coverage = latent_hub_coverage(posterior, node[representative])
        self.assertEqual(coverage.shape, (4,))
        self.assertTrue(np.isfinite(coverage).all())

    def test_three_dimensional_reach_representatives_and_query(self):
        rng = np.random.default_rng(41)
        displacement = rng.normal(size=(40, 3)).astype(np.float32)
        representative = representative_reach_offsets(displacement, 8)
        self.assertEqual(representative.shape, (8, 3))
        endpoints = rng.normal(size=(20, 3)).astype(np.float32)
        distance, index = EmpiricalReachQuery(
            endpoints, representative
        ).query(np.zeros((2, 3), np.float32))
        self.assertEqual(distance.shape, (2, 8))
        self.assertEqual(index.shape, (2, 8))

    def test_three_dimensional_passage_estimator_contract(self):
        states = np.asarray(
            [[1.0, 2.0, 3.0, 7.0], [2.0, 3.0, 4.0, 8.0]], np.float32
        )
        commands = states[:, :3] + 1.0
        feature = passage_features(states, commands)
        self.assertEqual(feature.shape, (2, 10))
        primitive = np.stack(
            (states[:, :3] + 0.5, states[:, :3] + 1.0), axis=1
        )
        target = passage_targets(states, primitive, sigma=0.5, passage_bins=2)
        self.assertEqual(target.shape, (2, 2, 3))
        import torch

        model = PassageMLP(10, 2, 8, 1, goal_dim=3)
        self.assertEqual(tuple(model(torch.zeros((4, 10))).shape), (4, 2, 3))


if __name__ == "__main__":
    unittest.main()
