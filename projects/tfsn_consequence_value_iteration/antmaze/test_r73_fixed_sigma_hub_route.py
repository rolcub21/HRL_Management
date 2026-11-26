import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r73_fixed_sigma_hub_route as r73,
)


class FixedSigmaHubRouteTest(unittest.TestCase):
    def test_protocol_is_matched_fixed_sigma_arm(self):
        self.assertFalse(r73.PROTOCOL_VALUE["latent_radius_posterior_used"])
        self.assertEqual(r73.PROTOCOL_VALUE["task_ids"], list(range(9)))
        self.assertEqual(r73.PROTOCOL_VALUE["episodes"], 10)
        self.assertEqual(r73.PROTOCOL_VALUE["commands_per_episode"], 20)

    def test_configuration_installs_fixed_memory_not_latent_memory(self):
        r73.configure_implementation()
        self.assertIs(r73.r67._area_posterior, r73.fixed_area_posterior)
        self.assertIs(
            r73.r67._negative_path_update,
            r73.fixed_negative_path_update,
        )
        self.assertIs(
            r73.r68._all_hub_coverage,
            r73.fixed_all_hub_coverage,
        )
        self.assertEqual(
            r73.implementation.METHOD_REVISION, r73.METHOD_REVISION
        )

    def test_exactly_visited_goal_loses_all_residual_mass(self):
        posterior = r73.AreaWeightedSigmaKernelPosterior(
            np.asarray([[0.0, 0.0], [1.0, 0.0]]), 0.25
        )
        r73.fixed_negative_path_update(
            posterior, np.asarray([[0.0, 0.0]])
        )
        self.assertEqual(float(posterior.goal_probability[0]), 0.0)
        self.assertAlmostEqual(float(posterior.goal_probability[1]), 1.0)

    def test_path_update_is_one_minus_max_sigma_kernel(self):
        posterior = r73.AreaWeightedSigmaKernelPosterior(
            np.asarray([[0.0, 0.0], [2.0, 0.0]]), 1.0
        )
        prior = posterior.goal_probability.copy()
        path = np.asarray([[0.5, 0.0], [1.0, 0.0]])
        distance = np.linalg.norm(
            path[:, None] - posterior.goal_xy[None], axis=2
        ).min(axis=0)
        expected = prior * (1.0 - np.exp(-0.5 * distance**2))
        expected /= expected.sum()
        r73.fixed_negative_path_update(posterior, path)
        np.testing.assert_allclose(
            posterior.goal_probability, expected, atol=1e-12
        )

    def test_vectorized_hub_coverage_matches_posterior_readout(self):
        posterior = r73.AreaWeightedSigmaKernelPosterior(
            np.asarray([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]), 0.5
        )
        hubs = np.asarray([[0.2, 0.0], [2.0, 0.0]])
        expected = np.asarray([
            posterior.contact_probability(hub.reshape(1, 2))
            for hub in hubs
        ])
        actual = r73.fixed_all_hub_coverage(posterior, hubs)
        np.testing.assert_allclose(actual, expected, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
