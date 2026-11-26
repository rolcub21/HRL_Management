import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r68_e549_dynamic_hubs as r68,
)
from projects.tfsn_consequence_value_iteration.shared.latent_detection import (
    LatentRadiusPosterior,
)


class R68DynamicHubTest(unittest.TestCase):
    def test_contract_has_no_method_cutoff_or_fixed_hub_count(self):
        self.assertEqual(r68.PROTOCOL_VALUE["hub_candidate_count"], "data_derived")
        self.assertIsNone(r68.PROTOCOL_VALUE["reachability_cutoff"])
        self.assertEqual(r68.PROTOCOL_VALUE["declared_deviations"], [])

    def test_vectorized_coverage_matches_exact_scalar_readout(self):
        support = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            np.float64,
        )
        posterior = LatentRadiusPosterior(support, 0.1)
        posterior.update(np.asarray([[0.5, 0.5]]), np.asarray([0]))
        hubs = np.asarray([[0.2, 0.2], [0.8, 0.8], [2.0, 2.0]])
        expected = np.asarray([
            posterior.contact_probability(hub.reshape(1, 2)) for hub in hubs
        ])
        actual = r68._all_hub_coverage(posterior, hubs)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
