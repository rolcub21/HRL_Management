import json
import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r39_live_posterior_passage as r39,
)


class R39LivePosteriorPassageTest(unittest.TestCase):
    def test_contract_is_exact_and_goal_blind(self):
        protocol = r39.load_contract()
        self.assertEqual(protocol["base_method_revision"], r39.r37.METHOD_REVISION)
        self.assertEqual(protocol["oracle_inputs"], [])
        self.assertFalse(protocol["confirmation_panel_accessed"])

    def test_normalized_detection_mass(self):
        mass = np.array([1.0, 0.0, 1.0, 1.0])
        masks = np.array([
            [True, True, False, False],
            [False, False, True, True],
        ])
        np.testing.assert_allclose(
            r39.normalized_detection_mass(mass, masks), [1 / 3, 2 / 3]
        )

    def test_zero_mass_is_safe(self):
        result = r39.normalized_detection_mass(
            np.zeros(3), np.ones((2, 3), bool)
        )
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_protocol_file_matches_loaded_contract(self):
        self.assertEqual(json.loads(r39.PROTOCOL.read_text()), r39.PROTOCOL_VALUE)


if __name__ == "__main__":
    unittest.main()
