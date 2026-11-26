import json
import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r38_route_posterior_mass as r38,
)


class R38RoutePosteriorMassTest(unittest.TestCase):
    def test_contract_is_exact_and_goal_blind(self):
        protocol = r38.load_contract()
        self.assertEqual(protocol["base_method_revision"], r38.r37.METHOD_REVISION)
        self.assertEqual(protocol["oracle_inputs"], [])
        self.assertFalse(protocol["confirmation_panel_accessed"])

    def test_polyline_detection_tube(self):
        points = np.array([[0.5, 0.5], [0.5, 1.01], [2.0, 0.0]])
        line = np.array([[0.0, 0.0], [1.0, 0.0]])
        np.testing.assert_array_equal(
            r38.points_within_polyline(points, line, 0.5),
            [True, False, False],
        )

    def test_route_mass_counts_overlap_once_and_weights_reach(self):
        mass = np.ones(4)
        masks = [
            np.array([True, True, False, False]),
            np.array([False, True, True, False]),
        ]
        # First edge contributes 2 * .5.  The second is reached with
        # probability .5 * .5 and contributes only the newly covered cell 2.
        self.assertAlmostEqual(
            r38.expected_first_coverage(mass, masks, [0.5, 0.5]), 1.25
        )

    def test_protocol_file_matches_loaded_contract(self):
        self.assertEqual(json.loads(r38.PROTOCOL.read_text()), r38.PROTOCOL_VALUE)


if __name__ == "__main__":
    unittest.main()
