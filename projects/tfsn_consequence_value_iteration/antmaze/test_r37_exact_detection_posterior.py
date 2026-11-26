import json
import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r37_exact_detection_posterior as r37,
)


class R37ExactDetectionPosteriorTest(unittest.TestCase):
    def test_contract_is_exact_and_goal_blind(self):
        protocol = r37.load_contract()
        self.assertEqual(protocol["base_method_revision"], r37.r36.METHOD_REVISION)
        self.assertEqual(
            protocol["posterior_likelihood"],
            "hard_indicator_outside_union_of_realized_detection_footprints",
        )
        self.assertEqual(protocol["oracle_inputs"], [])
        self.assertFalse(protocol["confirmation_panel_accessed"])

    def test_exact_field_eliminates_inside_and_keeps_outside(self):
        field = r37.ExactDetectionFields(
            r37.r31.SIGMA, r37.r35.DETECTION_FOOTPRINT_M
        )
        field.add([0.0, 0.0], 0.0)
        field.rebuild()
        radius = r37.r35.DETECTION_FOOTPRINT_M
        np.testing.assert_array_equal(
            field.search([[0.0, 0.0], [0.5 * radius, 0.0], [1.1 * radius, 0.0]]),
            [0.0, 0.0, 1.0],
        )

    def test_return_bandwidth_remains_landing_precision(self):
        field = r37.ExactDetectionFields(
            r37.r31.SIGMA, r37.r35.DETECTION_FOOTPRINT_M
        )
        self.assertEqual(field.sigma, r37.r31.SIGMA)
        self.assertEqual(field.coarse_sigma, r37.r35.DETECTION_FOOTPRINT_M)

    def test_protocol_file_matches_loaded_contract(self):
        self.assertEqual(json.loads(r37.PROTOCOL.read_text()), r37.PROTOCOL_VALUE)


if __name__ == "__main__":
    unittest.main()
