import json
import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r35_detection_footprint_acquisition as r35,
)
from scripts.antmaze.current_tfsn.fields import TaskFields


class R35DetectionFootprintTest(unittest.TestCase):
    def test_contract_declares_known_observation_model_input(self):
        protocol = r35.load_contract()
        self.assertEqual(protocol["base_method_revision"], r35.r34.METHOD_REVISION)
        self.assertEqual(
            protocol["task_observation_model_inputs"],
            ["reward_contact_radius_m"],
        )
        self.assertEqual(protocol["oracle_inputs"], [])
        self.assertFalse(protocol["confirmation_panel_accessed"])

    def test_planner_separates_search_and_landing_scales(self):
        planner = object.__new__(r35.DetectionFootprintPlanner)
        planner.configure_hidden_fields(r35.r31.SIGMA)
        self.assertEqual(planner.acquisition_search_mode, "coarse")
        self.assertEqual(
            planner.coarse_search_sigma, r35.DETECTION_FOOTPRINT_M
        )

    def test_task_field_uses_broad_search_but_narrow_reward_rate(self):
        fields = TaskFields(
            r35.r31.SIGMA,
            search_mode="coarse",
            coarse_sigma=r35.DETECTION_FOOTPRINT_M,
        )
        fields.add([0.0, 0.0], 0.0)
        fields.rebuild()
        distance = r35.DETECTION_FOOTPRINT_M
        expected = 1.0 - np.exp(-0.5)
        self.assertAlmostEqual(
            float(fields.search([[distance, 0.0]])[0]), expected, places=6
        )
        self.assertEqual(fields.sigma, r35.r31.SIGMA)

    def test_protocol_file_matches_loaded_contract(self):
        self.assertEqual(json.loads(r35.PROTOCOL.read_text()), r35.PROTOCOL_VALUE)


if __name__ == "__main__":
    unittest.main()
