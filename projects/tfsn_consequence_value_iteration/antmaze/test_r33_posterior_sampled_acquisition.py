import json
import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r33_posterior_sampled_acquisition as r33,
)


class ResidualField:
    def __init__(self, values):
        self.values = np.asarray(values, np.float32)

    def search(self, xy):
        if len(xy) != len(self.values):
            raise AssertionError("unexpected support")
        return self.values


class R33PosteriorSampledAcquisitionTest(unittest.TestCase):
    def test_contract_is_development_only_and_goal_blind(self):
        protocol = r33.load_contract()
        self.assertEqual(protocol["gate_task_ids"], [3, 5])
        self.assertEqual(protocol["oracle_inputs"], [])
        self.assertFalse(protocol["confirmation_panel_accessed"])
        self.assertFalse(
            protocol["hidden_goal_visible_to_planner_executor_or_models"]
        )

    def test_posterior_is_residual_mass_over_unique_cells(self):
        support = np.asarray([[0, 0], [1, 0], [2, 0]], np.float32)
        mass, total, exhausted = r33.posterior_mass(
            support, ResidualField([0.0, 0.25, 0.75])
        )
        np.testing.assert_allclose(mass, [0.0, 0.25, 0.75])
        self.assertEqual(total, 1.0)
        self.assertFalse(exhausted)

    def test_exhausted_posterior_has_maximum_entropy_fallback(self):
        support = np.asarray([[0, 0], [1, 0]], np.float32)
        mass, total, exhausted = r33.posterior_mass(
            support, ResidualField([0.0, 0.0])
        )
        np.testing.assert_array_equal(mass, [1.0, 1.0])
        self.assertEqual(total, 2.0)
        self.assertTrue(exhausted)

    def test_hypothesis_field_is_not_positive_evidence(self):
        field = r33.PosteriorGoalField(0.5, [1.0, 2.0])
        self.assertTrue(field.contacted)
        self.assertEqual(field.coords, [])
        self.assertEqual(field.rewards, [])
        self.assertAlmostEqual(float(field.field([[1.0, 2.0]])[0]), 1.0)

    def test_protocol_file_matches_loaded_contract(self):
        self.assertEqual(json.loads(r33.PROTOCOL.read_text()), r33.PROTOCOL_VALUE)


if __name__ == "__main__":
    unittest.main()
