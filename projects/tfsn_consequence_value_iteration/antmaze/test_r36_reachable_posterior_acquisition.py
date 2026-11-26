import json
import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r36_reachable_posterior_acquisition as r36,
)


class R36ReachablePosteriorTest(unittest.TestCase):
    def test_contract_is_narrow_and_goal_blind(self):
        protocol = r36.load_contract()
        self.assertEqual(protocol["base_method_revision"], r36.r35.METHOD_REVISION)
        self.assertEqual(
            protocol["posterior_hypotheses_per_selection_global"], 16
        )
        self.assertEqual(protocol["oracle_inputs"], [])
        self.assertFalse(protocol["confirmation_panel_accessed"])

    def test_batch_sampling_is_weighted_without_replacement(self):
        rng = np.random.default_rng(7)
        selected = r36.posterior_batch_indices(
            [0.0, 1.0, 2.0, 3.0], 3, rng
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(np.unique(selected)), 3)
        self.assertNotIn(0, selected)

    def test_batch_size_is_capped_by_positive_support(self):
        rng = np.random.default_rng(9)
        selected = r36.posterior_batch_indices([1.0, 0.0, 1.0], 16, rng)
        np.testing.assert_array_equal(np.sort(selected), [0, 2])

    def test_protocol_file_matches_loaded_contract(self):
        self.assertEqual(json.loads(r36.PROTOCOL.read_text()), r36.PROTOCOL_VALUE)

    def test_extended_budget_is_isolated_and_reports_both_horizons(self):
        protocol = r36.load_extended_contract()
        self.assertEqual(protocol["episodes_per_task"], 10)
        self.assertEqual(protocol["development_task_ids"], list(range(15)))
        self.assertEqual(
            protocol["claim_level"],
            "consumed_development_extended_budget_diagnostic",
        )
        self.assertFalse(protocol["confirmation_panel_accessed"])


if __name__ == "__main__":
    unittest.main()
