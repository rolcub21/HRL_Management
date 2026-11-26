import json
import unittest

from projects.tfsn_consequence_value_iteration.antmaze import (
    r34_posterior_consistent_acquisition as r34,
)


class R34PosteriorConsistencyTest(unittest.TestCase):
    def test_contract_is_narrow_and_goal_blind(self):
        protocol = r34.load_contract()
        self.assertEqual(protocol["base_method_revision"], r34.r33.METHOD_REVISION)
        self.assertEqual(protocol["gate_task_ids"], [3, 5])
        self.assertEqual(protocol["oracle_inputs"], [])
        self.assertFalse(protocol["confirmation_panel_accessed"])
        self.assertFalse(
            protocol["hidden_goal_visible_to_planner_executor_or_models"]
        )

    def test_survival_ratio_retains_untouched_hypothesis(self):
        self.assertEqual(r34.retention_probability(0.8, 0.8), 1.0)

    def test_survival_ratio_rejects_fully_searched_hypothesis(self):
        self.assertEqual(r34.retention_probability(0.8, 0.0), 0.0)

    def test_survival_ratio_is_fractional_without_a_threshold(self):
        self.assertEqual(r34.retention_probability(0.8, 0.2), 0.25)

    def test_roundoff_cannot_make_retention_exceed_one(self):
        self.assertEqual(r34.retention_probability(0.8, 0.8000001), 1.0)

    def test_protocol_file_matches_loaded_contract(self):
        self.assertEqual(json.loads(r34.PROTOCOL.read_text()), r34.PROTOCOL_VALUE)


if __name__ == "__main__":
    unittest.main()
