import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r70_corrected_hub_reachability as r70,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    RECORDED_EVIDENCE_PRIOR_WEIGHT,
)


class R70CorrectedHubReachabilityTest(unittest.TestCase):
    def test_protocol_is_focused_consumed_task_replay(self):
        self.assertEqual(r70.PROTOCOL_VALUE["task_ids"], [8])
        self.assertEqual(
            r70.PROTOCOL_VALUE["corrected_hub_selection"],
            "exact_global_argmax_by_coverage_upper_bound_branch_and_bound",
        )
        self.assertIsNone(r70.PROTOCOL_VALUE["hub_evaluation_cutoff"])
        self.assertEqual(
            r70.PROTOCOL_VALUE["claim_level"],
            "consumed_development_single_task_mechanism_replay",
        )

    def test_zero_support_preserves_bridge_prior(self):
        weight = np.asarray([[0.2, 0.8]], np.float32)
        corrected = r70.corrected_bridge_weights(
            weight, np.zeros_like(weight), np.zeros_like(weight)
        )
        np.testing.assert_allclose(corrected, weight, rtol=0.0, atol=1e-7)

    def test_failure_reduces_and_success_increases_weight(self):
        weight = np.asarray([[0.7]], np.float32)
        support = np.asarray([[1.0]], np.float32)
        failed = r70.corrected_bridge_weights(
            weight, support, np.zeros_like(support)
        )[0, 0]
        succeeded = r70.corrected_bridge_weights(
            weight, support, support
        )[0, 0]
        expected_failed = (
            RECORDED_EVIDENCE_PRIOR_WEIGHT * 0.7
            / (RECORDED_EVIDENCE_PRIOR_WEIGHT + 1.0)
        )
        self.assertAlmostEqual(float(failed), expected_failed, places=6)
        self.assertLess(float(failed), 0.7)
        self.assertGreater(float(succeeded), 0.7)

    def test_wall_crossing_geometry(self):
        # The central horizontal segment enters the wall centered at (2, 2).
        self.assertTrue(r70.segment_crosses_wall([-1.0, 2.0], [5.0, 2.0]))
        # This short segment stays within the open top-left cell.
        self.assertFalse(r70.segment_crosses_wall([-11.0, 10.0], [-9.0, 10.0]))

    def test_online_mass_is_exact_at_matching_link(self):
        support = np.zeros((1, 1), np.float32)
        outcome = np.zeros((1, 1), np.float32)
        nodes = np.zeros((1, 29), np.float32)
        target = np.asarray([[[1.0, 2.0]]], np.float32).reshape(1, 1, 2)
        r70._add_observation_mass(
            support,
            outcome,
            nodes,
            np.ones(29, np.float32),
            target,
            nodes[0],
            np.asarray([1.0, 2.0], np.float32),
            np.asarray([1.0, 2.0], np.float32),
            0.0,
            0.3,
        )
        self.assertAlmostEqual(float(support[0, 0]), 1.0, places=6)
        self.assertEqual(float(outcome[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
