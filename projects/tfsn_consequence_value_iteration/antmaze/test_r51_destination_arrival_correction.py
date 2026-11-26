import numpy as np
import unittest

from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r51_destination_arrival_correction as r51,
)


class DestinationArrivalCorrectionTest(unittest.TestCase):
    def test_arrival_event_uses_deployed_sigma_without_an_extra_threshold(self):
        self.assertEqual(
            r51.destination_arrival([0.9, 0.0], [0.0, 0.0], 1.0), 1.0
        )
        self.assertEqual(
            r51.destination_arrival([1.1, 0.0], [0.0, 0.0], 1.0), 0.0
        )

    def test_repeated_dead_command_drives_connection_down(self):
        prior = np.asarray([[0.8, 0.6, 0.4]], np.float32)
        support = np.asarray([4.0], np.float32)
        misses = np.asarray([0.0], np.float32)
        posterior = r48.bayesian_local_connection(prior, support, misses)
        np.testing.assert_allclose(posterior, prior / 5.0)

    def test_ingested_outcome_is_destination_arrival_not_soft_agreement(self):
        planner = object.__new__(r51.DestinationArrivalOnlinePlanner)
        planner.online_starts = []
        planner.online_commands = []
        planner.online_destinations = []
        planner.online_outcomes = []
        planner.pending_start = np.zeros(29, np.float32)
        planner.pending_command = np.asarray([0.0, 0.0], np.float32)
        planner.pending_destination = np.asarray([0.0, 0.0], np.float32)
        planner.pending_online_length = 0
        endpoint = np.zeros(29, np.float32)
        endpoint[:2] = [1.1 * r48.SIGMA, 0.0]
        outcome = planner._ingest_pending([endpoint])
        self.assertEqual(outcome, 0.0)
        self.assertEqual(planner.online_outcomes, [0.0])
        self.assertEqual(len(planner.online_destinations), 1)

    def test_protocol_audit_keeps_pointmaze_frozen(self):
        report = r51.audit()
        self.assertTrue(report["passed"])
        self.assertFalse(report["pointmaze_shared_implementation"])
        self.assertFalse(report["pointmaze_r11_changed"])

    def test_positive_evidence_command_is_its_own_destination(self):
        planner = object.__new__(r51.DestinationArrivalOnlinePlanner)
        planner.pending_destination = None
        command = np.asarray([1.25, -2.5], np.float32)
        base_choose = r51.r48.RuleFreeOnlinePlanner.choose
        original = base_choose
        try:
            r51.r48.RuleFreeOnlinePlanner.choose = lambda *args, **kwargs: (
                command.copy(),
                {
                    "selected_kind": "online_positive_evidence",
                    "selected_connection_weight": 0.5,
                },
            )
            selected, _diagnostic = planner.choose(
                np.zeros(29, np.float32), 0, object()
            )
        finally:
            r51.r48.RuleFreeOnlinePlanner.choose = original
        np.testing.assert_array_equal(selected, command)
        np.testing.assert_array_equal(planner.pending_destination, command)


if __name__ == "__main__":
    unittest.main()
