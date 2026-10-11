from copy import deepcopy
from types import SimpleNamespace
import unittest

from example.helper.occupancy_pressure import (
    OCCUPANCY_PRESSURE_METRIC_CONTRACT,
    OccupancyPressureTracker,
    summarize_occupancy_pressure_runs,
)


def _block(
    position=None,
    *,
    stored=False,
    carrying=False,
    delivered=False,
):
    return SimpleNamespace(
        position=position,
        stored=stored,
        carrying=carrying,
        delivered=delivered,
    )


class OccupancyPressureTrackerTests(unittest.TestCase):
    def make_env(self):
        storage = [(1, col) for col in range(1, 6)]
        return SimpleNamespace(
            storage_positions=storage,
            waiting_cell=(0, 0),
            pickup_cell=(0, 1),
            blocks=[
                _block(storage[0], stored=True),
                _block(storage[1], stored=True, carrying=True),
                _block((0, 0)),
                _block((0, 1)),
                _block(storage[2], stored=True, delivered=True),
            ],
        )

    def test_tracks_inventory_physical_pressure_queue_and_candidates(self):
        env = self.make_env()
        tracker = OccupancyPressureTracker(env)

        # The reset state affects extrema, but not the transition denominator.
        initial = tracker.metrics()
        self.assertEqual(initial["peak_active_stored_count"], 2)
        self.assertEqual(initial["peak_physical_storage_occupancy_count"], 1)
        self.assertEqual(initial["minimum_free_storage_cells"], 4)
        self.assertEqual(initial["maximum_inbound_waiting_count"], 1)
        self.assertEqual(initial["maximum_inbound_queue_count"], 2)
        self.assertEqual(initial["storage_pressure_observed_steps"], 0)

        # Four of five cells are physically occupied after transition one.
        env.blocks[1].carrying = False
        env.blocks[2].position = env.storage_positions[2]
        env.blocks[2].stored = True
        env.blocks[3].position = env.storage_positions[3]
        env.blocks[3].stored = True
        tracker.observe_transition()

        # Transition two falls below the 80% threshold.
        env.blocks[3].delivered = True
        tracker.observe_transition()
        tracker.observe_decision(
            {"candidate_count": 7, "mode_counts": {"accept": 3}}
        )
        tracker.observe_candidate_set(2, accept_count=1)
        tracker.observe_candidate_set(1, accept_count=0)
        # A fixed baseline may expose only the assignment-cell candidate set,
        # not a hierarchy-wide set; those are deliberately kept distinct.
        tracker.observe_candidate_set(None, accept_count=2)

        result = tracker.metrics()
        self.assertEqual(
            result["occupancy_pressure_metric_contract"],
            OCCUPANCY_PRESSURE_METRIC_CONTRACT,
        )
        self.assertEqual(result["storage_capacity_cells"], 5)
        self.assertEqual(result["peak_active_stored_count"], 4)
        self.assertEqual(result["peak_active_stored_fraction"], 0.8)
        self.assertEqual(
            result["peak_physical_storage_occupancy_count"], 4
        )
        self.assertEqual(
            result["peak_physical_storage_occupancy_fraction"], 0.8
        )
        self.assertEqual(result["minimum_free_storage_cells"], 1)
        self.assertEqual(result["storage_pressure_observed_steps"], 2)
        self.assertEqual(
            result["storage_steps_at_or_above_80pct_occupied"], 1
        )
        self.assertEqual(
            result["storage_step_fraction_at_or_above_80pct_occupied"],
            0.5,
        )
        self.assertEqual(result["minimum_live_candidate_count"], 1)
        self.assertEqual(result["minimum_live_accept_candidate_count"], 1)

    def test_run_summary_pools_step_fraction_and_preserves_optional_counts(self):
        first = OccupancyPressureTracker(self.make_env()).metrics()
        first.update(
            {
                "storage_pressure_observed_steps": 2,
                "storage_steps_at_or_above_80pct_occupied": 1,
                "storage_step_fraction_at_or_above_80pct_occupied": 0.5,
                "minimum_live_candidate_count": 2,
                "minimum_live_accept_candidate_count": 4,
            }
        )
        second = deepcopy(first)
        second.update(
            {
                "storage_pressure_observed_steps": 8,
                "storage_steps_at_or_above_80pct_occupied": 1,
                "storage_step_fraction_at_or_above_80pct_occupied": 0.125,
                "minimum_free_storage_cells": 2,
                "maximum_inbound_queue_count": 5,
                "minimum_live_candidate_count": None,
                "minimum_live_accept_candidate_count": 3,
            }
        )

        summary = summarize_occupancy_pressure_runs([first, second])

        # Pooling uses pressure steps, not an unweighted mean of episode rates.
        self.assertEqual(summary["storage_pressure_observed_steps"], 10)
        self.assertEqual(
            summary["storage_steps_at_or_above_80pct_occupied"], 2
        )
        self.assertEqual(
            summary["storage_step_fraction_at_or_above_80pct_occupied"],
            0.2,
        )
        self.assertEqual(summary["minimum_free_storage_cells"], 2)
        self.assertEqual(summary["maximum_inbound_queue_count"], 5)
        self.assertEqual(summary["minimum_live_candidate_count"], 2)
        self.assertEqual(summary["minimum_live_accept_candidate_count"], 3)
        self.assertEqual(summary["mean_minimum_live_accept_candidate_count"], 3.5)

    def test_summary_rejects_partial_instrumentation(self):
        measured = OccupancyPressureTracker(self.make_env()).metrics()
        with self.assertRaisesRegex(ValueError, "with and without"):
            summarize_occupancy_pressure_runs([measured, {}])


if __name__ == "__main__":
    unittest.main()
