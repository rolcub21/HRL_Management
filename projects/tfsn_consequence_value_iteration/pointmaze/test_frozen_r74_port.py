"""Cross-domain identity tests for the frozen R74 PointMaze port."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r64_r52_offline_hub_ranking as ant_hubs,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r65_r52_area_weighted_hubs as ant_area,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r72_hub_route_correction as ant_route,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r74_reach_scale_route_correction as ant_reach,
)
from projects.tfsn_consequence_value_iteration.shared import frozen_hub_method

from .frozen_r74_planner import METHOD_REVISION, PROTOCOL, load_protocol
from .planner import PointMazeRoadmapPlanner


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results/pointmaze"


class FrozenR74PointPortTest(unittest.TestCase):
    def test_protocol_has_no_domain_rule_or_cutoff(self):
        value = load_protocol()
        self.assertEqual(value["method_revision"], METHOD_REVISION)
        self.assertIsNone(value["link_similarity_cutoff"])
        self.assertIsNone(value["route_similarity_cutoff"])
        self.assertIsNone(value["hub_evaluation_cutoff"])
        self.assertFalse(value["success_radius_read_by_planner"])
        self.assertFalse(value["wall_geometry_used_by_planner"])
        self.assertEqual(value["oracle_inputs"], [])

    def test_hub_partition_is_exactly_the_frozen_ant_formula(self):
        rng = np.random.default_rng(8)
        xy = rng.normal(size=(100, 2))
        expected = ant_hubs.alias_hub_candidates(xy, 0.2)
        actual = frozen_hub_method.alias_hub_candidates(xy, 0.2)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)

    def test_area_prior_is_exactly_the_frozen_ant_formula(self):
        rng = np.random.default_rng(9)
        xy = rng.normal(size=(100, 2))
        expected = ant_area.area_weights(xy, 0.2)
        actual = frozen_hub_method.area_weights(xy, 0.2)
        np.testing.assert_allclose(expected[0], actual[0], atol=0.0, rtol=0.0)
        self.assertEqual(expected[1], actual[1])
        np.testing.assert_array_equal(expected[2], actual[2])

    def test_route_kernel_and_posterior_are_frozen_formulas(self):
        source = np.asarray([[0.2, -0.1], [1.0, 2.0]])
        hub = np.asarray([[2.0, 1.0], [-1.0, 0.4]])
        observed_source = np.asarray([[0.0, 0.0]])
        observed_hub = np.asarray([[1.5, 0.5]])
        covariance = np.asarray([[1.2, 0.2], [0.2, 0.7]])
        inverse = np.linalg.inv(covariance)
        expected = ant_reach.reach_route_similarity(
            source, hub, observed_source, observed_hub, inverse
        )
        actual = frozen_hub_method.route_similarity(
            source, hub, observed_source, observed_hub, inverse
        )
        np.testing.assert_allclose(expected, actual, atol=0.0, rtol=0.0)
        prior = np.asarray([0.2, 0.8])
        support = np.asarray([0.5, 2.0])
        mass = np.asarray([0.1, 1.0])
        np.testing.assert_allclose(
            ant_route.correct_hub_route_probability(prior, support, mass),
            frozen_hub_method.posterior_probability(
                prior, support, mass, prior=1.0
            ),
        )

    def test_r11_development_baseline_is_exactly_22_of_24(self):
        paths = (
            RESULTS / "continuous_reference_development_r11/summary.json",
            RESULTS / (
                "continuous_reference_hidden_goal_r11_disjoint15_e120/"
                "summary.json"
            ),
        )
        acquired = 0
        tasks = 0
        for path in paths:
            summary = json.loads(path.read_text())
            tasks += len(summary["task_metrics"])
            acquired += sum(
                row["acquisition_at_horizon"] > 0
                for row in summary["task_metrics"]
            )
        self.assertEqual(tasks, 24)
        self.assertEqual(acquired, 22)

    def test_link_outcome_is_arrival_at_graph_destination(self):
        planner = object.__new__(PointMazeRoadmapPlanner)
        planner.sigma = 0.1
        planner.destination_arrival_correction = True
        planner.reset_task_memory()
        start = np.asarray([0.0, 0.0, 0.0, 0.0], np.float32)
        command = np.asarray([1.0, 0.0], np.float32)
        destination = np.asarray([0.5, 0.0], np.float32)
        planner.pending_arrival = (start.copy(), command.copy(), destination)
        planner.observe_transition(
            start, command, np.asarray([1.0, 0.0, 0.0, 0.0], np.float32)
        )
        self.assertEqual(planner.arrival_outcomes, [0.0])


if __name__ == "__main__":
    unittest.main()
