"""Static contract tests for the isolated PointMaze R12 factorial."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .factorial_planner_r12 import FactorialPointPlanner


HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "shared"


class PointMazeFactorialContractTest(unittest.TestCase):
    def setUp(self):
        self.point_manifest = json.loads(
            (HERE / "method_manifest_factorial_r12.json").read_text()
        )
        self.shared_manifest = json.loads(
            (SHARED / "method_manifest_r12.json").read_text()
        )
        self.protocol = json.loads(
            (HERE / "factorial_protocol_r12.json").read_text()
        )
        self.latent5_protocol = json.loads(
            (HERE / "latent5_protocol_r12.json").read_text()
        )
        self.latent8_protocol = json.loads(
            (HERE / "latent8_corner_protocol_r12.json").read_text()
        )

    def test_all_global_computational_settings_match_antmaze_screen(self):
        self.assertEqual(
            self.point_manifest["categories"]["global_computational"],
            self.shared_manifest["categories"]["global_computational"],
        )

    def test_shared_passage_contract_matches_antmaze_screen(self):
        names = (
            "passage_model_class",
            "passage_loss",
            "passage_training",
            "passage_ensemble_reduction",
        )
        point = self.point_manifest["categories"]["global_decision_principles"]
        shared = self.shared_manifest["categories"]["global_decision_principles"]
        self.assertEqual({name: point[name] for name in names}, {
            name: shared[name] for name in names
        })

    def test_screen_is_exactly_four_arms_two_goals_three_episodes(self):
        self.assertEqual(len(self.protocol["factorial_arms"]), 4)
        self.assertEqual(len(set(self.protocol["factorial_arms"])), 4)
        self.assertEqual(len(self.protocol["goal_names"]), 2)
        self.assertEqual(self.protocol["episodes_per_goal"], 3)

    def test_no_oracle_or_failed_conformance_item(self):
        self.assertEqual(self.protocol["oracle_inputs"], [])
        self.assertTrue(all(
            row["status"] == "pass"
            for row in self.protocol["conformance_audit"]
        ))

    def test_batched_candidates_preserve_frozen_r11_passage_calls(self):
        class FrozenR11Passages:
            relative = np.zeros((1, 2, 2), np.float32)

            def __init__(self):
                self.calls = []

            def live_passage_samples(self, start_xy, endpoint_xy):
                start = np.asarray(start_xy, np.float32)
                endpoint = np.asarray(endpoint_xy, np.float32).reshape(-1, 2)
                if start.shape != (2,):
                    raise ValueError("frozen R11 requires one live start")
                self.calls.append((start.copy(), endpoint.copy()))
                displacement = endpoint - start[None]
                return np.stack((
                    start[None] + 0.25 * displacement,
                    start[None] + 0.75 * displacement,
                ), axis=1)

        passages = FrozenR11Passages()
        planner = object.__new__(FactorialPointPlanner)
        planner.base = SimpleNamespace(passages=passages)
        start = np.asarray([[0, 0], [0, 0], [2, 1]], np.float32)
        endpoint = np.asarray([[4, 0], [0, 4], [4, 3]], np.float32)
        value = planner._matched_passage(start, endpoint)
        self.assertEqual(value.shape, (3, 4, 2))
        np.testing.assert_array_equal(value[:, 0], start)
        np.testing.assert_array_equal(value[:, -1], endpoint)
        self.assertEqual(len(passages.calls), 2)

    def test_latent5_is_an_isolated_truncated_smoke(self):
        self.assertEqual(
            self.latent5_protocol["factorial_arms"],
            ["latent_radius__nearest_recorded"],
        )
        self.assertEqual(self.latent5_protocol["evaluation_command_cap"], 5)
        self.assertEqual(
            self.latent5_protocol["benchmark_horizon_commands"], 20
        )
        self.assertEqual(
            self.latent5_protocol["claim_level"],
            "exploratory_truncated_horizon_smoke_only",
        )

    def test_latent8_is_an_isolated_full_horizon_corner_extension(self):
        self.assertEqual(
            self.latent8_protocol["factorial_arms"],
            ["latent_radius__nearest_recorded"],
        )
        self.assertEqual(self.latent8_protocol["episodes_per_goal"], 8)
        self.assertEqual(self.latent8_protocol["evaluation_command_cap"], 20)
        self.assertEqual(
            self.latent8_protocol["evaluation_command_cap"],
            self.latent8_protocol["benchmark_horizon_commands"],
        )
        self.assertTrue(all(
            row["status"] == "pass"
            for row in self.latent8_protocol["conformance_audit"]
        ))


if __name__ == "__main__":
    unittest.main()
