import unittest
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.pointmaze.core import protocol
from projects.tfsn_consequence_value_iteration.pointmaze.dataset import (
    grouped_split,
    load_initial_dataset,
)
from projects.tfsn_consequence_value_iteration.pointmaze.panel import (
    load_development_panel,
    load_frozen_hidden_goal_panel,
)
from projects.tfsn_consequence_value_iteration.pointmaze.planner import (
    ConsumedSearchField,
    PointMazeRoadmapPlanner,
)
from projects.tfsn_consequence_value_iteration.pointmaze.fields import (
    SuppliedGoalField,
    TaskFields,
)
from projects.tfsn_consequence_value_iteration.pointmaze.roadmap import (
    RecordedRoadmap,
    bounded_search_values,
    graph_from_dataset,
    reach_targets,
)
from projects.tfsn_consequence_value_iteration.shared.density import (
    situation_density_weights,
)
from projects.tfsn_consequence_value_iteration.shared.manifest_audit import (
    audit_pointmaze,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    COMMAND_STEPS,
    SITUATION_DENSITY_TRUNCATION_SIGMAS,
    finite_horizon,
)


class PointMazeAdaptationTest(unittest.TestCase):
    def test_situation_density_counts_a_repeated_input_once(self):
        repeated = 100
        states = np.zeros((repeated + 1, 4), np.float32)
        commands = np.ones((repeated + 1, 2), np.float32)
        states[-1, :2] = 10.0
        commands[-1] = 11.0
        index = np.arange(repeated + 1)
        weight, _report = situation_density_weights(
            states, commands, index, position_scale=0.1,
            truncation_sigmas=SITUATION_DENSITY_TRUNCATION_SIGMAS,
        )
        self.assertAlmostEqual(
            float(weight[:-1].sum()), float(weight[-1]), places=4
        )

    def test_continuous_reach_measure_gives_no_ops_zero_mass(self):
        count = 8
        source = np.zeros((20, 4), np.float32)
        endpoint = np.zeros((20, 4), np.float32)
        command = np.zeros((20, 2), np.float32)
        endpoint[10:, 0] = np.linspace(1.0, 2.0, 10)
        command[10:, 0] = endpoint[10:, 0]
        nodes = np.concatenate((source, endpoint), axis=0)
        graph = RecordedRoadmap(
            nodes=nodes,
            source_index=np.arange(20),
            endpoint_index=20 + np.arange(20),
            commands=command,
            primitive_xy=np.repeat(endpoint[:, None, :2], 2, axis=1),
            state_scale=np.ones(4, np.float32),
        )
        _landmarks, _targets, _distance, offsets = reach_targets(
            graph, count, sigma=0.1, continuous_weighting=True
        )
        self.assertTrue(np.all(np.linalg.norm(offsets, axis=1) > 0.0))

    def test_advance_weighting_prefers_actual_progress(self):
        nodes = np.asarray([
            [0.0, 0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], np.float32)
        graph = RecordedRoadmap(
            nodes=nodes,
            source_index=np.zeros(0, np.int32),
            endpoint_index=np.zeros(0, np.int32),
            commands=np.zeros((0, 2), np.float32),
            primitive_xy=np.zeros((0, 2, 2), np.float32),
            state_scale=np.ones(4, np.float32),
        )
        planner = object.__new__(PointMazeRoadmapPlanner)
        planner.graph = graph
        planner.targets = np.asarray([[1, 2], [1, 2], [1, 2]], np.int32)
        planner.weight = np.ones((3, 2), np.float32)
        planner.values = np.zeros((2, 3), np.float32)
        planner.node_field = np.asarray([0.0, 0.9, 0.8], np.float32)
        planner.acquisition_active = False
        planner.advance_weighting = True
        planner.sigma = 0.1
        planner.outgoing_order = np.zeros(0, np.int32)
        planner.outgoing_offsets = np.zeros(4, np.int64)
        command, target, _score, factor, reliability, _kind = planner._greedy_outgoing(
            np.asarray([0], np.int32), remaining=1
        )
        np.testing.assert_allclose(command, [[1.0, 0.0]])
        np.testing.assert_array_equal(target, [2])
        self.assertGreater(float(factor[0]), 0.99)
        self.assertEqual(float(reliability[0]), 1.0)

    def test_horizon_and_executor_contract(self):
        value = protocol()
        self.assertEqual(value["environment"]["state"], ["x", "y", "vx", "vy"])
        self.assertNotIn("global_settings", value)
        self.assertEqual(COMMAND_STEPS, 100)
        self.assertEqual(
            finite_horizon(value["evaluator"]["episode_primitive_steps"]), 20
        )

    def test_smoke_dataset_split_holds_out_whole_groups(self):
        dataset = load_initial_dataset(max_groups_per_source=2)
        train, validation = grouped_split(dataset)
        train_groups = set(dataset.trajectory_id[train].tolist())
        validation_groups = set(dataset.trajectory_id[validation].tolist())
        self.assertFalse(train_groups & validation_groups)
        self.assertEqual(dataset.starts.shape[1], 4)
        self.assertEqual(dataset.commands.shape[1], 2)
        self.assertEqual(dataset.primitives.shape[1:], (100, 4))
        self.assertTrue(np.isfinite(dataset.endpoints).all())

    def test_recorded_graph_preserves_every_smoke_command(self):
        dataset = load_initial_dataset(max_groups_per_source=2)
        graph = graph_from_dataset(dataset, 0.2)
        self.assertEqual(graph.edge_count, dataset.rows)
        self.assertEqual(
            graph.node_count,
            dataset.rows + len(np.unique(dataset.trajectory_id)),
        )
        np.testing.assert_allclose(
            graph.nodes[graph.source_index], dataset.starts, atol=1e-5
        )
        np.testing.assert_allclose(
            graph.nodes[graph.endpoint_index], dataset.endpoints, atol=1e-5
        )

    def test_search_field_and_backup_are_bounded(self):
        fields = TaskFields(0.2)
        fields.extend([[0.0, 0.0]], [0.0])
        self.assertAlmostEqual(float(fields.search([[0.0, 0.0]])[0]), 0.0)
        self.assertGreater(float(fields.search([[1.0, 0.0]])[0]), 0.99)

        dataset = load_initial_dataset(max_groups_per_source=1)
        graph = graph_from_dataset(dataset, 0.2)
        targets = np.repeat(
            graph.endpoint_index[:1][None], graph.node_count, axis=0
        )
        weight = np.ones_like(targets, np.float32)
        recorded_gain = np.ones(graph.edge_count, np.float32) * 0.25
        bridge_gain = np.ones_like(targets, np.float32) * 0.25
        values = bounded_search_values(
            graph, targets, weight, recorded_gain, bridge_gain, horizon=3
        )
        self.assertGreaterEqual(float(values.min()), 0.0)
        self.assertLessEqual(float(values.max()), 1.0)

    def test_search_uses_only_the_data_derived_landing_scale(self):
        fields = TaskFields(0.1)
        fields.extend([[0.0, 0.0]], [1.0])
        expected = 1.0 - np.exp(-0.39**2 / (2.0 * 0.1**2))
        self.assertAlmostEqual(float(fields.search([[0.39, 0.0]])[0]), expected)

    def test_consumed_search_field_uses_landing_scale(self):
        fields = TaskFields(0.1)
        fields.extend([[10.0, 10.0]], [0.0])
        consumed = ConsumedSearchField(fields, [[0.0, 0.0]])
        np.testing.assert_allclose(
            consumed.search([[0.0, 0.0], [1.0, 0.0]]),
            np.asarray([0.0, 1.0], np.float32), atol=1e-6,
        )

    def test_supplied_goal_field_is_a_gaussian_bump(self):
        field = SuppliedGoalField(0.1, [1.0, 2.0])
        value = field.reward_rate([[1.0, 2.0], [1.1, 2.0]])
        self.assertAlmostEqual(float(value[0]), 1.0)
        self.assertAlmostEqual(float(value[1]), float(np.exp(-0.5)), places=6)
        self.assertTrue(field.contacted)

    def test_online_outcome_links_endpoint_to_next_movement(self):
        planner = object.__new__(PointMazeRoadmapPlanner)
        planner.sigma = 0.1
        planner.reset_task_memory()
        planner.begin_episode()
        planner.observe_transition(
            [0.0, 0.0, 0.0, 0.0], [1.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        )
        planner.observe_transition(
            [1.0, 0.0, 0.0, 0.0], [2.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        )
        self.assertGreater(planner.online_movement[0], 0.99)
        self.assertAlmostEqual(planner.online_endpoint_mobility[0], 0.0)

    def test_reference_modules_pass_manifest_audit(self):
        report = audit_pointmaze()
        self.assertTrue(report["passed"])
        self.assertFalse(
            report["detection_radius_decision"][
                "planner_reads_benchmark_detection_radius"
            ]
        )
        self.assertEqual(
            set(report["known_failures"]),
            {
                "spatial_diagnostic_native_goal_r01_c06",
                "spatial_diagnostic_native_goal_r06_c06",
            },
        )

    def test_freeze_records_manifest_and_stable_corner_failures(self):
        path = Path(__file__).with_name("R11_FREEZE.json")
        freeze = json.loads(path.read_text())
        self.assertEqual(freeze["freeze_status"], "frozen_reference")
        self.assertEqual(
            freeze["detection_radius_decision"]["acquisition_search_scale"],
            "data_derived_landing_scale",
        )
        self.assertEqual(
            {row["revision"] for row in freeze["development_evidence"]},
            {"pointmaze_density_movement_reach_r8", "tfsn_continuous_reference_r11"},
        )

    def test_development_panel_is_exposed_nine_by_eight(self):
        panel = load_development_panel()
        self.assertEqual(len(panel), 72)
        self.assertEqual(panel.goal_name.nunique(), 9)
        self.assertTrue(panel.groupby("goal_name").size().eq(8).all())

    def test_disjoint_hidden_goal_panel_is_fifteen_by_eight(self):
        path = Path(__file__).with_name("hidden_goal_protocol_panel15.json")
        contract = json.loads(path.read_text())
        panel = load_frozen_hidden_goal_panel(path)
        self.assertEqual(len(panel), 120)
        self.assertEqual(panel.goal_name.nunique(), 15)
        self.assertTrue(panel.groupby("goal_name").size().eq(8).all())
        self.assertFalse(
            set(panel.goal_name) & set(
                contract["development_subset"]["first_panel_goal_names"]
            )
        )
        self.assertEqual(contract["oracle_inputs"], [])

    def test_fresh_confirmation_panel_is_twenty_six_by_eight(self):
        path = Path(__file__).with_name(
            "hidden_goal_protocol_fresh_confirmation_seed0.json"
        )
        contract = json.loads(path.read_text())
        panel = load_frozen_hidden_goal_panel(path)
        self.assertEqual(len(panel), 208)
        self.assertEqual(panel.goal_name.nunique(), 26)
        self.assertTrue(panel.groupby("goal_name").size().eq(8).all())
        self.assertTrue(contract["confirmation_panel_accessed"])
        self.assertEqual(contract["oracle_inputs"], [])


if __name__ == "__main__":
    unittest.main()
