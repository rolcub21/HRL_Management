import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    RecordedTransitionGraph,
    build_recorded_graph,
    finite_horizon_values,
)
from projects.tfsn_consequence_value_iteration.antmaze.oracle_contract_diagnostic import (
    calibration_bins,
)
from projects.tfsn_consequence_value_iteration.antmaze.factorized_consequence import (
    FactorizedConsequenceEnsemble,
    _spatial_nll,
)
from projects.tfsn_consequence_value_iteration.antmaze.factorized_recorded_transition import (
    verify_graph_compatibility,
)
from projects.tfsn_consequence_value_iteration.antmaze.mobility_stitched_planning import (
    _diagonal_gaussian_circle_probability,
    sigma_landmarks,
    stitched_values,
)
from projects.tfsn_consequence_value_iteration.antmaze.shared_action_planning import (
    shared_action_values,
    shared_command_neighborhoods,
)
from projects.tfsn_consequence_value_iteration.antmaze.lazy_alias_advance import (
    LazyAliasAdvancePlanner,
    alias_kernel,
)
from projects.tfsn_consequence_value_iteration.antmaze.alias_partition_advance import (
    PartitionedAliasAdvancePlanner,
)
from projects.tfsn_consequence_value_iteration.antmaze.extended_budget_diagnostic import (
    COMMAND_BUDGET,
    ExtendedBudgetPlanner,
)
from projects.tfsn_consequence_value_iteration.antmaze.bellman_consistent_planning import (
    bellman_endpoint_backup,
)
from projects.tfsn_consequence_value_iteration.antmaze.integrated_bellman_planning import (
    mixture_expectation,
)
from projects.tfsn_consequence_value_iteration.antmaze.first_occupancy_bellman import (
    first_occupancy_backup,
    first_occupancy_stitched_values,
)
from projects.tfsn_consequence_value_iteration.antmaze.cached_first_occupancy import (
    projected_first_occupancy,
)
from projects.tfsn_consequence_value_iteration.antmaze.pessimistic_first_occupancy import (
    PessimisticFirstOccupancyPlanner,
)
from projects.tfsn_consequence_value_iteration.antmaze.measure_factorized_bandwidth import (
    posterior_expected_error,
)


def state(x, posture=0.0):
    value = np.zeros(29, np.float32)
    value[0] = x
    value[2] = posture
    return value


class TestRecordedTransitionPlanning(unittest.TestCase):
    def test_bandwidth_uses_posterior_responsibility_not_nearest_mode(self):
        weights = np.asarray([[0.5, 0.5]], np.float32)
        means = np.asarray([[[0.0, 0.0], [2.0, 0.0]]], np.float32)
        scales = np.asarray([[[0.1, 0.1], [0.1, 0.1]]], np.float32)
        error, responsibility, distance = posterior_expected_error(
            weights, means, scales, np.asarray([[0.05, 0.0]], np.float32)
        )
        self.assertGreater(float(responsibility[0, 0]), 0.999)
        self.assertAlmostEqual(float(distance[0, 0]), 0.05, places=6)
        self.assertAlmostEqual(float(error[0]), 0.05, places=6)

    def test_pessimistic_live_aggregation_uses_least_optimistic_member(self):
        planner = object.__new__(PessimisticFirstOccupancyPlanner)
        member_values = np.asarray(
            [[0.8, 0.2, 0.7], [0.3, 0.4, 0.5]], np.float32
        )
        np.testing.assert_allclose(
            planner._aggregate_member_values(member_values), [0.2, 0.3]
        )
        self.assertEqual(planner.ensemble_aggregation, "member_minimum")

    def test_projected_first_occupancy_weights_rare_contact_once(self):
        value = projected_first_occupancy(
            mixture_weight=np.asarray([[0.01, 0.99]], np.float32),
            contact=np.asarray([[1.0, 0.0]], np.float32),
            continuation=np.asarray([[0.0, 0.9]], np.float32),
            projection_support=np.ones((1, 2), np.float32),
        )
        np.testing.assert_allclose(value, [0.901], atol=1e-6)
        unsupported = projected_first_occupancy(
            mixture_weight=np.asarray([[0.01, 0.99]], np.float32),
            contact=np.asarray([[1.0, 0.0]], np.float32),
            continuation=np.asarray([[0.0, 0.9]], np.float32),
            projection_support=np.zeros((1, 2), np.float32),
        )
        np.testing.assert_allclose(unsupported, [0.01], atol=1e-6)

    def test_first_occupancy_is_bounded_and_contact_is_absorbing(self):
        contact = np.asarray([0.0, 0.25, 1.0], np.float32)
        continuation = np.asarray([0.7, 0.8, 0.4], np.float32)
        value = first_occupancy_backup(contact, continuation)
        np.testing.assert_allclose(value, [0.7, 0.85, 1.0])
        self.assertTrue(np.all(value >= 0))
        self.assertTrue(np.all(value <= 1))
        with self.assertRaises(ValueError):
            first_occupancy_backup(contact, continuation[:2])

    def test_first_occupancy_graph_does_not_accumulate_above_one(self):
        graph = RecordedTransitionGraph(
            nodes=np.stack((state(0), state(1))),
            source_index=[0],
            endpoint_index=[1],
            commands=[[1, 0]],
            edge_reliability=[1],
            state_scale=np.ones(29, np.float32),
        )
        values, _coverage, _field = first_occupancy_stitched_values(
            graph,
            targets=np.asarray([[1], [1]], np.int32),
            weight=np.ones((2, 1), np.float32),
            goal=np.asarray([1, 0], np.float32),
            horizon=10,
        )
        self.assertLessEqual(float(values.max()), 1.0)
        self.assertAlmostEqual(float(values[1, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(values[10, 0]), 1.0, places=6)

    def test_mixture_expectation_is_weighted_and_deterministic(self):
        weights = np.asarray([[0.25, 0.75], [0.6, 0.4]], np.float32)
        values = np.asarray([[4.0, 8.0], [10.0, 0.0]], np.float32)
        np.testing.assert_allclose(
            mixture_expectation(weights, values), [7.0, 6.0]
        )
        with self.assertRaises(ValueError):
            mixture_expectation(weights, values[:, :1])

    def test_factorized_spatial_modes_get_conditional_mean_posture(self):
        model = object.__new__(FactorizedConsequenceEnsemble)
        model.spatial_parameters = lambda inputs, member: (
            np.asarray([[0.25, 0.75]], np.float32),
            np.asarray([[[1.0, 2.0], [3.0, 4.0]]], np.float32),
            np.ones((1, 2, 2), np.float32),
        )
        posture_mean = np.zeros((1, 2, 2, 27), np.float32)
        posture_mean[:, :, 0] = 2.0
        posture_mean[:, :, 1] = 6.0
        model._posture_parameters = lambda inputs, endpoint, member: (
            np.asarray([[[0.25, 0.75], [0.5, 0.5]]], np.float32),
            posture_mean,
            np.ones_like(posture_mean),
        )
        inputs = np.zeros((1, 34), np.float32)
        inputs[:, 29:31] = [10.0, 20.0]
        weights, endpoints = model.spatial_component_mean_endpoints(inputs, 0)
        np.testing.assert_allclose(weights, [[0.25, 0.75]])
        np.testing.assert_allclose(
            endpoints[0, :, :2], [[11.0, 22.0], [13.0, 24.0]]
        )
        np.testing.assert_allclose(endpoints[0, 0, 2:], 5.0)
        np.testing.assert_allclose(endpoints[0, 1, 2:], 4.0)

    def test_bellman_endpoint_backup_uses_its_own_option_value(self):
        immediate = np.asarray([0.1, 0.4], np.float32)
        own = np.asarray([0.7, 0.2], np.float32)
        np.testing.assert_allclose(
            bellman_endpoint_backup(immediate, own), [0.8, 0.6]
        )
        with self.assertRaises(ValueError):
            bellman_endpoint_backup(immediate, own[:1])

    def test_extended_budget_changes_planning_horizon_only(self):
        self.assertEqual(COMMAND_BUDGET, 20)
        self.assertEqual(ExtendedBudgetPlanner.horizon, 20)
        self.assertEqual(PartitionedAliasAdvancePlanner.horizon, 10)

    def test_partitioned_alias_conserves_candidate_weight(self):
        planner = object.__new__(PartitionedAliasAdvancePlanner)
        association = np.asarray([0.0, 0.25, 0.75, 1.0], np.float32)
        bridge, advance = planner._association_weights(association)
        np.testing.assert_allclose(bridge + advance, np.ones(4), atol=1e-7)
        np.testing.assert_allclose(bridge, 1.0 - association, atol=1e-7)
        np.testing.assert_allclose(advance, association, atol=1e-7)
        self.assertEqual(float(bridge[-1]), 0.0)

    def test_alias_association_uses_existing_sigma_without_a_cutoff(self):
        values = alias_kernel(np.asarray([0.0, 0.2504, 0.5008]), 0.2504)
        self.assertAlmostEqual(float(values[0]), 1.0, places=6)
        self.assertAlmostEqual(float(values[1]), float(np.exp(-0.5)), places=6)
        self.assertGreater(float(values[2]), 0.0)
        self.assertGreater(float(values[1]), float(values[2]))

    def test_lazy_alias_uses_only_the_value_greedy_outgoing_action(self):
        graph = RecordedTransitionGraph(
            nodes=np.stack((state(0), state(1), state(2), state(3))),
            source_index=[1, 1],
            endpoint_index=[2, 3],
            commands=[[2, 0], [3, 0]],
            edge_reliability=[1.0, 0.5],
            state_scale=np.ones(29, np.float32),
        )
        planner = object.__new__(LazyAliasAdvancePlanner)
        planner.graph = graph
        planner.targets = np.asarray([[1], [2], [2], [3]], np.int32)
        planner.weight = np.zeros((4, 1), np.float32)
        planner.node_field = np.asarray([0, 0, 0.8, 1.0], np.float32)
        planner.values = np.zeros((2, 4), np.float32)
        planner.outgoing_order = np.argsort(
            graph.source_index, kind="stable"
        ).astype(np.int32)
        counts = np.bincount(graph.source_index, minlength=graph.node_count)
        planner.outgoing_offsets = np.concatenate(
            ([0], np.cumsum(counts, dtype=np.int64))
        )
        command, target, reliability, score, kind = planner._greedy_outgoing(
            np.asarray([1], np.int32), remaining=1
        )
        np.testing.assert_allclose(command, [[2, 0]])
        np.testing.assert_array_equal(target, [2])
        np.testing.assert_allclose(reliability, [1.0])
        np.testing.assert_allclose(score, [0.8])
        self.assertEqual(kind.tolist(), ["alias_next_recorded_edge"])

    def test_shared_commands_use_the_existing_sigma_neighborhood(self):
        graph = RecordedTransitionGraph(
            nodes=np.stack((state(0), state(0.2), state(1.0))),
            source_index=[0, 1, 2],
            endpoint_index=[1, 2, 2],
            commands=[[0.2, 0], [1, 0], [1, 0]],
            edge_reliability=[1, 1, 1],
            state_scale=np.ones(29, np.float32),
        )
        offsets, edges = shared_command_neighborhoods(graph, radius=0.25)
        np.testing.assert_array_equal(edges[offsets[0]:offsets[1]], [0, 1])
        np.testing.assert_array_equal(edges[offsets[2]:offsets[3]], [2])

    def test_shared_action_is_available_without_a_same_place_bridge(self):
        graph = RecordedTransitionGraph(
            nodes=np.stack((state(0), state(0.1), state(2))),
            source_index=[1],
            endpoint_index=[2],
            commands=[[2, 0]],
            edge_reliability=[1],
            state_scale=np.ones(29, np.float32),
        )
        bridge_targets = np.asarray([[1], [1], [2]], np.int32)
        bridge_weight = np.zeros((3, 1), np.float32)
        offsets = np.asarray([0, 1, 2, 2], np.int64)
        shared_edges = np.asarray([0, 0], np.int32)
        shared_target = np.asarray([2, 2], np.int32)
        shared_weight = np.ones(2, np.float32)
        values, _coverage, _field = shared_action_values(
            graph,
            bridge_targets,
            bridge_weight,
            offsets,
            shared_edges,
            shared_target,
            shared_weight,
            [2, 0],
            horizon=1,
        )
        self.assertGreater(values[1, 0], 0.99)
        self.assertGreater(values[1, 1], 0.99)

    def test_circle_probability_matches_centered_isotropic_gaussian(self):
        standard_deviation = 0.4
        radius = 0.2
        probability = _diagonal_gaussian_circle_probability(
            np.zeros((1, 1, 2), np.float32),
            np.full((1, 1, 2), standard_deviation, np.float32),
            np.zeros((1, 2), np.float32),
            radius,
        )
        expected = 1.0 - np.exp(
            -(radius**2) / (2.0 * standard_deviation**2)
        )
        self.assertAlmostEqual(float(probability[0, 0]), expected, places=5)

    def test_sigma_landmarks_cover_without_merging_exact_nodes(self):
        xy = np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [0.3, 0.0], [0.6, 0.0]],
            np.float32,
        )
        selected = sigma_landmarks(xy, 0.25)
        landmarks = xy[selected]
        nearest = np.linalg.norm(
            xy[:, None] - landmarks[None], axis=2
        ).min(axis=1)
        separation = np.linalg.norm(
            landmarks[:, None] - landmarks[None], axis=2
        )
        separation += np.eye(len(landmarks), dtype=np.float32) * 100.0
        self.assertLessEqual(float(nearest.max()), 0.25)
        self.assertGreaterEqual(float(separation.min()), 0.25 - 1e-6)
        self.assertEqual(len(xy), 4)

    def test_stitched_bridges_propagate_goal_value_across_strands(self):
        graph = RecordedTransitionGraph(
            nodes=np.stack((state(0), state(1), state(2), state(3))),
            source_index=[0],
            endpoint_index=[1],
            commands=[[1, 0]],
            edge_reliability=[1],
            state_scale=np.ones(29, np.float32),
        )
        targets = np.asarray([[1], [2], [3], [3]], np.int32)
        weights = np.ones((4, 1), np.float32)
        values, coverage, _field = stitched_values(
            graph, targets, weights, [3, 0], horizon=3
        )
        self.assertGreater(values[1, 2], 0.99)
        self.assertGreater(values[2, 1], 0.99)
        self.assertGreater(values[3, 0], 0.99)
        self.assertGreater(
            coverage[2]["meaningful_fraction"],
            coverage[0]["meaningful_fraction"],
        )

    def test_spatial_nll_prefers_the_matching_endpoint(self):
        weights = np.asarray([[0.5, 0.5]], np.float32)
        means = np.asarray([[[0.0, 0.0], [4.0, 0.0]]], np.float32)
        stds = np.full_like(means, 0.1)
        matching = _spatial_nll(
            weights, means, stds, np.asarray([[4.0, 0.0]], np.float32)
        )
        remote = _spatial_nll(
            weights, means, stds, np.asarray([[2.0, 2.0]], np.float32)
        )
        self.assertLess(float(matching[0]), float(remote[0]))

    def test_spatial_draw_uses_the_selected_residual(self):
        weights = np.asarray([[1.0]], np.float32)
        residual = np.asarray([[[0.25, -0.5]]], np.float32)
        stds = np.zeros_like(residual)
        draw = FactorizedConsequenceEnsemble._draw(
            weights, residual, stds, 3, np.random.default_rng(3)
        )
        np.testing.assert_allclose(
            draw,
            np.asarray([[[0.25, -0.5]] * 3], np.float32),
        )

    def test_stall_calibration_bins_include_zero_and_one_once(self):
        bins = calibration_bins(
            np.asarray([0.0, 0.05, 0.1, 0.95, 1.0]),
            np.asarray([0, 1, 0, 1, 1]),
        )
        self.assertEqual(sum(row["count"] for row in bins), 5)
        self.assertEqual(bins[0]["count"], 2)
        self.assertEqual(bins[1]["count"], 1)
        self.assertEqual(bins[9]["count"], 2)

    def test_graph_edges_are_exact_recorded_transitions(self):
        starts = np.stack((state(0), state(1), state(4, posture=2)))
        endpoints = np.stack((state(1), state(2), state(5, posture=3)))
        commands = np.asarray([[1, 0], [2, 0], [5, 0]], np.float32)
        graph = build_recorded_graph(
            starts,
            commands,
            endpoints,
            np.ones(29, np.float32),
        )
        np.testing.assert_array_equal(graph.nodes[graph.source_index], starts)
        np.testing.assert_array_equal(graph.nodes[graph.endpoint_index], endpoints)
        self.assertFalse(graph.metadata["transition_model_inside_graph"])
        self.assertFalse(
            graph.metadata["predicted_endpoint_projection_inside_graph"]
        )
        self.assertEqual(
            graph.metadata["exact_recorded_endpoint_reconstruction_max_error"],
            0.0,
        )

    def test_factorized_graph_compatibility_preserves_exact_graph(self):
        graph = RecordedTransitionGraph(
            nodes=np.stack((state(0), state(1))),
            source_index=np.zeros(30_000, np.int32),
            endpoint_index=np.ones(30_000, np.int32),
            commands=np.zeros((30_000, 2), np.float32),
            edge_reliability=np.ones(30_000, np.float32),
            state_scale=np.r_[
                np.asarray([0.2504, 0.2504], np.float32),
                np.ones(27, np.float32),
            ],
            metadata={
                "transition_model_inside_graph": False,
                "predicted_endpoint_projection_inside_graph": False,
            },
        )
        consequence = type("Consequence", (), {
            "input_std": np.ones(34, np.float32)
        })()
        verify_graph_compatibility(graph, consequence)

    def test_value_propagates_only_along_recorded_edges(self):
        nodes = np.stack((state(0), state(1), state(2), state(9)))
        graph = RecordedTransitionGraph(
            nodes=nodes,
            source_index=[0, 1],
            endpoint_index=[1, 2],
            commands=[[1, 0], [2, 0]],
            edge_reliability=[1, 1],
            state_scale=np.ones(29, np.float32),
        )
        values = finite_horizon_values(graph, [2, 0], sigma=0.25, horizon=2)
        self.assertGreater(values[1, 1], 0.99)
        self.assertGreater(values[2, 0], 0.99)
        self.assertEqual(float(values[2, 3]), 0.0)

    def test_unsupported_projection_has_smoothly_small_support(self):
        graph = RecordedTransitionGraph(
            nodes=np.stack((state(0), state(1))),
            source_index=[0],
            endpoint_index=[1],
            commands=[[1, 0]],
            edge_reliability=[1],
            state_scale=np.ones(29, np.float32),
            metadata={"revision": "test"},
        )
        _near_index, near_distance = graph.project([state(1.01)])
        _far_index, far_distance = graph.project([state(20)])
        near_support = np.exp(-0.5 * near_distance**2)
        far_support = np.exp(-0.5 * far_distance**2)
        self.assertGreater(float(near_support[0]), 0.99)
        self.assertLess(float(far_support[0]), 1e-20)

    def test_saved_graph_round_trips(self):
        graph = RecordedTransitionGraph(
            nodes=np.stack((state(0), state(1))),
            source_index=[0],
            endpoint_index=[1],
            commands=[[1, 0]],
            edge_reliability=[0.75],
            state_scale=np.ones(29, np.float32),
            metadata={"revision": "test"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.npz"
            graph.save(path)
            loaded = RecordedTransitionGraph.load(path)
        np.testing.assert_array_equal(loaded.nodes, graph.nodes)
        np.testing.assert_array_equal(loaded.endpoint_index, graph.endpoint_index)
        np.testing.assert_allclose(
            loaded.edge_reliability, graph.edge_reliability
        )
        self.assertEqual(loaded.metadata["revision"], "test")


if __name__ == "__main__":
    unittest.main()
