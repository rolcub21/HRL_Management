import math
import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r67_e549_hub_routed_exploration as r67,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r72_hub_route_correction as r72,
)


class HubRouteCorrectionTest(unittest.TestCase):
    def test_protocol_has_no_route_similarity_cutoff(self):
        self.assertIsNone(r72.PROTOCOL_VALUE["hub_route_similarity_cutoff"])
        self.assertIsNone(r72.PROTOCOL_VALUE["hub_evaluation_cutoff"])
        self.assertEqual(r72.PROTOCOL_VALUE["hub_route_prior_weight"], 1.0)
        self.assertEqual(r72.PROTOCOL_VALUE["task_ids"], list(range(9)))
        self.assertEqual(
            r72.PROTOCOL_VALUE["reachability_computational_backend"],
            "exact_batched_rolling_torch",
        )
        self.assertEqual(
            r72.PROTOCOL_VALUE["coverage_computational_backend"],
            "exact_chunked_torch",
        )

    def test_exact_route_option_has_unit_similarity(self):
        value = r72.hub_route_similarity(
            np.asarray([[1.0, 2.0]]),
            np.asarray([[3.0, 4.0]]),
            np.asarray([1.0, 2.0]),
            np.asarray([3.0, 4.0]),
            0.3,
        )
        self.assertAlmostEqual(float(value[0]), 1.0, places=7)

    def test_one_sigma_source_displacement_has_gaussian_weight(self):
        value = r72.hub_route_similarity(
            np.asarray([[1.0, 0.0]]),
            np.asarray([[0.0, 0.0]]),
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            1.0,
        )
        self.assertAlmostEqual(float(value[0]), math.exp(-0.5), places=7)

    def test_one_failed_attempt_halves_a_near_certain_prior(self):
        value = r72.correct_hub_route_probability(
            np.asarray([0.99]), np.asarray([1.0]), np.asarray([0.0])
        )
        self.assertAlmostEqual(float(value[0]), 0.495, places=7)

    def test_two_failed_attempts_reduce_prior_to_one_third(self):
        value = r72.correct_hub_route_probability(
            np.asarray([0.99]), np.asarray([2.0]), np.asarray([0.0])
        )
        self.assertAlmostEqual(float(value[0]), 0.33, places=7)

    def test_observer_records_failure_only_when_attempt_finishes(self):
        observer = r72.HubRouteAttemptObserver()
        observer.reset_task()
        observer.start_attempt(
            source_xy=[0.0, 0.0], hub_bank_index=3,
            hub_xy=[2.0, 0.0], remaining_commands=20,
            predicted_reachability=0.99,
        )
        record = observer.finish_attempt(False, [0.5, 0.0])
        self.assertFalse(record["arrived"])
        self.assertEqual(observer.summary()["hub_route_attempts"], 1)
        self.assertEqual(observer.summary()["hub_route_failures"], 1)

    def test_censored_attempt_is_not_training_evidence(self):
        observer = r72.HubRouteAttemptObserver()
        observer.reset_task()
        observer.start_attempt(
            source_xy=[0.0, 0.0], hub_bank_index=3,
            hub_xy=[2.0, 0.0], remaining_commands=20,
            predicted_reachability=0.99,
        )
        observer.cancel_attempt("task_contact_censored")
        self.assertEqual(observer.summary()["hub_route_attempts"], 0)
        self.assertEqual(observer.summary()["hub_route_censored_attempts"], 1)

    def test_batched_rolling_reachability_equals_scalar_numpy(self):
        rng = np.random.default_rng(92)
        nodes, bridges, recorded, horizon = 17, 4, 29, 6
        source = rng.integers(0, nodes, recorded, dtype=np.int32)
        endpoint = rng.integers(0, nodes, recorded, dtype=np.int32)
        edge_weight = rng.random(recorded, dtype=np.float32)
        bridge_target = rng.integers(
            0, nodes, (nodes, bridges), dtype=np.int32
        )
        bridge_weight = rng.random(
            (nodes, bridges), dtype=np.float32
        )
        targets = np.asarray([1, 7, 13, 16], np.int32)
        current = 8
        solver = r72.ExactBatchedReachability(
            source,
            endpoint,
            edge_weight,
            bridge_target,
            "cpu",
            target_batch=8,
        )
        solver.set_bridge_weight(bridge_weight)
        batched = solver.solve(targets, horizon, current)
        scalar = np.asarray([
            r67._one_target_layers(
                int(target),
                horizon,
                nodes,
                source,
                endpoint,
                edge_weight,
                bridge_target,
                bridge_weight,
            )[horizon, current]
            for target in targets
        ])
        np.testing.assert_array_equal(batched, scalar)

    def test_batched_branch_and_bound_retains_scalar_order_and_argmax(self):
        rng = np.random.default_rng(13)
        nodes, bridges, recorded, horizon = 21, 3, 38, 5
        source = rng.integers(0, nodes, recorded, dtype=np.int32)
        endpoint = rng.integers(0, nodes, recorded, dtype=np.int32)
        edge_weight = rng.random(recorded, dtype=np.float32)
        bridge_target = rng.integers(
            0, nodes, (nodes, bridges), dtype=np.int32
        )
        bridge_weight = rng.random(
            (nodes, bridges), dtype=np.float32
        )
        hub_nodes = np.asarray([2, 4, 7, 11, 16, 20], np.int32)
        coverage = np.asarray([0.63, 0.41, 0.91, 0.34, 0.72, 0.58])
        zeros = np.zeros(len(hub_nodes), np.float64)
        scalar = r72._scalar_branch_and_bound_selection(
            coverage,
            hub_nodes,
            horizon,
            9,
            nodes,
            source,
            endpoint,
            edge_weight,
            bridge_target,
            bridge_weight,
            zeros,
            zeros,
        )
        solver = r72.ExactBatchedReachability(
            source,
            endpoint,
            edge_weight,
            bridge_target,
            "cpu",
            target_batch=3,
        )
        batched = r72._batched_branch_and_bound_selection(
            coverage,
            hub_nodes,
            horizon,
            9,
            solver,
            bridge_weight,
            zeros,
            zeros,
        )
        self.assertEqual(scalar["selected"], batched["selected"])
        np.testing.assert_array_equal(
            scalar["evaluated_indices"], batched["evaluated_indices"]
        )
        np.testing.assert_array_equal(scalar["reaches"], batched["reaches"])


if __name__ == "__main__":
    unittest.main()
