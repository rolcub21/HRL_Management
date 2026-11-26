"""Focused conformance tests for the isolated R48 rules-only port."""

from __future__ import annotations

import json
import unittest

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    BRIDGE_CANDIDATES,
    COMMAND_STEPS,
    RECORDED_EVIDENCE_PRIOR_WEIGHT,
    RECORDED_RELIABILITY_NEIGHBORS,
)


class R48RuleRemovalTests(unittest.TestCase):
    def test_protocol_covers_all_nine_removals(self):
        rows = r48.PROTOCOL_VALUE["conformance_audit"]
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(row["status"] == "pass" for row in rows))
        self.assertEqual(
            r48.PROTOCOL_VALUE["planner_inputs"]["success_radius"],
            "forbidden",
        )
        self.assertEqual(r48.PROTOCOL_VALUE["development_task_ids"], list(range(15)))
        self.assertEqual(
            r48.PROTOCOL_VALUE["evaluation_sets"]["all15"]["claim_level"],
            "consumed_development_full_panel",
        )

    def test_manifest_contract_carries_exact_hashes(self):
        contract = r48.manifest_contract()
        self.assertEqual(
            contract["shared_method_manifest_sha256"],
            r48.sha256(r48.MANIFEST_PATH),
        )
        self.assertEqual(
            contract["method_reference_sha256"], r48.sha256(r48.REFERENCE)
        )

    def test_horizon_and_candidate_budget_come_from_manifest(self):
        evaluator = r48.PROTOCOL_VALUE["evaluator"]
        self.assertEqual(
            r48.HORIZON,
            int(evaluator["episode_primitive_steps"]) // COMMAND_STEPS,
        )
        self.assertEqual(r48.CANDIDATE_BUDGET, BRIDGE_CANDIDATES)

    def test_active_source_has_no_old_planner_inputs(self):
        source = r48.Path(r48.__file__).read_text()
        forbidden = (
            "CONTACT" + "_RADIUS",
            "DETECTION" + "_FOOTPRINT",
            "STALL" + "_RADIUS",
            "ENDPOINT" + "_SAMPLES",
        )
        self.assertFalse([token for token in forbidden if token in source])

    def test_reach_weight_is_the_declared_continuous_product(self):
        state = np.zeros((3, r48.STATE_DIM), np.float32)
        endpoint = state.copy()
        command = np.asarray([[0.0, 0.0], [r48.SIGMA, 0.0], [2 * r48.SIGMA, 0.0]], np.float32)
        endpoint[:, 0] = np.asarray([0.0, 2 * r48.SIGMA, r48.SIGMA], np.float32)
        graph = r48.ManifestRecordedTransitionGraph(
            np.concatenate((state, endpoint)),
            np.arange(3),
            np.arange(3, 6),
            command,
            np.ones(3, np.float32),
            np.ones(r48.STATE_DIM, np.float32),
        )
        actual = r48.movement_reach_weight(graph)
        requested = np.exp(
            -np.square(command).sum(axis=1) / (2.0 * r48.SIGMA**2)
        )
        achieved = np.exp(
            -np.square(endpoint[:, :2]).sum(axis=1) / (2.0 * r48.SIGMA**2)
        )
        np.testing.assert_allclose(
            actual, (1.0 - requested) * (1.0 - achieved), rtol=1e-6
        )
        self.assertEqual(float(actual[0]), 0.0)

    def test_online_prior_is_the_shared_prior(self):
        prior = np.asarray([[0.2, 0.4], [0.6, 0.8]], np.float32)
        support = np.asarray([0.0, 3.0], np.float32)
        mass = np.asarray([0.0, 1.5], np.float32)
        actual = r48.bayesian_local_connection(prior, support, mass)
        expected = (
            RECORDED_EVIDENCE_PRIOR_WEIGHT * prior + mass[:, None]
        ) / (RECORDED_EVIDENCE_PRIOR_WEIGHT + support[:, None])
        np.testing.assert_allclose(actual, expected)

    def test_graph_support_uses_shared_neighbor_and_prior_settings(self):
        self.assertEqual(
            RECORDED_RELIABILITY_NEIGHBORS,
            r48.METHOD_MANIFEST["categories"]["global_computational"][
                "recorded_reliability_neighbors"
            ],
        )
        self.assertEqual(
            RECORDED_EVIDENCE_PRIOR_WEIGHT,
            r48.METHOD_MANIFEST["categories"]["global_computational"][
                "recorded_evidence_prior_weight"
            ],
        )

    def test_exact_joint_component_integration_uses_all_modes(self):
        model = r48.AntConsequenceAdapter(r48.R30_CHECKPOINT, "cpu")
        graph = r48.r31.verify_graph(r48.r31.graph_path())
        states = graph.nodes[graph.source_index[:2]]
        commands = graph.commands[:2]
        weight, endpoint = model.component_mean_endpoint_outcomes(
            states, commands, member=0
        )
        spatial = int(model.payload["spatial_config"]["components"])
        posture = int(model.payload["posture_config"]["components"])
        self.assertEqual(weight.shape, (2, spatial * posture))
        self.assertEqual(endpoint.shape, (2, spatial * posture, r48.STATE_DIM))
        np.testing.assert_allclose(weight.sum(axis=1), 1.0, atol=1e-5)

    def test_protocol_is_machine_readable(self):
        value = json.loads(r48.PROTOCOL.read_text())
        self.assertEqual(value["method_revision"], r48.METHOD_REVISION)

    def test_passage_audit_weighted_quantiles_use_bridge_mass(self):
        values = np.asarray([1.0, 2.0, 10.0])
        weights = np.asarray([1.0, 8.0, 1.0])
        actual = r48._weighted_quantiles(values, weights, (0.5, 0.9, 1.0))
        self.assertEqual(actual, {"p50": 2.0, "p90": 2.0, "p100": 10.0})


if __name__ == "__main__":
    unittest.main()
