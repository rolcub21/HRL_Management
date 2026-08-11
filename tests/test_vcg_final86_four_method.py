from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import run_vcg_final86_four_method as final86


class Final86ProtocolTests(unittest.TestCase):
    def test_exact_row_grid_and_rng_namespaces(self):
        root = Path("/tmp/final86-grid-test")
        paths = final86._expected_ledger_paths(root)
        self.assertEqual(len(paths), 600)
        self.assertEqual(
            sum(final86.V11_METHOD in str(path) for path in paths), 90,
        )
        self.assertEqual(
            sum(final86.V23_METHOD in str(path) for path in paths), 360,
        )
        self.assertEqual(
            sum(final86.DYNAMIC_METHOD in str(path) for path in paths), 30,
        )
        self.assertEqual(
            sum(final86.GA_METHOD in str(path) for path in paths), 120,
        )
        v23 = {
            final86.V23_POLICY_RNG_BASE + 4 * index + replicate
            for index in range(30)
            for replicate in final86.RNG_INDICES
        }
        ga_bases = [
            final86.GA_RNG_BASE
            + final86.GA_RNG_STRIDE * (4 * index + replicate)
            for index in range(30)
            for replicate in final86.RNG_INDICES
        ]
        self.assertEqual(len(v23), 120)
        self.assertEqual(len(set(ga_bases)), 120)
        for left, right in zip(ga_bases, ga_bases[1:]):
            self.assertGreater(right - left, final86.MAX_STEPS)

    def test_timing_derivation_has_frozen_sign_and_boundary(self):
        deviations = (-40, -20, -10, 0, 10, 20, 30, 40)
        metrics = final86._timing_metrics(deviations)
        self.assertEqual(metrics["mean_signed_deviation"], 3.75)
        self.assertEqual(metrics["mean_absolute_error"], 21.25)
        self.assertEqual(metrics["mean_tardiness"], 12.5)
        self.assertEqual(metrics["mean_earliness"], 8.75)
        self.assertEqual(metrics["within_target_window_rate"], 0.625)

    def test_nested_v23_reduction_returns_one_point_per_instance(self):
        rows = []
        for instance_index, seed in enumerate(final86.INSTANCE_SEEDS):
            for model_seed in final86.V23_MODEL_SEEDS:
                for rng_index in final86.RNG_INDICES:
                    value = float(instance_index + model_seed + rng_index)
                    row = {
                        "instance_seed": seed,
                        "model_seed": model_seed,
                        "rng_index": rng_index,
                    }
                    row.update({metric: value for metric in final86.METRICS})
                    rows.append(row)
        points = final86._method_instance_points(final86.V23_METHOD, rows)
        self.assertEqual(set(points), set(final86.INSTANCE_SEEDS))
        expected = (sum(final86.V23_MODEL_SEEDS) / 3.0) + 1.5
        self.assertEqual(points[86_000]["mean_absolute_error"], expected)
        self.assertEqual(
            points[86_029]["mean_absolute_error"], 29.0 + expected,
        )

    def test_failed_row_is_bound_to_exact_input(self):
        identity = {
            "instance_seed": 86_000,
            "instance_index": 0,
            "episode_instance_id": "instance-id",
            "schedule_id": "schedule-id",
            "episode_instance_sha256": "a" * 64,
        }
        contract = {"contract_sha256": "b" * 64}
        manifest = {"manifest_sha256": "c" * 64}
        input_contract = final86._input_contract(
            contract=contract,
            manifest=manifest,
            method=final86.GA_METHOD,
            identity=identity,
            model_seed=None,
            rng_index=0,
            rng_seed=final86.GA_RNG_BASE,
            checkpoint_sha256=None,
        )
        row = final86._failed_row(
            method=final86.GA_METHOD,
            identity=identity,
            model_seed=None,
            rng_index=0,
            rng_seed=final86.GA_RNG_BASE,
            checkpoint_sha256=None,
            error=RuntimeError("synthetic"),
        )
        final86._validate_row_binding(row, input_contract)
        tampered = deepcopy(row)
        tampered["rng_seed"] += 1
        with self.assertRaises(final86.Final86Error):
            final86._validate_row_binding(tampered, input_contract)

    def test_v11_frontier_safety_is_derived_before_normalization(self):
        identity = {
            "instance_seed": 86_000,
            "instance_index": 0,
            "episode_instance_id": "instance-id",
            "schedule_id": "schedule-id",
            "episode_instance_sha256": "a" * 64,
        }
        raw = {
            "method_audit": {
                "complete_frontier_exactly_verified": True,
                "frontiers": [
                    {
                        "complete_frontier_exactly_verified": True,
                        "candidates": [{"certificate": {"status": "SAFE"}}],
                    }
                ],
            },
            "delivery_deviations": [0.0] * 8,
            "dense_objective_return": 1.0,
            "steps": 8,
            "delivery_count": 8,
            "required_deliveries": 8,
            "physical_storage_relocations": 0,
            "strict_method_success": 1.0,
            "completion_rate": 1.0,
            "illegal_drops": 0,
            "invalid_assignments": 0,
            "fallbacks": 0,
            "witness_mismatches": 0,
            "method_failure_reason": None,
        }
        exact, compact = final86._v11_exact_safety_audit(raw)
        self.assertTrue(exact)
        self.assertNotIn("frontiers", compact)
        normalized_input = dict(raw)
        normalized_input["all_selected_candidates_exact_safe"] = exact
        row = final86._common_row(
            method=final86.V11_METHOD,
            raw=normalized_input,
            identity=identity,
            model_seed=0,
            rng_index=None,
            rng_seed=None,
            checkpoint_sha256="b" * 64,
        )
        self.assertTrue(row["strict_safe_complete"])

        unsafe = deepcopy(raw)
        unsafe["method_audit"]["frontiers"][0]["candidates"][0][
            "certificate"
        ]["status"] = "UNSAFE"
        exact, _ = final86._v11_exact_safety_audit(unsafe)
        self.assertFalse(exact)
        unsafe["all_selected_candidates_exact_safe"] = exact
        row = final86._common_row(
            method=final86.V11_METHOD,
            raw=unsafe,
            identity=identity,
            model_seed=0,
            rng_index=None,
            rng_seed=None,
            checkpoint_sha256="b" * 64,
        )
        self.assertFalse(row["strict_safe_complete"])
        self.assertIn("learned_candidate_not_exact_safe", row["safety_issues"])

    def test_ledger_tree_rejects_unplanned_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run-ledger" / "unexpected").mkdir(parents=True)
            with self.assertRaises(final86.Final86Error):
                final86._validate_ledger_tree(root, require_complete=False)

    def test_instance_tree_rejects_unplanned_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "instances"
            directory.mkdir()
            (directory / "seed-86030.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(final86.Final86Error):
                final86._validate_instance_tree(root, require_complete=False)

    def test_source_registry_is_stable_and_transitive(self):
        first = final86._source_paths()
        second = final86._source_paths()
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 84)
        for required in (
            "project:train_vcg_dense_proper.py",
            "project:viability_graph_constrained_v2_3.py",
            "project:PSLAP/ga_capacity_aware.py",
            "prospective:runner",
            "prospective:v23_final_adapter",
        ):
            self.assertIn(required, first)


if __name__ == "__main__":
    unittest.main()
