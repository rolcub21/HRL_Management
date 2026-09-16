from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import compare_vcg_v2_3_matched_baselines as comparison


def _identities():
    return {
        seed: {
            "episode_instance_id": f"instance-{seed}",
            "schedule_id": f"schedule-{seed}",
            "episode_instance_sha256": f"{seed:064x}",
        }
        for seed in comparison.PANEL_SEEDS
    }


def _row(method, seed, *, value=0.0, physical=0, model_seed=None, rng_index=None):
    identity = comparison._identity_fields(seed, _identities())
    deviations = (float(value),) * comparison.EXPECTED_DELIVERIES
    timing = comparison._timing_metrics(deviations)
    is_v23 = method == comparison.V23_METHOD
    is_v11 = method == comparison.V11_METHOD
    baseline = method in comparison.DETERMINISTIC_METHODS
    if is_v23:
        components = {field: None for field in comparison.DECOMPOSITION_COMPONENTS}
        status = (
            "decomposition_not_authenticated_in_source;selected_reconfigure_"
            "action_count_equals_total_physical"
        )
    else:
        components = {
            "target_bound_obstruction_clearances": physical,
            "standalone_reconfigurations": 0,
            "standalone_with_direct_delivery_available": 0,
            "standalone_without_direct_delivery_available": 0,
            "directly_deliverable_self_reconfigurations": 0,
        }
        status = "authenticated_canonical_event_time_schema_v1"
    row = {
        "protocol": comparison.PROTOCOL,
        "method_id": method,
        "method_category": "deterministic_online_primary" if baseline else "learned_online_primary",
        "information_regime": (
            "online_arrived_only_exact_closed_admission_certificate"
            if is_v23
            else "online_arrived_only"
        ),
        "replication_design": "test",
        "model_seed": model_seed,
        "policy_rng_index": rng_index,
        "policy_rng_seed": (
            comparison.validation_policy_rng_seed(
                comparison.PANEL_SEEDS.index(seed), rng_index
            )
            if is_v23
            else None
        ),
        **identity,
        "dense_objective_return": 100.0 + float(value),
        "delivery_deviations": deviations,
        **timing,
        "steps": 100 + int(value),
        "required_deliveries": comparison.EXPECTED_DELIVERIES,
        "delivery_count": comparison.EXPECTED_DELIVERIES,
        "physical_storage_relocations": physical,
        "physical_rehandles_per_100_required_deliveries": 12.5 * physical,
        **components,
        "rehandle_decomposition_status": status,
        "strict_method_success": True,
        "completion_rate": 1.0,
        "all_selected_candidates_exact_safe": True if not baseline else None,
        "illegal_drops": 0,
        "invalid_assignments": 0,
        "fallbacks": 0,
        "witness_mismatches": 0,
        "method_failure_reason": None,
        "evaluation_learning": False,
        "evaluation_epsilon": 0.0,
        "evaluation_policy": (
            comparison.V23_POLICY_REALIZATION
            if is_v23
            else "deterministic_greedy_epsilon_zero"
            if is_v11
            else "deterministic_online_duration_aware_scheduler"
        ),
        "map_selection_used": False if is_v23 else None,
        "stochastic_selection_only": True if is_v23 else False if is_v11 else None,
        "fresh_evaluation_clone": True if is_v23 else None if is_v11 else True,
        "training_agent_unchanged": True if is_v23 else None if is_v11 else True,
        "validation_batch_state_unchanged": True if is_v23 else None if is_v11 else True,
        "source_origin": "test",
        "source_execution_reused": not baseline,
        "source_checkpoint_sha256": "a" * 64 if not baseline else None,
        "source_validation_checkpoint_sha256": "d" * 64 if not baseline else None,
        "source_ledger_sha256": "b" * 64 if not baseline else None,
        "source_manifest_sha256": "c" * 64,
    }
    if is_v23:
        row["selected_reconfigure_action_count"] = physical
    if baseline:
        dense_delta = comparison._dual_rescore_from_legacy_return(
            0.0, deviations, comparison.FROZEN_OBJECTIVE_SPEC
        )[1]
        row.update(
            {
                "assignment_source": comparison.DETERMINISTIC_METHOD_TO_SOURCE[method],
                "legacy_environment_return": row["dense_objective_return"] - dense_delta,
                "planning_seconds": 0.1,
                "episode_wall_seconds": 0.2,
                "method_audit": {},
            }
        )
    comparison._validate_normalized_row_schema(row, method)
    return row


class MatchedV23BaselineTests(unittest.TestCase):
    def test_v23_nested_rng_aggregation_uses_12_instance_clusters(self):
        rows = [
            _row(
                comparison.V23_METHOD,
                seed,
                value=float(instance_index),
                physical=int(rng_index == 0),
                model_seed=10,
                rng_index=rng_index,
            )
            for instance_index, seed in enumerate(comparison.PANEL_SEEDS)
            for rng_index in range(4)
        ]
        summary = comparison._summarize_method(
            comparison.V23_METHOD, rows, _identities()
        )
        self.assertTrue(summary["whole_method_numeric_eligible"])
        self.assertEqual(len(summary["instance_cluster_points"]), 12)
        self.assertTrue(
            all(point["replication_count"] == 4 for point in summary["instance_cluster_points"])
        )
        self.assertAlmostEqual(summary["metrics"]["mean_absolute_error"], 5.5)
        self.assertAlmostEqual(
            summary["metrics"]["physical_rehandles_per_100_required_deliveries"],
            3.125,
        )
        stat = summary["instance_cluster_statistics"][
            "physical_rehandles_per_100_required_deliveries"
        ]
        self.assertEqual(stat["n_episode_instance_clusters"], 12)
        self.assertEqual(
            stat["one_sided_t_quantile_df11"],
            comparison.DESCRIPTIVE_ONE_SIDED_T_95_DF11,
        )
        self.assertEqual(summary["delivery_position_metrics"]["first_two"]["n"], 96)

    def test_v11_equal_seed_aggregation_retains_safe_seed_summaries(self):
        rows = [
            _row(
                comparison.V11_METHOD,
                seed,
                value=float(model_seed * 10),
                physical=model_seed,
                model_seed=model_seed,
            )
            for model_seed in comparison.MODEL_SEEDS
            for seed in comparison.PANEL_SEEDS
        ]
        summary = comparison._summarize_method(
            comparison.V11_METHOD, rows, _identities()
        )
        self.assertTrue(summary["whole_method_numeric_eligible"])
        self.assertEqual(set(summary["model_seed_summaries"]), {"0", "1", "2"})
        self.assertTrue(
            all(
                item["whole_seed_safe_complete"]
                for item in summary["model_seed_summaries"].values()
            )
        )
        self.assertAlmostEqual(summary["metrics"]["mean_absolute_error"], 10.0)
        self.assertAlmostEqual(
            summary["training_seed_replication"]["descriptive_variability"]
            ["mean_absolute_error"]["range"],
            20.0,
        )

    def test_one_unsafe_row_suppresses_whole_method(self):
        rows = [
            _row(comparison.NEAREST_METHOD, seed, value=1.0)
            for seed in comparison.PANEL_SEEDS
        ]
        rows[7]["illegal_drops"] = 1
        summary = comparison._summarize_method(
            comparison.NEAREST_METHOD, rows, _identities()
        )
        self.assertFalse(summary["whole_method_numeric_eligible"])
        self.assertIsNone(summary["metrics"])
        self.assertEqual(summary["instance_cluster_points"], ())
        self.assertEqual(len(summary["row_safety_issues"]), 1)

    def test_baseline_dense_rescore_is_required(self):
        row = _row(comparison.DYNAMIC_METHOD, comparison.PANEL_SEEDS[0], value=2.0)
        self.assertFalse(comparison._row_safety_issues(row, _identities()))
        missing = deepcopy(row)
        missing["legacy_environment_return"] = None
        self.assertIn(
            "baseline_legacy_return_missing_for_dense_authentication",
            comparison._row_safety_issues(missing, _identities()),
        )
        wrong = deepcopy(row)
        wrong["dense_objective_return"] += 1.0
        self.assertIn(
            "baseline_dense_rescore_mismatch",
            comparison._row_safety_issues(wrong, _identities()),
        )

    def test_atomic_ledger_rejects_contract_tampering(self):
        seed = comparison.PANEL_SEEDS[0]
        method = comparison.NEAREST_METHOD
        contract = {
            "assignment_source": comparison.DETERMINISTIC_METHOD_TO_SOURCE[method],
            "episode_instance_id": _identities()[seed]["episode_instance_id"],
            "schedule_id": _identities()[seed]["schedule_id"],
            "episode_instance_sha256": _identities()[seed]["episode_instance_sha256"],
        }
        row = _row(method, seed, value=1.0)
        with tempfile.TemporaryDirectory() as directory:
            loaded = comparison._load_or_execute_baseline(
                output_dir=Path(directory), method=method, seed=seed,
                input_contract=contract, execute=True, executor=lambda: row,
            )
            self.assertEqual(loaded, row)
            with self.assertRaises(comparison.MatchedComparisonError):
                comparison._load_or_execute_baseline(
                    output_dir=Path(directory), method=method, seed=seed,
                    input_contract={**contract, "schedule_id": "changed"},
                    execute=False,
                    executor=lambda: self.fail("must not execute"),
                )
            path = comparison._baseline_ledger_path(Path(directory), method, seed)
            original = comparison._load_json(path)
            wrong_method = deepcopy(original)
            wrong_method.pop("ledger_sha256")
            wrong_method["run"]["method_id"] = comparison.DYNAMIC_METHOD
            wrong_method["ledger_sha256"] = comparison._digest_json(wrong_method)
            comparison._atomic_json(path, wrong_method)
            with self.assertRaises(comparison.MatchedComparisonError):
                comparison._load_or_execute_baseline(
                    output_dir=Path(directory), method=method, seed=seed,
                    input_contract=contract, execute=False,
                    executor=lambda: self.fail("must not execute"),
                )
            missing_legacy = deepcopy(original)
            missing_legacy.pop("ledger_sha256")
            missing_legacy["run"]["legacy_environment_return"] = None
            missing_legacy["ledger_sha256"] = comparison._digest_json(missing_legacy)
            comparison._atomic_json(path, missing_legacy)
            with self.assertRaises(comparison.MatchedComparisonError):
                comparison._load_or_execute_baseline(
                    output_dir=Path(directory), method=method, seed=seed,
                    input_contract=contract, execute=False,
                    executor=lambda: self.fail("must not execute"),
                )

    def test_capacity_screen_is_not_training_seed_evidence(self):
        def summary(method, mae, rehandles):
            return {
                "method_id": method,
                "whole_method_numeric_eligible": True,
                "metrics": {
                    "mean_absolute_error": mae,
                    "physical_rehandles_per_100_required_deliveries": rehandles,
                },
            }

        sources = SimpleNamespace(
            v23_selection={
                "selection_recomputed_without_training_summary": True,
                "selected_episode": 160,
                "selected_validation": {
                    "strict_integrity_gate": True,
                    "development_candidate_eligible": True,
                },
            }
        )
        passed = comparison._seed_stability_screen(
            [summary(comparison.V23_METHOD, 10.0, 5.0),
             summary(comparison.NEAREST_METHOD, 9.0, 6.0)],
            sources,
        )
        self.assertTrue(passed["passed"])
        self.assertFalse(passed["training_seed_stability_established"])
        failed = comparison._seed_stability_screen(
            [summary(comparison.V23_METHOD, 10.0, 5.0),
             summary(comparison.NEAREST_METHOD, 9.0, 5.0)],
            sources,
        )
        self.assertFalse(failed["passed"])

    def test_final_panel_seed_is_rejected(self):
        with self.assertRaises(comparison.MatchedComparisonError):
            comparison._baseline_input_contract(
                comparison.NEAREST_METHOD,
                86_000,
                sources=SimpleNamespace(identities=_identities()),
                comparison_contract={"contract_sha256": "a" * 64},
            )

    def test_real_opened_source_chain_authenticates_without_rollout(self):
        root = Path(__file__).resolve().parents[1]
        paths = (
            root / "results/vcg-constrained-v2-3-gamma1-ablation-seed10-200ep",
            root / "results/vcg-dense-v1-1-v2-2-panel-control-12instance",
            root / "results/vcg-constrained-v2-2-development-seed10-200ep",
        )
        if not all(path.is_dir() for path in paths):
            self.skipTest("completed development artifacts are not installed")
        sources = comparison.authenticate_sources(
            v23_source_dir=paths[0], v11_control_dir=paths[1],
            v22_source_dir=paths[2],
        )
        self.assertEqual(sources.v23_selection["selected_episode"], 160)
        self.assertEqual(sources.v23_selection["eligible_episodes"], (160,))
        self.assertEqual(len(sources.v23_rows), 48)
        self.assertEqual(len(sources.v11_rows), 36)


if __name__ == "__main__":
    unittest.main()
