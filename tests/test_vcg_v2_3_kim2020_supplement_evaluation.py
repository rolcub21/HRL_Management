from __future__ import annotations

import copy
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path("/home/ai_diagnosis/HRL_Management")
STAGED_ROOT = Path(__file__).parents[1]
MODULE_PATH = STAGED_ROOT / "vcg_v2_3_kim2020_supplement_evaluation.py"
sys.path.insert(0, str(PROJECT_ROOT))
SPEC = importlib.util.spec_from_file_location("kim_supplement_evaluation", MODULE_PATH)
evaluation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluation)


def fake_instance_record(panel_index: int = 0) -> dict:
    seed = 85000 + panel_index
    return {
        "panel_index": panel_index,
        "instance_seed": seed,
        "episode_instance_id": f"instance-{seed}",
        "schedule_id": f"schedule-{seed}",
        "episode_instance_sha256": f"canonical-{seed}",
        "saved_file_raw_sha256": f"raw-{seed}",
        "saved_path": f"/fixed/seed-{seed}.json",
    }


def fake_manifest() -> dict:
    return {
        "manifest_sha256": "m" * 64,
        "episode_instance_panel": {
            "records": [fake_instance_record(index) for index in range(12)]
        },
    }


def fake_training_completion(model_seed: int = 0) -> dict:
    return {
        "completion_sha256": f"completion-{model_seed}",
        "selected_checkpoint_raw_sha256": f"raw-checkpoint-{model_seed}",
        "selected_checkpoint_deployment_sha256": f"deployment-{model_seed}",
        "selected_episode": 100 * (model_seed + 1),
        "selected_checkpoint_inference_origin_counters": {
            "assignment_count": 80 + model_seed,
            "gradient_steps": 10 + model_seed,
            "completed_outcome_count": 70 + model_seed,
            "censored_outcome_count": 2 + model_seed,
        },
    }


def fake_raw(*, deviations=None, physical: int = 1) -> dict:
    if deviations is None:
        deviations = [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]
    timing = evaluation._timing_metrics(deviations)
    proposal_ids = [f"proposal-{index}" for index in range(8)]
    completion = fake_training_completion(0)
    source_counters = completion["selected_checkpoint_inference_origin_counters"]
    record = fake_instance_record(0)
    geometry = {
        "geometry_contract": "open_yard_right_aligned_bottom_gate_v1",
        "grid_rows": 5,
        "grid_cols": 5,
        "start_state": [1, 1],
        "door_cell": [0, 3],
        "pickup_cell": [1, 3],
        "waiting_cell": [0, 3],
        "exit_cells": [[4, 1], [4, 2], [4, 3]],
        "storage_positions": [
            [1, 1], [1, 2], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]
        ],
        "room_rows": ["###.#", "#...#", "#...#", "#...#", "#...#"],
        "geometry_signature": "f4d984dd3f4c27c7",
        "geometry_regime": "ordinary_default_gate",
        "requested_exit_width": None,
        "actual_exit_width": 3,
        "storage_cell_count": 8,
        "block_count": 8,
        "nominal_storage_density": 1.0,
    }
    return {
        "method": evaluation.TRACK_B_METHOD,
        "method_variant": evaluation.TRACK_B_METHOD_VARIANT,
        "evaluation_scope": evaluation.TRACK_B_EVALUATION_SCOPE,
        "controller_architecture": evaluation.TRACK_B_CONTROLLER_ARCHITECTURE,
        "controller_action_interface": evaluation.TRACK_B_CONTROLLER_ACTION_INTERFACE,
        "policy_realization": evaluation.TRACK_B_POLICY_REALIZATION,
        "assignment_source_family": "learned_spatial_assignment_policy",
        "assignment_source_version": "kim2020_grid_actor_critic_adapted_v1",
        "assignment_commitment": evaluation.DECISION_EPOCH_RESERVED,
        "assignment_source": evaluation.TRACK_A_KIM2020_A3C_SPATIAL,
        "track": "B",
        "information_regime": "online_arrived_only",
        "eval_seed": 85000,
        "instance_id": record["episode_instance_id"],
        "schedule_id": record["schedule_id"],
        "geometry": geometry,
        "assignment_policy_realization": evaluation.DEPLOYMENT_STOCHASTIC,
        "assignment_policy_seed": evaluation.policy_seed(0, 0, 0),
        "selector_deployment_digest": completion[
            "selected_checkpoint_deployment_sha256"
        ],
        "lambda": 10.0,
        "mu": 80.0,
        "target_window": 20.0,
        "delivery_deviations": list(deviations),
        "delivery_count": 8,
        **timing,
        "return": 100.0,
        "steps": 120,
        "contention_metric_schema_version": (
            "physical_storage_relocation_decomposition_v1"
        ),
        "physical_storage_relocations": physical,
        "target_bound_obstruction_clearances": physical,
        "standalone_reconfigurations": 0,
        "standalone_with_direct_delivery_available": 0,
        "standalone_without_direct_delivery_available": 0,
        "directly_deliverable_self_reconfigurations": 0,
        "relocations": physical,
        "obstructive_moves": physical,
        "success": 1,
        "strict_method_success": 1,
        "truncated": 0,
        "method_failure_reason": None,
        "illegal_drops": 0,
        "reservation_integrity": True,
        "storage_flow_fully_observed": True,
        "storage_flow_manifest_count": 8,
        "storage_flow_completed_count": 8,
        "storage_flow_unfinished_count": 0,
        "urgency_scheduler_audit": {
            "method": evaluation.TRACK_B_METHOD,
            "scheduler_architecture": evaluation.TRACK_B_CONTROLLER_ARCHITECTURE,
            "policy_realization": evaluation.TRACK_B_POLICY_REALIZATION,
            "controller_action_interface": evaluation.TRACK_B_CONTROLLER_ACTION_INTERFACE,
            "decision_epoch_contract": evaluation.TRACK_B_DECISION_EPOCH_CONTRACT,
            "assignment_commitment_contract": (
                evaluation.TRACK_B_ASSIGNMENT_COMMITMENT_CONTRACT
            ),
            "reservation_contract": evaluation.TRACK_B_RESERVATION_CONTRACT,
            "assignment_source": evaluation.TRACK_A_KIM2020_A3C_SPATIAL,
            "assignment_source_family": "learned_spatial_assignment_policy",
            "assignment_source_version": "kim2020_grid_actor_critic_adapted_v1",
            "lookahead_margin_steps": 2.0,
        },
        "selector_audit": {
            "frozen": True,
            "assignment_source": evaluation.TRACK_A_KIM2020_A3C_SPATIAL,
            "assignment_source_family": "learned_spatial_assignment_policy",
            "assignment_source_learned": True,
            "assignment_source_version": "kim2020_grid_actor_critic_adapted_v1",
            "decision_count": 8,
            "valid_assignment_count": 8,
            "reserved_commit_count": 8,
            "post_pickup_recompute_count": 0,
            "invalid_assignment_count": 0,
            "fallback_count": 0,
            "decisions": [
                {
                    "valid": True,
                    "commitment_contract": (
                        evaluation.TRACK_B_ASSIGNMENT_COMMITMENT_CONTRACT
                    ),
                    "proposal_id": proposal_id,
                }
                for proposal_id in proposal_ids
            ],
            "source_audit": {
                "assignment_source": evaluation.TRACK_A_KIM2020_A3C_SPATIAL,
                "assignment_source_family": "learned_spatial_assignment_policy",
                "assignment_source_version": "kim2020_grid_actor_critic_adapted_v1",
                "selector_feature_version": "kim2020_relative_dwell_spatial_v1",
                "reward_contract": "kim2020_delayed_target_retrieval_relocations_v1",
                "return_contract": "kim2020_episodic_discounted_placement_return_v1",
                "action_mapping": "row_major_grid_cell_masked_by_shared_candidates_v1",
                "learning_enabled": False,
                "deployment_mode": evaluation.DEPLOYMENT_STOCHASTIC,
                "policy_seed": evaluation.policy_seed(0, 0, 0),
                **source_counters,
                "last_episode": {
                    "instance_id": record["episode_instance_id"],
                    "success": True,
                    "truncated": False,
                    "outcome_tracking": False,
                    "integrity": None,
                    "trained": False,
                    "selection_count": 0,
                    "completed_count": 0,
                    "censored_count": 0,
                    "records": [],
                },
            },
        },
        "scheduler_audit": {
            "inbound_successes": 8,
            "inbound_failures": 0,
            "retrieve_successes": 8,
            "retrieve_failures": 0,
            "retrieve_relocations": physical,
            "reservation_bound_count": 8,
            "reservation_commit_count": 8,
            "reservation_execution_match_count": 8,
            "reservation_invalidation_count": 0,
            "bound_proposal_ids": proposal_ids,
            "committed_proposal_ids": proposal_ids,
        },
    }


def normalized_fake_row() -> dict:
    return evaluation._normalize_run(
        fake_raw(),
        grid_row=evaluation.expected_grid(0)[0],
        instance_record=fake_instance_record(0),
        training_completion=fake_training_completion(0),
        evaluation_manifest_sha256="m" * 64,
    )


def synthetic_analysis_row(model_seed: int, panel_index: int, rollout: int) -> dict:
    value = 100.0 * model_seed + 10.0 * panel_index + rollout
    return {
        "model_seed": model_seed,
        "panel_index": panel_index,
        "instance_seed": 85000 + panel_index,
        "rollout_index": rollout,
        "policy_rng_seed": evaluation.policy_seed(model_seed, panel_index, rollout),
        "episode_instance_id": f"instance-{85000 + panel_index}",
        "strict_safe_complete": True,
        "safety_issues": [],
        "dense_objective_return": value,
        "mean_absolute_error": value,
        "mean_signed_deviation": value,
        "mean_tardiness": value,
        "mean_earliness": value,
        "within_target_window_rate": value,
        "steps": value,
        "physical_storage_relocations": rollout,
        "physical_rehandles_per_100_required_deliveries": 12.5 * rollout,
    }


def fake_stability() -> dict:
    metrics = {
        metric: 10.0 for metric in evaluation.TABLE_METRICS
    }
    return {
        "equal_seed_aggregate": {
            "available": True,
            "model_seeds": [11, 12, 13],
            "metrics": metrics,
        },
        "individual_seed_summaries": [
            {
                "model_seed": seed,
                "development_candidate_eligible": True,
                "metrics": metrics,
            }
            for seed in (11, 12, 13)
        ],
    }


def fake_repair() -> dict:
    ids = [
        "vcg_constrained_v2_3_gamma1_episode160",
        "vcg_dense_v1_1_selected_three_seed",
        "duration_aware_nearest_free",
        "duration_aware_dynamic_pslap",
        "duration_aware_pslap_ga_2009_rolling",
        "duration_aware_pslap_ga_duration_aware_rolling",
        "duration_aware_pslap_ga_operational_rolling",
        "duration_aware_enhanced_complete_rolling_ga",
        "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial",
        "duration_aware_pslap_ga_duration_aware_rolling_capacity_aware_partial",
        "duration_aware_pslap_ga_operational_rolling_capacity_aware_partial",
        "duration_aware_enhanced_complete_rolling_ga_capacity_aware_partial",
    ]
    failed = ids[4:8]
    repaired = ids[8:12]
    summaries = []
    for method_id in ids:
        eligible = method_id not in failed
        summaries.append(
            {
                "method_id": method_id,
                "whole_method_numeric_eligible": eligible,
                "metrics": (
                    {metric: 20.0 for metric in evaluation.TABLE_METRICS}
                    if eligible
                    else None
                ),
            }
        )
    return {
        "method_summaries": summaries,
        "whole_method_safety_exclusions": failed,
        "method_registry": {
            "historical_overconstrained_ga_adapters_retained": [
                {"repaired_successor": successor} for successor in repaired
            ]
        },
    }


class GridTests(unittest.TestCase):
    def test_exact_grid(self):
        rows = evaluation.expected_grid()
        self.assertEqual(len(rows), 180)
        self.assertEqual(len({row["policy_seed"] for row in rows}), 180)
        self.assertEqual(rows[0]["policy_seed"], 632000000)
        self.assertEqual(rows[-1]["policy_seed"], 632002114)

    def test_no_sealed_namespaces(self):
        rows = evaluation.expected_grid()
        self.assertFalse(any(86000 <= row["instance_seed"] < 86030 for row in rows))
        self.assertFalse(any(622000000 <= row["policy_seed"] < 623000000 for row in rows))


class DenseRescoreTests(unittest.TestCase):
    def test_legacy_delivery_term_is_replaced_exactly(self):
        deviations = (-50.0, -10.0, 0.0, 30.0)
        legacy_total = 17.25
        legacy, dense = evaluation._dense_rescore(legacy_total, deviations)
        legacy_delivery = sum(
            evaluation.delivery_reward(
                deviation,
                evaluation.FROZEN_OBJECTIVE_SPEC,
                evaluation.LEGACY_CLIPPED,
            )
            for deviation in deviations
        )
        dense_delivery = sum(
            evaluation.delivery_reward(
                deviation,
                evaluation.FROZEN_OBJECTIVE_SPEC,
                evaluation.DENSE_PIECEWISE,
            )
            for deviation in deviations
        )
        self.assertEqual(legacy, legacy_total)
        self.assertAlmostEqual(
            dense, legacy_total - legacy_delivery + dense_delivery, places=12
        )

    def test_nonfinite_is_rejected_not_sanitized(self):
        with self.assertRaises(evaluation.EvaluationProtocolError):
            evaluation._json_safe(float("nan"))


class RowAuthenticationTests(unittest.TestCase):
    def test_normalize_and_reload_semantics(self):
        row = normalized_fake_row()
        self.assertTrue(row["strict_safe_complete"])
        evaluation._validate_normalized_row(
            row,
            expected=evaluation.expected_grid(0)[0],
            manifest=fake_manifest(),
            training_completion=fake_training_completion(0),
        )

    def test_rehashed_safety_claim_cannot_hide_unsafe_primitive(self):
        row = normalized_fake_row()
        row.pop("row_sha256")
        row["method_audit"]["invalid_assignment_count"] = 1
        row = evaluation._add_self_hash(row, "row_sha256")
        with self.assertRaises(evaluation.EvaluationProtocolError):
            evaluation._validate_normalized_row(
                row,
                expected=evaluation.expected_grid(0)[0],
                manifest=fake_manifest(),
                training_completion=fake_training_completion(0),
            )

    def test_identity_rebinding_rejects_rehashed_row(self):
        row = normalized_fake_row()
        row.pop("row_sha256")
        row["episode_instance_id"] = "substituted-instance"
        row = evaluation._add_self_hash(row, "row_sha256")
        with self.assertRaises(evaluation.EvaluationProtocolError):
            evaluation._validate_normalized_row(
                row,
                expected=evaluation.expected_grid(0)[0],
                manifest=fake_manifest(),
                training_completion=fake_training_completion(0),
            )

    def test_checkpoint_rebinding_rejects_rehashed_row(self):
        row = normalized_fake_row()
        row.pop("row_sha256")
        row["source_checkpoint_raw_sha256"] = "substituted-checkpoint"
        row = evaluation._add_self_hash(row, "row_sha256")
        with self.assertRaises(evaluation.EvaluationProtocolError):
            evaluation._validate_normalized_row(
                row,
                expected=evaluation.expected_grid(0)[0],
                manifest=fake_manifest(),
                training_completion=fake_training_completion(0),
            )

    def test_controller_rebinding_cannot_hide_behind_rehashed_safety(self):
        row = normalized_fake_row()
        row.pop("row_sha256")
        row["method_audit"]["controller_architecture"] = "adjacent-controller"
        row["safety_checks"] = evaluation._recompute_row_safety(
            row,
            training_completion=fake_training_completion(0),
            instance_record=fake_instance_record(0),
        )
        row["safety_issues"] = sorted(
            name for name, passed in row["safety_checks"].items() if not passed
        )
        row["strict_safe_complete"] = False
        row = evaluation._add_self_hash(row, "row_sha256")
        evaluation._validate_normalized_row(
            row,
            expected=evaluation.expected_grid(0)[0],
            manifest=fake_manifest(),
            training_completion=fake_training_completion(0),
        )
        self.assertIn("exact_track_b_executor", row["safety_issues"])

    def test_bool_as_integer_is_rejected_even_when_rehashed(self):
        row = normalized_fake_row()
        row.pop("row_sha256")
        row["source_neutral_scheduler"] = 1
        row = evaluation._add_self_hash(row, "row_sha256")
        with self.assertRaises(evaluation.EvaluationProtocolError):
            evaluation._validate_normalized_row(
                row,
                expected=evaluation.expected_grid(0)[0],
                manifest=fake_manifest(),
                training_completion=fake_training_completion(0),
            )

    def test_integer_as_float_is_rejected_even_when_rehashed(self):
        row = normalized_fake_row()
        row.pop("row_sha256")
        row["required_deliveries"] = 8.0
        row = evaluation._add_self_hash(row, "row_sha256")
        with self.assertRaises(evaluation.EvaluationProtocolError):
            evaluation._validate_normalized_row(
                row,
                expected=evaluation.expected_grid(0)[0],
                manifest=fake_manifest(),
                training_completion=fake_training_completion(0),
            )

    def test_step_limit_is_semantically_enforced(self):
        row = normalized_fake_row()
        row.pop("row_sha256")
        row["steps"] = 2001
        row["safety_checks"] = evaluation._recompute_row_safety(
            row,
            training_completion=fake_training_completion(0),
            instance_record=fake_instance_record(0),
        )
        row["safety_issues"] = sorted(
            name for name, passed in row["safety_checks"].items() if not passed
        )
        row["strict_safe_complete"] = False
        row = evaluation._add_self_hash(row, "row_sha256")
        with self.assertRaises(evaluation.EvaluationProtocolError):
            evaluation._validate_normalized_row(
                row,
                expected=evaluation.expected_grid(0)[0],
                manifest=fake_manifest(),
                training_completion=fake_training_completion(0),
            )


class HierarchicalAnalysisTests(unittest.TestCase):
    def make_ledgers(self):
        return {
            seed: {
                "rows": [
                    synthetic_analysis_row(seed, panel_index, rollout)
                    for panel_index in range(12)
                    for rollout in range(5)
                ]
            }
            for seed in range(3)
        }

    def test_exact_five_then_twelve_then_three_aggregation(self):
        summary = evaluation._summarize_kim_ledgers(self.make_ledgers())
        self.assertTrue(summary["whole_method_numeric_eligible"])
        self.assertEqual(len(summary["model_instance_points"]), 36)
        self.assertEqual(len(summary["model_summaries"]), 3)
        self.assertEqual(len(summary["instance_cluster_points"]), 12)
        # E[value] = E[100*m] + E[10*i] + E[r] = 100 + 55 + 2.
        self.assertAlmostEqual(summary["metrics"]["mean_absolute_error"], 157.0)
        self.assertAlmostEqual(
            summary["metrics"][
                "physical_rehandles_per_100_required_deliveries"
            ],
            25.0,
        )
        self.assertEqual(
            summary["instance_cluster_statistics"]["mean_absolute_error"][
                "n_episode_instance_clusters"
            ],
            12,
        )

    def test_one_unsafe_row_suppresses_whole_method(self):
        ledgers = self.make_ledgers()
        ledgers[2]["rows"][-1]["strict_safe_complete"] = False
        ledgers[2]["rows"][-1]["safety_issues"] = ["forced_test_failure"]
        summary = evaluation._summarize_kim_ledgers(ledgers)
        self.assertFalse(summary["whole_method_numeric_eligible"])
        self.assertIsNone(summary["metrics"])
        self.assertEqual(summary["unsafe_row_count"], 1)
        self.assertEqual(summary["model_instance_points"], [])


class TableAndGateTests(unittest.TestCase):
    def kim_summary(self, mae=30.0, rehandles=30.0):
        metrics = {metric: 30.0 for metric in evaluation.REPORT_METRICS}
        metrics["mean_absolute_error"] = mae
        metrics["physical_rehandles_per_100_required_deliveries"] = rehandles
        return {"whole_method_numeric_eligible": True, "metrics": metrics}

    def test_exact_nine_method_table_includes_dynamic_pslap(self):
        table, suppressed = evaluation._comparison_table(
            fake_repair(), fake_stability(), self.kim_summary()
        )
        self.assertEqual(
            [row["method_id"] for row in table],
            list(evaluation.NUMERIC_TABLE_METHOD_IDS),
        )
        self.assertEqual(len(table), 9)
        self.assertIn("duration_aware_dynamic_pslap", [row["method_id"] for row in table])
        self.assertEqual(len(suppressed), 4)

    def test_extended_gate_adds_kim_without_requiring_kim_to_win(self):
        table, _ = evaluation._comparison_table(
            fake_repair(), fake_stability(), self.kim_summary()
        )
        gate = evaluation._extended_gate(table, fake_stability(), self.kim_summary())
        self.assertEqual(len(gate["comparator_method_ids"]), 8)
        self.assertIn(evaluation.KIM_STOCHASTIC, gate["comparator_method_ids"])
        self.assertTrue(gate["passed"])
        self.assertFalse(gate["kim_required_to_win"])

    def test_gate_is_unevaluable_when_kim_is_whole_method_unsafe(self):
        kim = {"whole_method_numeric_eligible": False, "metrics": None}
        table, _ = evaluation._comparison_table(fake_repair(), fake_stability(), kim)
        gate = evaluation._extended_gate(table, fake_stability(), kim)
        self.assertIsNone(gate["passed"])
        self.assertEqual(gate["individual_seed_screens"], [])


class RuntimeAndLockTests(unittest.TestCase):
    @staticmethod
    def make_structural_output(root: Path) -> None:
        (root / "evaluation-run-manifest.json").touch()
        for directory in (
            "training-completion", "run-ledger", "seed-completion", "run-locks"
        ):
            (root / directory).mkdir()
        for seed in evaluation.MODEL_SEEDS:
            (root / "training-completion" / f"seed-{seed}.json").touch()
            (root / "run-locks" / f"seed-{seed}.lock").touch()

    def test_runtime_is_observed_fixed(self):
        evaluation._configure_runtime()
        runtime = evaluation._fixed_runtime()
        self.assertTrue(runtime["deterministic_algorithms"])
        self.assertEqual(runtime["torch_num_threads"], 1)

    def test_same_seed_lock_is_nonblocking_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "run-locks").mkdir()
            (output / "run-locks" / "seed-0.lock").touch()
            evaluator_calls = 0
            with evaluation._exclusive_seed_lock(output, 0):
                with self.assertRaises(evaluation.EvaluationProtocolError):
                    with evaluation._exclusive_seed_lock(output, 0):
                        evaluator_calls += 1
            self.assertEqual(evaluator_calls, 0)

    def test_write_once_json_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            evaluation._write_new_json(path, {"value": 1})
            with self.assertRaises(evaluation.EvaluationProtocolError):
                evaluation._write_new_json(path, {"value": 2})

    def test_intermediate_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            (real / "artifact.json").write_text("{}", encoding="utf-8")
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaises(evaluation.EvaluationProtocolError):
                evaluation._read_regular_bytes(linked / "artifact.json")

    def test_report_only_transition_is_recoverable_but_audit_only_is_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.make_structural_output(output)
            (output / "extended-report.json").touch()
            evaluation._validate_output_tree(output)
            (output / "extended-report.json").unlink()
            (output / "extended-audit.json").touch()
            with self.assertRaises(evaluation.EvaluationProtocolError):
                evaluation._validate_output_tree(output)


if __name__ == "__main__":
    unittest.main()
