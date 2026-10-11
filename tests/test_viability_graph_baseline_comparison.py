import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from benchmark_viability_critic_priority import (
    _freeze_agent,
    _load_controller_checkpoint,
)
from compare_viability_graph_baselines import (
    DYNAMIC_METHOD,
    VCG_METHOD,
    _checkpoint_readiness,
    _dual_rescore_from_legacy_return,
    _normalize_baseline,
    _paired_comparisons,
    _summary,
)
from vcg_objective_audit import TimingObjectiveSpec
from viability_graph_episodic_audit import (
    EpisodicViabilityGraphConfig,
    EpisodicViabilityGraphHierarchyAgent,
)


def row(method, instance, *, return_value, steps, mae, strict=1.0):
    return {
        "method": method,
        "instance_seed": 0,
        "instance_id": instance,
        "return": float(return_value),
        "legacy_rescored_return": float(return_value),
        "dense_rescored_return": float(return_value),
        "success": float(strict),
        "strict_method_success": float(strict),
        "completion_rate": float(strict),
        "steps": int(steps),
        "mean_absolute_error": float(mae),
        "first_two_mean_absolute_error": float(mae),
        "positions_three_plus_mean_absolute_error": float(mae),
        "mean_tardiness": 0.0,
        "mean_earliness": float(mae),
        "within_target_window_rate": 0.5,
        "first_two_within_target_window_rate": 0.5,
        "positions_three_plus_within_target_window_rate": 0.5,
        "contention_metric_schema_version": (
            "physical_storage_relocation_decomposition_v1"
        ),
        "physical_storage_relocations": 0,
        "target_bound_obstruction_clearances": 0,
        "standalone_reconfigurations": 0,
        "standalone_with_direct_delivery_available": 0,
        "standalone_without_direct_delivery_available": 0,
        "directly_deliverable_self_reconfigurations": 0,
        "relocations": 0,
        "obstructive_moves": 0,
        "delivery_count": int(8 * strict),
        "illegal_drops": 0,
        "invalid_assignments": 0,
        "infeasible_epochs": 0,
        "fallbacks": 0,
        "planning_seconds": 0.1,
        "method_failure_reason": None if strict else "failed",
    }


class VcgBaselineComparisonContractTests(unittest.TestCase):
    def test_event_time_baseline_relocation_survives_unterminated_option_audit(self):
        raw = {
            "eval_seed": 7,
            "instance_id": "instance-7",
            "schedule_id": "schedule-7",
            "return": -1.0,
            "success": 0.0,
            "strict_method_success": 0.0,
            "steps": 10,
            "delivery_count": 0,
            "delivery_deviations": (),
            "mean_signed_deviation": None,
            "mean_absolute_error": None,
            "mean_tardiness": None,
            "mean_earliness": None,
            "within_target_window_rate": None,
            "p90_tardiness": None,
            "p90_absolute_error": None,
            "contention_metric_schema_version": (
                "physical_storage_relocation_decomposition_v1"
            ),
            "physical_storage_relocations": 1,
            "target_bound_obstruction_clearances": 1,
            "standalone_reconfigurations": 0,
            "standalone_with_direct_delivery_available": 0,
            "standalone_without_direct_delivery_available": 0,
            "directly_deliverable_self_reconfigurations": 0,
            "relocations": 1,
            "obstructive_moves": 1,
            "illegal_drops": 0,
            "method_failure_reason": "episode_step_limit",
            "decision_seconds": 0.2,
            "assignment_source": "fixture",
            "assignment_source_version": "v1",
            "controller_action_interface": "fixture",
            "controller_architecture": "fixture",
            "assignment_commitment": "fixture",
            "reservation_integrity": True,
            "selector_audit": {
                "assignment_seconds": 0.1,
                "invalid_assignment_count": 0,
                "infeasible_epoch_count": 0,
                "fallback_count": 0,
            },
            "urgency_scheduler_audit": {
                "preview_seconds": 0.1,
                "preview_failure_count": 0,
            },
            # The move happened, but truncation prevented the option from
            # entering this completed-option audit.
            "scheduler_audit": {
                "retrieve_relocations": 0,
                "retrieve_replans": 0,
                "inbound_failures": 0,
                "retrieve_failures": 0,
            },
        }

        normalized = _normalize_baseline(
            DYNAMIC_METHOD,
            raw,
            SimpleNamespace(schedule_id="schedule-7"),
            8,
            TimingObjectiveSpec.dense(),
        )

        self.assertEqual(normalized["physical_storage_relocations"], 1)
        self.assertEqual(normalized["target_bound_obstruction_clearances"], 1)
        self.assertEqual(
            normalized["method_audit"]["completed_option_retrieve_relocations"],
            0,
        )

    def test_dense_rescore_replaces_only_realized_delivery_terms(self):
        legacy, dense = _dual_rescore_from_legacy_return(
            100.0, (-65.0,), TimingObjectiveSpec.dense()
        )

        self.assertEqual(legacy, 100.0)
        # At |e|=65, legacy delivery reward is 10 and dense is -80.
        self.assertEqual(dense, 10.0)

    def test_freezer_dispatches_strictly_to_episodic_checkpoint_loader(self):
        agent = EpisodicViabilityGraphHierarchyAgent(
            config=EpisodicViabilityGraphConfig(
                gamma=0.99, episode_horizon_steps=2_000
            ),
            seed=3,
            device="cpu",
        )
        payload = agent.checkpoint(include_replay=False)

        frozen = _freeze_agent(payload, device=torch.device("cpu"), seed=3)

        self.assertIsInstance(frozen, EpisodicViabilityGraphHierarchyAgent)
        self.assertEqual(frozen.epsilon, 0.0)
        self.assertFalse(frozen.Q_local.training)

    def test_objective_screen_readiness_cannot_authorize_a_claim(self):
        readiness = _checkpoint_readiness(
            {
                "protocol": "vcg_objective_gamma_factorial_arm_training_v1",
                "development_mechanism_screen_only": True,
                "completed_training_episodes": 100,
                "best_validation_record": {"deployment_eligible": True},
                "agent_state": {},
            }
        )

        self.assertTrue(readiness["development_mechanism_screen_checkpoint"])
        self.assertFalse(readiness["performance_claim_authorized"])
        self.assertIn("three_seed", readiness["interpretation"])

    def test_calibration_checkpoint_is_explicitly_non_claim_bearing(self):
        readiness = _checkpoint_readiness(
            {
                "protocol": "vcg_smdp_calibration_only_smoke_v1",
                "instance_regime": "calibration_only",
                "train_instance_seeds": (1,),
                "agent_state": {
                    "transition_count": 25,
                    "gradient_steps": 18,
                    "target_updates": 0,
                    "epsilon": 0.816,
                },
            }
        )

        self.assertTrue(readiness["calibration_smoke_checkpoint"])
        self.assertFalse(readiness["performance_claim_authorized"])
        self.assertEqual(readiness["training_episodes"], 1)
        self.assertEqual(readiness["target_updates"], 0)

    def test_proper_checkpoint_readiness_uses_completed_episode_clock(self):
        readiness = _checkpoint_readiness(
            {
                "protocol": "vcg_smdp_contention_proper_training_v1",
                "instance_regime": "contention_proper_training",
                "completed_training_episodes": 500,
                "best_validation_record": {"deployment_eligible": True},
                "agent_state": {
                    "transition_count": 10_501,
                    "gradient_steps": 10_374,
                    "target_updates": 51,
                    "epsilon": 0.05,
                },
            }
        )

        self.assertEqual(readiness["training_episodes"], 500)
        self.assertTrue(readiness["deployment_checkpoint_eligible"])
        self.assertFalse(readiness["performance_claim_authorized"])
        self.assertIn("single_training_seed", readiness["interpretation"])

    def test_dense_proper_readiness_requires_both_finalization_flags(self):
        base = {
            "protocol": "vcg_dense_g099_proper_training_v1_1",
            "proper_training_confirmation_candidate": True,
            "completed_training_episodes": 500,
            "best_validation_record": {"deployment_eligible": True},
            "agent_state": {},
        }
        incomplete_flags = (
            {},
            {"protocol_training_complete": True},
            {"selection_finalized_after_total_episodes": True},
            {
                "protocol_training_complete": False,
                "selection_finalized_after_total_episodes": True,
            },
            {
                "protocol_training_complete": True,
                "selection_finalized_after_total_episodes": False,
            },
        )

        for flags in incomplete_flags:
            with self.subTest(flags=flags):
                readiness = _checkpoint_readiness({**base, **flags})
                self.assertTrue(
                    readiness["proper_training_confirmation_candidate"]
                )
                self.assertFalse(readiness["deployment_checkpoint_eligible"])
                self.assertEqual(
                    readiness["interpretation"],
                    "dense_proper_training_not_finalized",
                )

    def test_dense_proper_readiness_requires_eligible_final_selection(self):
        finalized = {
            "protocol": "vcg_dense_g099_proper_training_v1_1",
            "proper_training_confirmation_candidate": True,
            "protocol_training_complete": True,
            "selection_finalized_after_total_episodes": True,
            "checkpoint_role": "best_deployment_finalized",
            "trainer_resumable": False,
            "deployment_checkpoint_eligible": True,
            "completed_training_episodes": 500,
            "best_validation_record": {"deployment_eligible": True},
            "agent_state": {},
        }

        readiness = _checkpoint_readiness(finalized)
        self.assertTrue(readiness["protocol_training_complete"])
        self.assertTrue(readiness["selection_finalized_after_total_episodes"])
        self.assertTrue(readiness["best_validation_deployment_eligible"])
        self.assertTrue(readiness["finalized_best_deployment_artifact"])
        self.assertTrue(readiness["deployment_checkpoint_eligible"])
        self.assertIn("single_training_seed", readiness["interpretation"])

        ineligible = _checkpoint_readiness(
            {
                **finalized,
                "best_validation_record": {"deployment_eligible": False},
            }
        )
        self.assertFalse(ineligible["best_validation_deployment_eligible"])
        self.assertFalse(ineligible["deployment_checkpoint_eligible"])
        self.assertEqual(
            ineligible["interpretation"],
            "dense_proper_finalized_best_ineligible",
        )

        latest = _checkpoint_readiness(
            {
                **finalized,
                "checkpoint_role": "latest_resumable",
                "trainer_resumable": True,
                "deployment_checkpoint_eligible": False,
            }
        )
        self.assertFalse(latest["finalized_best_deployment_artifact"])
        self.assertFalse(latest["deployment_checkpoint_eligible"])
        self.assertEqual(
            latest["interpretation"],
            "dense_proper_use_finalized_best_checkpoint",
        )

    def test_loader_normalizes_proper_resume_contract_without_rewriting(self):
        payload = {
            "training_protocol": "vcg_smdp_contention_proper_training_v1",
            "completed_training_episodes": 2,
            "exact_safe_mask_authoritative": True,
            "baseline_teacher": False,
            "resume_contract": {
                "train_instance_seed_base": 30_000_000,
                "environment": {
                    "grid_rows": 5,
                    "grid_cols": 5,
                    "number_blocks": 8,
                    "arrival_rate": 10.0,
                    "proc_mean": 80,
                },
                "search_config": {"max_nodes": 20_000},
                "liveness_rule": {
                    "max_option_steps": 10,
                    "max_consecutive_defer_decisions": 16,
                },
            },
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "proper.pth"
            torch.save(payload, path)
            loaded = _load_controller_checkpoint(path)

        self.assertEqual(loaded["environment"]["number_blocks"], 8)
        self.assertEqual(loaded["viability_search"]["max_nodes"], 20_000)
        self.assertEqual(loaded["protocol"], payload["training_protocol"])
        self.assertEqual(
            loaded["train_instance_seeds"], (30_000_000, 30_000_001)
        )

    def test_paired_metric_orientation_is_positive_when_vcg_is_better(self):
        runs = [
            row(VCG_METHOD, "a", return_value=20, steps=10, mae=2),
            row(DYNAMIC_METHOD, "a", return_value=15, steps=12, mae=4),
            row(VCG_METHOD, "b", return_value=30, steps=20, mae=3),
            row(DYNAMIC_METHOD, "b", return_value=25, steps=21, mae=5),
        ]

        comparison = _paired_comparisons(
            runs, (VCG_METHOD, DYNAMIC_METHOD), samples=100
        )[0]

        self.assertEqual(comparison["metric_orientation"], "positive_favors_vcg_smdp")
        self.assertEqual(
            comparison["metrics"]["return_advantage"]["mean"], 5.0
        )
        self.assertEqual(
            comparison["metrics"]["step_reduction"]["mean"], 1.5
        )
        self.assertEqual(
            comparison["metrics"]["absolute_error_reduction"]["mean"], 2.0
        )

    def test_summary_uses_only_finite_timing_values(self):
        complete = row(VCG_METHOD, "a", return_value=10, steps=8, mae=4)
        failed = row(
            VCG_METHOD,
            "b",
            return_value=0,
            steps=2,
            mae=math.nan,
            strict=0,
        )

        summary = _summary(VCG_METHOD, (complete, failed))

        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["strict_method_success_rate"], 0.5)
        self.assertEqual(summary["mean_absolute_error"], 4.0)
        self.assertEqual(len(summary["method_failures"]), 1)


if __name__ == "__main__":
    unittest.main()
