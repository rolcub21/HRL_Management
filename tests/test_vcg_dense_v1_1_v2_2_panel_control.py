import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import evaluate_vcg_dense_v1_1_v2_2_panel_control as control


def _row(model_seed, *, dense_return, deviations, steps, rehandles):
    return {
        "model_seed": model_seed,
        "dense_objective_return": dense_return,
        "delivery_deviations": tuple(deviations),
        "delivery_count": len(deviations),
        "steps": steps,
        "physical_storage_relocations": rehandles,
        "strict_method_success": True,
        "completion_rate": 1.0,
        "all_selected_candidates_exact_safe": True,
        "illegal_drops": 0,
        "fallbacks": 0,
        "method_failure_reason": None,
        "evaluation_learning": False,
        "evaluation_epsilon": 0.0,
    }


class V11V22PanelControlTests(unittest.TestCase):
    def test_frozen_panel_is_only_opened_85xxx_and_never_final_86xxx(self):
        control._validate_protocol_constants()
        self.assertEqual(control.EVALUATION_SEEDS, tuple(range(85_000, 85_012)))
        self.assertFalse(
            set(control.EVALUATION_SEEDS).intersection(control.FINAL_PANEL_SEEDS)
        )
        self.assertEqual(control.MODEL_SEEDS, (0, 1, 2))
        self.assertEqual(control.V22_REFERENCE_EPISODE, 80)

    def test_embedded_hash_authentication_fails_closed_on_tampering(self):
        payload = {"schema_version": 1, "opened": False}
        payload["manifest_sha256"] = control._digest_json(payload)
        self.assertEqual(
            control._verify_embedded_hash(
                payload, "manifest_sha256", name="synthetic manifest"
            ),
            payload["manifest_sha256"],
        )
        tampered = copy.deepcopy(payload)
        tampered["opened"] = True
        with self.assertRaisesRegex(control.V11PanelControlError, "mismatch"):
            control._verify_embedded_hash(
                tampered, "manifest_sha256", name="synthetic manifest"
            )

    def test_position_metrics_preserve_first_two_and_later_split(self):
        rows = (
            {"delivery_deviations": (-10, -30, 5, 25)},
            {"delivery_deviations": (10, 20, -5, -15)},
        )
        summary = control._delivery_position_summary(rows, window=20.0)
        self.assertEqual(len(summary["by_delivery_position"]), 4)
        self.assertEqual(summary["first_two"]["n"], 4)
        self.assertAlmostEqual(summary["first_two"]["mean_absolute_error"], 17.5)
        self.assertAlmostEqual(summary["first_two"]["within_target_window_rate"], 0.75)
        self.assertEqual(summary["positions_three_plus"]["n"], 4)
        self.assertAlmostEqual(
            summary["positions_three_plus"]["mean_absolute_error"], 12.5
        )
        self.assertEqual(
            tuple(item["n"] for item in summary["by_delivery_position"]),
            (2, 2, 2, 2),
        )

    def test_equal_seed_aggregate_weights_model_seeds_equally(self):
        rows = {
            0: (_row(0, dense_return=10, deviations=(-10, 10), steps=100, rehandles=0),),
            1: (_row(1, dense_return=20, deviations=(-20, 20), steps=200, rehandles=1),),
            2: (_row(2, dense_return=30, deviations=(-30, 30), steps=300, rehandles=2),),
        }
        summaries = {seed: control._summarize_rows(value) for seed, value in rows.items()}
        aggregate = control._equal_seed_aggregate(summaries)
        self.assertEqual(aggregate["model_seed_count"], 3)
        self.assertAlmostEqual(aggregate["mean_dense_return"], 20.0)
        self.assertAlmostEqual(aggregate["mean_absolute_error"], 20.0)
        self.assertAlmostEqual(aggregate["mean_steps"], 200.0)
        self.assertAlmostEqual(
            aggregate["physical_rehandles_per_100_deliveries"], 50.0
        )
        self.assertTrue(aggregate["all_complete_exact_safe"])

    def test_summary_rejects_delivery_count_sequence_mismatch(self):
        row = _row(0, dense_return=10, deviations=(-1, 1), steps=100, rehandles=0)
        row["delivery_count"] = 3
        with self.assertRaisesRegex(control.V11PanelControlError, "delivery count"):
            control._summarize_rows((row,))

    def test_executor_requires_frozen_no_learning_exact_safe_rollout(self):
        frontier = {
            "complete_frontier_exactly_verified": True,
            "candidates": ({"certificate": {"status": "SAFE"}},),
        }
        raw = {
            "training": False,
            "gradient_steps": 0,
            "complete_frontier_exactly_verified": True,
            "frontiers": (frontier,),
        }
        finished = {
            "method_audit": {
                "frontiers": (frontier,),
                "complete_frontier_exactly_verified": True,
            }
        }
        arm = SimpleNamespace(
            payload={"environment": {"number_blocks": 8}},
            model_seed=1,
        )
        with (
            patch.object(control.pareto, "run_arm", return_value=raw) as run,
            patch.object(control.pareto, "_search_config", return_value="search"),
            patch.object(control.pareto, "_liveness_rule", return_value="liveness"),
            patch.object(control.pareto, "_normalize_vcg", return_value={"normalized": True}),
            patch.object(control.pareto, "_identity_for_policy", return_value={}),
            patch.object(control.pareto, "_finish_row", return_value=finished),
        ):
            result = control._execute_one(
                arm, SimpleNamespace(), 85_000, torch.device("cpu")
            )
        self.assertFalse(result["evaluation_learning"])
        self.assertEqual(result["evaluation_epsilon"], 0.0)
        self.assertEqual(result["gradient_steps"], 0)
        self.assertTrue(result["all_selected_candidates_exact_safe"])
        self.assertTrue(result["method_audit"]["all_frontiers_complete_exact"])
        self.assertNotIn("frontiers", result["method_audit"])
        self.assertEqual(run.call_args.kwargs["arm"], control.pareto.EXACT_FULL)
        self.assertEqual(run.call_args.kwargs["max_steps"], control.FROZEN_MAX_STEPS)

    def test_executor_rejects_any_learning_or_inexact_frontier(self):
        arm = SimpleNamespace(
            payload={"environment": {"number_blocks": 8}},
            model_seed=0,
        )
        unsafe = {
            "training": True,
            "gradient_steps": 1,
            "complete_frontier_exactly_verified": False,
            "frontiers": (),
        }
        with (
            patch.object(control.pareto, "run_arm", return_value=unsafe),
            patch.object(control.pareto, "_search_config", return_value="search"),
            patch.object(control.pareto, "_liveness_rule", return_value="liveness"),
        ):
            with self.assertRaisesRegex(control.V11PanelControlError, "training flag"):
                control._execute_one(
                    arm, SimpleNamespace(), 85_000, torch.device("cpu")
                )


if __name__ == "__main__":
    unittest.main()
