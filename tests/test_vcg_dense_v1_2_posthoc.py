from pathlib import Path
import unittest

from compare_vcg_dense_pareto import EVALUATION_SEEDS, PolicyArm
from compare_vcg_dense_v1_2_posthoc import (
    ENHANCED_GA_METHOD,
    EXPECTED_ARTIFACTS,
    EXPECTED_CHOSEN_EPISODES,
    MAE_MARGIN,
    POSTHOC_GROUP,
    SELECTION_RULE,
    _build_report,
    _full_validation_eligible,
    _paired,
    _posthoc_row,
    _select_posthoc_record,
)
from train_vcg_dense_proper import FROZEN_VALIDATION_SEEDS


def _record(episode, *, mae, relocations, dense, eligible=True):
    return {
        "checkpoint_episode": episode,
        "deployment_eligible": eligible,
        "summary": {
            "episodes": len(FROZEN_VALIDATION_SEEDS),
            "instance_seeds": list(FROZEN_VALIDATION_SEEDS),
            "strict_method_success_rate": 1.0,
            "completion_rate": 1.0,
            "delivery_count": len(FROZEN_VALIDATION_SEEDS) * 8,
            "method_failures": [],
            "mean_absolute_error": mae,
            "relocations_per_100_deliveries": relocations,
            "mean_dense_rescored_return": dense,
        },
    }


def _selection_summary(model_seed):
    reference_episode = {0: 500, 1: 450, 2: 375}[model_seed]
    chosen_episode = EXPECTED_CHOSEN_EPISODES[model_seed]
    history = []
    for episode in range(25, 501, 25):
        history.append(
            _record(
                episode,
                mae=13.0,
                relocations=25.0,
                dense=100.0,
            )
        )
    by_episode = {item["checkpoint_episode"]: item for item in history}
    reference = by_episode[reference_episode]
    reference["summary"].update(
        mean_absolute_error=10.0,
        relocations_per_100_deliveries=20.0,
        mean_dense_rescored_return=120.0,
    )
    chosen = by_episode[chosen_episode]
    chosen["summary"].update(
        mean_absolute_error=11.5,
        relocations_per_100_deliveries=5.0,
        mean_dense_rescored_return=115.0,
    )
    return {"validation_history": history, "best_validation_record": reference}


def _eval_row(seed, method, *, dense, mae, relocations):
    return {
        "method_id": method,
        "model_seed": seed if method.startswith("posthoc") else None,
        "instance_seed": seed,
        "instance_id": f"instance-{seed}",
        "schedule_id": f"schedule-{seed}",
        "strict_method_success": 1.0,
        "completion_rate": 1.0,
        "delivery_count": 8,
        "dense_rescored_return": dense,
        "mean_absolute_error": mae,
        "first_two_mean_absolute_error": mae + 1.0,
        "positions_three_plus_mean_absolute_error": mae - 1.0,
        "mean_tardiness": mae / 2.0,
        "mean_earliness": mae / 2.0,
        "within_target_window_rate": 0.8,
        "steps": 200,
        "relocations": relocations,
        "obstructive_moves": relocations,
    }


class VcgDenseV12PosthocTests(unittest.TestCase):
    def test_frozen_rule_selects_expected_preserved_episodes(self):
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                result = _select_posthoc_record(_selection_summary(seed), seed)
                self.assertEqual(
                    result["chosen_episode"], EXPECTED_CHOSEN_EPISODES[seed]
                )
                self.assertEqual(result["mae_threshold"], 10.0 + MAE_MARGIN)
                self.assertEqual(result["selection_rule"], SELECTION_RULE)
        self.assertEqual(
            EXPECTED_ARTIFACTS,
            {
                0: "best-candidate-slot-a.pth",
                1: "best-candidate-slot-a.pth",
                2: "latest.pth",
            },
        )

    def test_rule_requires_strict_full_deployment_eligible_validation(self):
        record = _record(25, mae=10.0, relocations=1.0, dense=100.0)
        self.assertTrue(_full_validation_eligible(record))
        for mutation in (
            ("deployment_eligible", False),
            ("strict_method_success_rate", 0.95),
            ("completion_rate", 0.99),
            ("delivery_count", 159),
            ("episodes", 19),
        ):
            broken = _record(25, mae=10.0, relocations=1.0, dense=100.0)
            name, value = mutation
            if name == "deployment_eligible":
                broken[name] = value
            else:
                broken["summary"][name] = value
            with self.subTest(field=name):
                self.assertFalse(_full_validation_eligible(broken))

    def test_tie_break_prefers_dense_then_mae_then_earlier(self):
        summary = _selection_summary(0)
        by_episode = {
            item["checkpoint_episode"]: item
            for item in summary["validation_history"]
        }
        # Same minimum relocation as episode 475, but lower dense return.
        by_episode[450]["summary"].update(
            mean_absolute_error=10.5,
            relocations_per_100_deliveries=5.0,
            mean_dense_rescored_return=114.0,
        )
        result = _select_posthoc_record(summary, 0)
        self.assertEqual(result["chosen_episode"], 475)

    def test_paired_deltas_have_positive_favors_posthoc_orientation(self):
        chosen = [
            _eval_row(seed, "posthoc", dense=20.0, mae=5.0, relocations=1)
            for seed in EVALUATION_SEEDS
        ]
        baseline = [
            _eval_row(seed, "baseline", dense=10.0, mae=8.0, relocations=3)
            for seed in EVALUATION_SEEDS
        ]
        result = _paired(chosen, baseline, "baseline")
        self.assertEqual(result["metric_orientation"], "positive_favors_posthoc_v1_2")
        self.assertEqual(
            result["metrics"]["dense_return_advantage"]["mean"], 10.0
        )
        self.assertEqual(
            result["metrics"]["absolute_error_reduction"]["mean"], 3.0
        )
        self.assertEqual(
            result["metrics"]["relocation_reduction"]["mean"], 2.0
        )

    def test_reused_seed2_row_remains_posthoc_and_deployment_ineligible(self):
        arm = PolicyArm(
            method_id="vcg_dense_seed2_posthoc_v1_2_ep500",
            policy_group=POSTHOC_GROUP,
            model_seed=2,
            checkpoint_variant="posthoc_preserved_artifact",
            checkpoint_weight_episode=500,
            checkpoint_path=Path("latest.pth"),
            checkpoint_sha256="a" * 64,
            deployment_policy_digest="b" * 64,
            primary_analysis=False,
            diagnostic_only=True,
            readiness={
                "deployment_checkpoint_eligible": False,
                "interpretation": "diagnostic",
            },
            payload={},
        )
        source = _eval_row(80000, "source-final", dense=10, mae=5, relocations=1)
        row = _posthoc_row(source, arm, reused_from="source-final")
        self.assertTrue(row["execution_reused"])
        self.assertFalse(row["deployment_checkpoint_eligible"])
        self.assertTrue(row["diagnostic_only"])
        self.assertEqual(row["policy_group"], POSTHOC_GROUP)

    def test_report_never_promotes_posthoc_selection(self):
        rows = []
        source = {ENHANCED_GA_METHOD: []}
        selections = []
        artifacts = []
        for model_seed in (0, 1, 2):
            method = f"posthoc-{model_seed}"
            selected_method = f"vcg_dense_seed{model_seed}_selected_best"
            source[selected_method] = []
            selections.append(
                {
                    "model_seed": model_seed,
                    "chosen_episode": EXPECTED_CHOSEN_EPISODES[model_seed],
                }
            )
            artifacts.append({"model_seed": model_seed})
            for instance_seed in EVALUATION_SEEDS:
                chosen = _eval_row(
                    instance_seed,
                    method,
                    dense=20.0,
                    mae=5.0,
                    relocations=1,
                )
                chosen["model_seed"] = model_seed
                rows.append(chosen)
                source[selected_method].append(
                    _eval_row(
                        instance_seed,
                        selected_method,
                        dense=19.0,
                        mae=5.5,
                        relocations=2,
                    )
                )
                if model_seed == 0:
                    source[ENHANCED_GA_METHOD].append(
                        _eval_row(
                            instance_seed,
                            ENHANCED_GA_METHOD,
                            dense=18.0,
                            mae=6.0,
                            relocations=2,
                        )
                    )
        report = _build_report(rows, source, selections, artifacts)
        self.assertFalse(report["performance_claim_authorized"])
        self.assertFalse(report["deployment_checkpoint_eligible"])
        self.assertTrue(report["guardrails"]["no_new_baseline_runs"])
        self.assertEqual(report["guardrails"]["physical_new_rollouts"], 60)
        contrast = report["aggregate_contrasts"][
            "candidate_vs_selected_best"
        ]["metrics"]
        self.assertEqual(
            contrast["relocations_saving_per_100_deliveries"][
                "mean_across_training_seeds"
            ],
            12.5,
        )
        self.assertEqual(
            contrast["mae_cost"]["mean_across_training_seeds"], -0.5
        )
        self.assertEqual(
            contrast["mae_cost"]["t_degrees_of_freedom"], 2
        )


if __name__ == "__main__":
    unittest.main()
