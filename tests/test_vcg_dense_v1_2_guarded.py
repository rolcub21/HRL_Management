import unittest

from compare_vcg_dense_pareto import EVALUATION_SEEDS
from derive_vcg_dense_v1_2_guarded import (
    EXPECTED_GUARDED_EPISODES,
    GUARD_MAE_MARGIN,
    GUARDED_GROUP,
    _build_report,
    _relabel_guarded,
    _sequential_guard,
    _strict_full,
    _validation_rank,
)
from compare_vcg_dense_v1_2_posthoc import (
    ENHANCED_GA_METHOD,
    SOURCE_SELECTED_METHODS,
)


def _row(instance_seed, *, mae, relocations, dense=100.0):
    return {
        "method_id": "synthetic",
        "model_seed": None,
        "instance_seed": instance_seed,
        "instance_id": f"instance-{instance_seed}",
        "schedule_id": f"schedule-{instance_seed}",
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


def _panel(*, mae, relocations, dense=100.0):
    return [
        _row(seed, mae=mae, relocations=relocations, dense=dense)
        for seed in EVALUATION_SEEDS
    ]


def _selection(reference_episode, ranked_episodes):
    table = []
    for rank, episode in enumerate(ranked_episodes):
        table.append(
            {
                "checkpoint_episode": episode,
                "feasible": True,
                "relocations_per_100_deliveries": float(rank),
                "mean_dense_rescored_return": 100.0 - rank,
                "mean_absolute_error": 10.0,
            }
        )
    return {
        "reference_episode": reference_episode,
        "selection_table": table,
    }


class VcgDenseV12GuardedTests(unittest.TestCase):
    def test_validation_rank_is_frozen_before_panel_b_guard(self):
        selection = {
            "selection_table": [
                {
                    "checkpoint_episode": 25,
                    "feasible": True,
                    "relocations_per_100_deliveries": 5.0,
                    "mean_dense_rescored_return": 100.0,
                    "mean_absolute_error": 10.0,
                },
                {
                    "checkpoint_episode": 50,
                    "feasible": True,
                    "relocations_per_100_deliveries": 2.0,
                    "mean_dense_rescored_return": 90.0,
                    "mean_absolute_error": 11.0,
                },
            ]
        }
        rank = _validation_rank(selection)
        self.assertEqual([item["checkpoint_episode"] for item in rank], [50, 25])

    def test_seed1_rejects_first_ranked_candidate_then_accepts_episode500(self):
        reference = _panel(mae=10.0, relocations=2)
        failed = _panel(mae=12.5, relocations=1)
        passed = _panel(mae=11.0, relocations=1)
        result = _sequential_guard(
            model_seed=1,
            selection=_selection(450, (425, 500)),
            reference_rows=reference,
            available={425: failed, 500: passed},
            provenance={425: {"artifact": "425"}, 500: {"artifact": "500"}},
        )
        self.assertEqual(result["chosen_episode"], 500)
        self.assertFalse(result["sequential_panel_B_trials"][0]["accepted"])
        self.assertTrue(result["sequential_panel_B_trials"][1]["accepted"])
        self.assertGreater(
            result["sequential_panel_B_trials"][0]["panel_B_mae_cost"],
            GUARD_MAE_MARGIN,
        )

    def test_timing_safe_candidate_without_relocation_saving_is_rejected(self):
        reference = _panel(mae=10.0, relocations=2)
        no_saving = _panel(mae=10.5, relocations=2)
        passed = _panel(mae=11.0, relocations=1)
        result = _sequential_guard(
            model_seed=1,
            selection=_selection(450, (425, 500)),
            reference_rows=reference,
            available={425: no_saving, 500: passed},
            provenance={425: {"artifact": "425"}, 500: {"artifact": "500"}},
        )
        self.assertEqual(result["chosen_episode"], 500)
        first = result["sequential_panel_B_trials"][0]
        self.assertLessEqual(first["panel_B_mae_cost"], GUARD_MAE_MARGIN)
        self.assertEqual(
            first["panel_B_relocation_saving_per_100_deliveries"], 0.0
        )
        self.assertFalse(first["accepted"])

    def test_reference_is_used_when_no_nonreference_candidate_passes(self):
        reference = _panel(mae=10.0, relocations=2)
        no_saving = _panel(mae=10.5, relocations=2)
        result = _sequential_guard(
            model_seed=0,
            selection=_selection(475, (500,)),
            reference_rows=reference,
            available={500: no_saving},
            provenance={500: {"artifact": "500"}},
        )
        self.assertEqual(result["chosen_episode"], 475)
        fallback = result["sequential_panel_B_trials"][-1]
        self.assertTrue(fallback["selected_as_mandatory_reference_fallback"])
        self.assertFalse(fallback["accepted"])

    def test_expected_seed0_and_seed2_first_candidates_pass(self):
        cases = ((0, 500, 475), (2, 375, 500))
        for seed, reference_episode, candidate_episode in cases:
            with self.subTest(seed=seed):
                result = _sequential_guard(
                    model_seed=seed,
                    selection=_selection(reference_episode, (candidate_episode,)),
                    reference_rows=_panel(mae=10.0, relocations=3),
                    available={
                        candidate_episode: _panel(mae=11.5, relocations=1)
                    },
                    provenance={candidate_episode: {"artifact": "frozen"}},
                )
                self.assertEqual(
                    result["chosen_episode"], EXPECTED_GUARDED_EPISODES[seed]
                )

    def test_strict_full_gate_rejects_incomplete_candidate(self):
        rows = _panel(mae=10.0, relocations=1)
        self.assertTrue(_strict_full(rows))
        rows[0]["completion_rate"] = 0.0
        self.assertFalse(_strict_full(rows))

    def test_guarded_relabel_cannot_be_mistaken_for_deployment_evidence(self):
        rows = _relabel_guarded(_panel(mae=10.0, relocations=1), 0, 475)
        self.assertTrue(all(row["panel_B_selected"] for row in rows))
        self.assertTrue(all(row["policy_group"] == GUARDED_GROUP for row in rows))
        self.assertTrue(
            all(row["deployment_checkpoint_eligible"] is False for row in rows)
        )
        self.assertTrue(
            all(row["performance_claim_authorized"] is False for row in rows)
        )

    def test_report_contains_seed_aware_uncertainty_and_zero_new_executions(self):
        guarded = []
        source = {ENHANCED_GA_METHOD: _panel(mae=11.0, relocations=2)}
        for seed, episode in EXPECTED_GUARDED_EPISODES.items():
            chosen = _panel(mae=10.5 + seed * 0.1, relocations=1)
            reference = _panel(mae=10.0, relocations=2)
            source[SOURCE_SELECTED_METHODS[seed]] = reference
            guarded.append(
                {
                    "model_seed": seed,
                    "reference_episode": {0: 500, 1: 450, 2: 375}[seed],
                    "chosen_episode": episode,
                    "chosen_rows": chosen,
                    "validation_A_relocation_rank": (),
                    "sequential_panel_B_trials": (),
                    "reference_summary_panel_B": {},
                    "chosen_summary_panel_B": {},
                    "guard_rule": "synthetic",
                }
            )
        report = _build_report(guarded, source, {"hash": "frozen"}, {})
        self.assertEqual(report["new_policy_executions"], 0)
        self.assertEqual(report["new_baseline_executions"], 0)
        self.assertFalse(report["performance_claim_authorized"])
        metric = report["aggregate_contrasts"]["guarded_vs_selected_best"][
            "metrics"
        ]["mae_cost"]
        self.assertEqual(metric["n_training_seeds"], 3)
        self.assertEqual(metric["t_degrees_of_freedom"], 2)
        self.assertIn("conditional_30_instance_bootstrap_95_ci", metric)
        self.assertIn("crossed_seed_instance_bootstrap_sensitivity_95_ci", metric)


if __name__ == "__main__":
    unittest.main()
