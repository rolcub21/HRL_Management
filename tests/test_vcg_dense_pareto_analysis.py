import copy
import math
import unittest

from vcg_dense_pareto_analysis import (
    BASELINE,
    FINAL_DIAGNOSTIC,
    RELOCATION_METRIC,
    SELECTED_BEST,
    ParetoAnalysisError,
    build_pareto_report,
    pareto_relation,
    summarize_seed_values,
    validate_normalized_rows,
)


def _row(
    method_id,
    policy_group,
    model_seed,
    instance_index,
    *,
    mae,
    relocations,
    delivery_count=8,
    first_two=None,
    later=None,
    tardiness=None,
    earliness=None,
    window=0.8,
):
    first_two = mae if first_two is None else first_two
    later = mae if later is None else later
    tardiness = mae / 4.0 if tardiness is None else tardiness
    earliness = mae / 3.0 if earliness is None else earliness
    return {
        "method_id": method_id,
        "policy_group": policy_group,
        "model_seed": model_seed,
        "instance_id": f"instance-{instance_index}",
        "schedule_id": f"schedule-{instance_index}",
        "strict_method_success": True,
        "completion_rate": 1.0,
        "delivery_count": delivery_count,
        "mean_absolute_error": float(mae),
        "first_two_mean_absolute_error": float(first_two),
        "positions_three_plus_mean_absolute_error": float(later),
        "mean_tardiness": float(tardiness),
        "mean_earliness": float(earliness),
        "within_target_window_rate": float(window),
        "steps": 100 + instance_index,
        "dense_rescored_return": 200.0 - float(mae),
        "legacy_rescored_return": 220.0 - float(mae),
        "relocations": relocations,
        # Extra authenticated provenance is intentionally accepted.  The
        # statistical core does not reinterpret it.
        "checkpoint_sha256": f"digest-{method_id}",
    }


def synthetic_rows(instance_count=4):
    rows = []
    for seed in (0, 1, 2):
        for instance in range(instance_count):
            selected_mae = 10.0 + seed + 0.5 * instance
            rows.append(
                _row(
                    f"selected-s{seed}",
                    SELECTED_BEST,
                    seed,
                    instance,
                    mae=selected_mae,
                    relocations=2,
                    first_two=selected_mae + 1.0,
                    later=selected_mae - 1.0,
                    tardiness=2.0 + 0.1 * instance,
                    earliness=3.0 + 0.1 * seed,
                    window=0.8,
                )
            )
            rows.append(
                _row(
                    f"final-s{seed}",
                    FINAL_DIAGNOSTIC,
                    seed,
                    instance,
                    mae=selected_mae + 1.0,
                    relocations=1,
                    first_two=selected_mae + 3.0,
                    later=selected_mae - 0.5,
                    tardiness=2.25 + 0.1 * instance,
                    earliness=3.75 + 0.1 * seed,
                    window=0.7,
                )
            )

    for instance in range(instance_count):
        # This baseline is worse on both primary axes than either learned
        # group, making the point and every instance relation unambiguous.
        rows.append(
            _row(
                "baseline-dominated",
                BASELINE,
                None,
                instance,
                mae=20.0 + instance,
                relocations=3,
                first_two=22.0 + instance,
                later=19.0 + instance,
                tardiness=5.0,
                earliness=6.0,
                window=0.3,
            )
        )
        # Better relocation but worse timing than the learned arms.
        rows.append(
            _row(
                "baseline-tradeoff",
                BASELINE,
                None,
                instance,
                mae=5.0 + 0.1 * instance,
                relocations=0,
                first_two=6.0,
                later=5.0,
                tardiness=1.0,
                earliness=2.0,
                window=0.95,
            )
        )
    return rows


class VcgDenseParetoAnalysisTests(unittest.TestCase):
    def test_complete_report_uses_seed_aware_estimands_and_diagnostic_labels(self):
        report = build_pareto_report(
            synthetic_rows(),
            mae_noninferiority_margin=1.5,
            bootstrap_samples=200,
            rng_seed=17,
        )

        self.assertEqual(report["validation"]["training_seeds"], [0, 1, 2])
        self.assertEqual(report["validation"]["instance_count"], 4)
        self.assertEqual(report["validation"]["row_count"], 32)
        self.assertEqual(len(report["method_points"]), 8)

        selected = report["policy_group_points"][SELECTED_BEST]
        final = report["policy_group_points"][FINAL_DIAGNOSTIC]
        self.assertAlmostEqual(selected["point"]["mean_absolute_error"], 11.75)
        self.assertAlmostEqual(final["point"]["mean_absolute_error"], 12.75)
        self.assertAlmostEqual(selected["point"][RELOCATION_METRIC], 25.0)
        self.assertAlmostEqual(final["point"][RELOCATION_METRIC], 12.5)
        self.assertFalse(selected["diagnostic_only"])
        self.assertTrue(final["diagnostic_only"])
        self.assertFalse(final["deployment_claim_eligible"])

        capacity = report["best_vs_final"]
        self.assertEqual(capacity["point_pareto_relation"], "tradeoff")
        self.assertAlmostEqual(capacity["point_estimate"]["mae_cost"], 1.0)
        self.assertAlmostEqual(
            capacity["point_estimate"]["relocation_saving_per_100_deliveries"],
            12.5,
        )
        self.assertTrue(capacity["mae_noninferiority"]["point_estimate_pass"])
        self.assertTrue(
            capacity["mae_noninferiority"]["seed_t95_upper_bound_pass"]
        )
        self.assertEqual(capacity["capacity_signal"]["supporting_seed_count"], 3)

        mae_seed = capacity["seed_level_t95"]["mae_cost"]
        self.assertEqual(mae_seed["degrees_of_freedom"], 2)
        self.assertEqual(mae_seed["t95_interval"], [1.0, 1.0])
        conditional = capacity["conditional_paired_instance_bootstrap"]
        self.assertEqual(
            conditional["metrics"]["mae_cost"]["percentile_95_interval"],
            [1.0, 1.0],
        )
        self.assertEqual(
            conditional["joint_pareto_relation_fractions"], {"tradeoff": 1.0}
        )
        self.assertEqual(conditional["capacity_margin_fraction"], 1.0)

        deployment_frontier = report["pareto"][
            "deployment_frontier_excluding_diagnostic_final"
        ]
        self.assertNotIn(FINAL_DIAGNOSTIC, deployment_frontier)
        self.assertIn(FINAL_DIAGNOSTIC, report["pareto"]["participants"])
        self.assertTrue(
            report["interpretation_limits"]["final_weights_are_diagnostic_only"]
        )

    def test_baseline_contrasts_and_actual_instance_quadrants_are_paired(self):
        report = build_pareto_report(
            synthetic_rows(),
            mae_noninferiority_margin=1.5,
            bootstrap_samples=100,
            rng_seed=23,
        )
        selected_dominated_baseline = next(
            item
            for item in report["learned_vs_baselines"]
            if item["policy_group"] == SELECTED_BEST
            and item["baseline_method_id"] == "baseline-dominated"
        )
        self.assertEqual(
            selected_dominated_baseline["point_pareto_relation"],
            "learned_dominates_baseline",
        )
        self.assertEqual(
            selected_dominated_baseline["seed_joint_dominance_count"], 3
        )
        quadrants = selected_dominated_baseline["paired_instance_quadrants"]
        for seed in ("0", "1", "2"):
            self.assertEqual(
                quadrants["per_model_seed"][seed]["counts"][
                    "timing_better_and_relocation_better"
                ],
                4,
            )
            self.assertEqual(
                quadrants["per_model_seed"][seed]["n_paired_instances"], 4
            )

        capacity_quadrants = report["best_vs_final"][
            "paired_instance_quadrants"
        ]
        for seed in ("0", "1", "2"):
            self.assertEqual(
                capacity_quadrants["per_model_seed"][seed]["counts"][
                    "timing_worse_and_relocation_better"
                ],
                4,
            )
        self.assertEqual(
            capacity_quadrants["equal_seed_mean_fractions"][
                "timing_worse_and_relocation_better"
            ],
            1.0,
        )

    def test_fixed_rng_makes_all_bootstrap_outputs_reproducible(self):
        kwargs = {
            "mae_noninferiority_margin": 1.5,
            "bootstrap_samples": 75,
            "rng_seed": 99,
        }
        first = build_pareto_report(synthetic_rows(), **kwargs)
        second = build_pareto_report(synthetic_rows(), **kwargs)

        self.assertEqual(first, second)

    def test_relocation_point_is_ratio_of_sums_not_mean_episode_ratio(self):
        rows = synthetic_rows(instance_count=2)
        for row in rows:
            row["delivery_count"] = 4 if row["instance_id"] == "instance-0" else 8
            if row["method_id"].startswith("selected"):
                row["relocations"] = 1 if row["instance_id"] == "instance-0" else 0
            elif row["method_id"].startswith("final"):
                row["relocations"] = 0
        report = build_pareto_report(
            rows,
            mae_noninferiority_margin=2.0,
            bootstrap_samples=30,
        )

        selected = report["policy_group_points"][SELECTED_BEST]["point"]
        self.assertAlmostEqual(selected[RELOCATION_METRIC], 100.0 / 12.0)
        self.assertNotAlmostEqual(selected[RELOCATION_METRIC], 12.5)

    def test_validation_fails_closed_on_bad_or_incomplete_rows(self):
        cases = {}

        missing = synthetic_rows()
        missing[0].pop("mean_tardiness")
        cases["missing required"] = missing

        duplicate = synthetic_rows()
        duplicate.append(copy.deepcopy(duplicate[0]))
        cases["duplicate method/instance"] = duplicate

        grid = synthetic_rows()
        grid.pop(0)
        cases["incomplete paired grid"] = grid

        schedule = synthetic_rows()
        schedule[0]["schedule_id"] = "different-schedule"
        cases["maps to multiple schedules"] = schedule

        deliveries = synthetic_rows()
        deliveries[0]["delivery_count"] = 7
        cases["inconsistent delivery counts"] = deliveries

        only_two_seeds = [
            row for row in synthetic_rows() if row["model_seed"] != 2
        ]
        cases["exactly three training seeds"] = only_two_seeds

        for expected, rows in cases.items():
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ParetoAnalysisError, expected):
                    validate_normalized_rows(rows)

    def test_failed_baseline_is_wholly_excluded_but_retained_in_safety_ledger(self):
        rows = synthetic_rows(instance_count=30)
        failed = next(
            row
            for row in rows
            if row["method_id"] == "baseline-dominated"
            and row["instance_id"] == "instance-20"
        )
        failed.update(
            {
                "strict_method_success": 0.0,
                "completion_rate": 0.0,
                "delivery_count": 0,
                "mean_absolute_error": None,
                "first_two_mean_absolute_error": None,
                "positions_three_plus_mean_absolute_error": None,
                "mean_tardiness": None,
                "mean_earliness": None,
                "within_target_window_rate": None,
                "method_failure_reason": (
                    "scheduler_infeasible:not enough storage cells"
                ),
            }
        )

        report = build_pareto_report(
            rows,
            mae_noninferiority_margin=1.5,
            bootstrap_samples=25,
            rng_seed=7,
        )

        safety = report["safety_gate"]
        self.assertTrue(safety["numeric_pareto_authorized"])
        self.assertFalse(safety["complete_case_filtering_used"])
        self.assertIn("baseline-dominated", safety["excluded_method_ids"])
        ledger = safety["method_ledgers"]["baseline-dominated"]
        self.assertEqual(ledger["row_count"], 30)
        self.assertEqual(ledger["failed_or_invalid_row_count"], 1)
        self.assertEqual(ledger["finite_metric_row_count"], 29)
        self.assertTrue(ledger["whole_method_excluded"])
        self.assertEqual(
            ledger["failure_reason_counts"],
            {"scheduler_infeasible:not enough storage cells": 1},
        )
        self.assertNotIn("baseline-dominated", report["method_points"])
        self.assertNotIn("baseline-dominated", report["baseline_points"])
        self.assertNotIn("baseline-dominated", report["pareto"]["participants"])
        self.assertFalse(
            any(
                item["baseline_method_id"] == "baseline-dominated"
                for item in report["learned_vs_baselines"]
            )
        )
        self.assertIn("baseline-tradeoff", report["baseline_points"])

    def test_ineligible_learned_method_withholds_all_numeric_pareto(self):
        rows = synthetic_rows()
        failed = next(row for row in rows if row["method_id"] == "selected-s1")
        failed["mean_absolute_error"] = math.nan

        report = build_pareto_report(
            rows,
            mae_noninferiority_margin=1.5,
            bootstrap_samples=10,
        )

        self.assertFalse(report["safety_gate"]["numeric_pareto_authorized"])
        self.assertIn(
            "selected-s1", report["safety_gate"]["ineligible_learned_method_ids"]
        )
        self.assertIsNone(report["best_vs_final"])
        self.assertFalse(report["pareto"]["analysis_performed"])
        self.assertEqual(report["method_points"], {})

    def test_method_identity_and_baseline_seed_contracts_fail_closed(self):
        inconsistent = synthetic_rows()
        inconsistent[0]["method_id"] = "final-s0"
        with self.assertRaisesRegex(ParetoAnalysisError, "inconsistent policy metadata"):
            validate_normalized_rows(inconsistent)

        duplicate_arm = synthetic_rows()
        for row in duplicate_arm:
            if row["method_id"] == "selected-s1":
                row["method_id"] = "selected-s0"
        with self.assertRaisesRegex(
            ParetoAnalysisError, "inconsistent policy metadata|multiple method_ids|duplicate"
        ):
            validate_normalized_rows(duplicate_arm)

        baseline_seed = synthetic_rows()
        next(row for row in baseline_seed if row["policy_group"] == BASELINE)[
            "model_seed"
        ] = 0
        with self.assertRaisesRegex(ParetoAnalysisError, "baseline model_seed"):
            validate_normalized_rows(baseline_seed)

    def test_seed_t_summary_and_pareto_relation_are_explicit(self):
        summary = summarize_seed_values({0: 1.0, 1: 2.0, 2: 3.0})
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["sample_standard_deviation"], 1.0)
        self.assertEqual(summary["degrees_of_freedom"], 2)
        self.assertLess(summary["t95_interval"][0], 0.0)
        self.assertGreater(summary["t95_interval"][1], 4.0)

        left = {"mean_absolute_error": 2.0, RELOCATION_METRIC: 5.0}
        dominated = {"mean_absolute_error": 3.0, RELOCATION_METRIC: 6.0}
        tradeoff = {"mean_absolute_error": 1.0, RELOCATION_METRIC: 7.0}
        tied = dict(left)
        self.assertEqual(pareto_relation(left, dominated), "left_dominates_right")
        self.assertEqual(pareto_relation(left, tradeoff), "tradeoff")
        self.assertEqual(pareto_relation(left, tied), "tie")

    def test_margin_and_bootstrap_configuration_are_validated(self):
        rows = synthetic_rows()
        with self.assertRaisesRegex(ParetoAnalysisError, "non-negative"):
            build_pareto_report(
                rows, mae_noninferiority_margin=-0.1, bootstrap_samples=10
            )
        with self.assertRaisesRegex(ParetoAnalysisError, "positive integer"):
            build_pareto_report(
                rows, mae_noninferiority_margin=1.0, bootstrap_samples=0
            )


if __name__ == "__main__":
    unittest.main()
