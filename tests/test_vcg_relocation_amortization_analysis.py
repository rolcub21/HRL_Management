import copy
import math
import unittest

from vcg_relocation_amortization_analysis import (
    D_Q_BRANCH,
    D_SELF_BRANCH,
    R_BRANCH,
    W_BRANCH,
    RelocationAmortizationAnalysisError,
    build_relocation_amortization_report,
    validate_branch_rows,
)


def _branch_row(
    *,
    seed,
    instance,
    event,
    branch,
    dense_return,
    errors=None,
    future_rehandles=None,
    total_steps=None,
):
    errors = errors or ({"A": 1.0, "B": 3.0} if branch == R_BRANCH else {"A": 5.0, "B": 7.0})
    if future_rehandles is None:
        future_rehandles = 1 if branch == R_BRANCH else 4
    if total_steps is None:
        total_steps = 90 if branch == R_BRANCH else 100
    action = {
        R_BRANCH: "RECONFIGURE",
        D_Q_BRANCH: "DELIVER",
        D_SELF_BRANCH: "DELIVER",
        W_BRANCH: "DEFER",
    }[branch]
    initial = 1 if branch == R_BRANCH else 0
    return {
        "event_id": f"seed{seed}-instance{instance}-event{event}",
        "method_id": f"vcg-seed{seed}",
        "model_seed": seed,
        "instance_seed": 80_000 + instance,
        "instance_id": f"instance-{instance}",
        "schedule_id": f"schedule-{instance}",
        "event_index": event,
        "decision_index": 10 + event,
        "branch_id": branch,
        "eligible": True,
        "initial_action_type": action,
        "initial_physical_rehandles": initial,
        "dense_return": float(dense_return),
        "discounted_dense_return": float(dense_return) - 5.0,
        "post_initial_physical_rehandles": future_rehandles,
        "total_physical_rehandles": initial + future_rehandles,
        "steps_from_boundary": total_steps,
        "strict_method_success": True,
        "complete": True,
        "terminal": True,
        "remaining_delivery_count": 2,
        "completed_remaining_delivery_count": 2,
        "expected_remaining_labels": ["A", "B"],
        "signed_delivery_errors": dict(errors),
        "guard_activations": 0 if branch == R_BRANCH else 2,
        "guard_forced_decisions": 0 if branch == R_BRANCH else 1,
        "exact_safe_execution": True,
        "macro_failures": 0,
        "illegal_drops": 0,
        "method_failure_reason": None,
        "target_window": 4.0,
        "checkpoint_sha256": f"digest-seed{seed}",
    }


def _event_rows(seed, instance, event, *, r_gain=10.0, include_optional=False):
    rows = [
        _branch_row(
            seed=seed,
            instance=instance,
            event=event,
            branch=R_BRANCH,
            dense_return=40.0 + r_gain,
        ),
        _branch_row(
            seed=seed,
            instance=instance,
            event=event,
            branch=D_Q_BRANCH,
            dense_return=40.0,
        ),
    ]
    if include_optional:
        rows.extend(
            [
                _branch_row(
                    seed=seed,
                    instance=instance,
                    event=event,
                    branch=D_SELF_BRANCH,
                    dense_return=39.0,
                ),
                _branch_row(
                    seed=seed,
                    instance=instance,
                    event=event,
                    branch=W_BRANCH,
                    dense_return=38.0,
                ),
            ]
        )
    return rows


def _panel_rows():
    rows = []
    # Seed 0 deliberately has two events in one instance and one in another.
    # Its primary dense estimate is mean(mean(0,100), mean(0)) = 25, whereas
    # its event-weighted secondary estimate is 100/3.
    rows += _event_rows(0, 0, 0, r_gain=0.0, include_optional=True)
    rows += _event_rows(0, 0, 1, r_gain=100.0)
    rows += _event_rows(0, 1, 0, r_gain=0.0)
    for seed in (1, 2):
        rows += _event_rows(seed, 0, 0, r_gain=10.0)
        rows += _event_rows(seed, 1, 0, r_gain=10.0)
    return rows


class RelocationAmortizationAnalysisTests(unittest.TestCase):
    def test_report_computes_positive_favors_R_deltas_and_mechanisms(self):
        report = build_relocation_amortization_report(
            _panel_rows(),
            target_window=4.0,
            expected_model_seeds=(0, 1, 2),
            bootstrap_samples=100,
            rng_seed=17,
        )

        self.assertEqual(report["validation"]["event_count"], 7)
        self.assertEqual(report["validation"]["branch_counts"][R_BRANCH], 7)
        self.assertEqual(report["validation"]["branch_counts"][D_Q_BRANCH], 7)
        self.assertEqual(report["secondary_R_vs_D_SELF"]["event_count"], 1)
        self.assertEqual(report["secondary_R_vs_W"]["event_count"], 1)

        event = report["primary_R_vs_D_Q"]["events"][0]
        delta = event["deltas_positive_favors_R"]
        self.assertEqual(delta["post_initial_physical_rehandles_avoided"], 3.0)
        self.assertEqual(delta["total_physical_rehandles_avoided"], 2.0)
        self.assertEqual(delta["steps_avoided"], 10.0)
        self.assertEqual(delta["sum_absolute_timing_error_reduced"], 8.0)
        self.assertEqual(delta["mean_absolute_timing_error_reduced"], 4.0)
        self.assertEqual(delta["sum_tardiness_reduced"], 8.0)
        self.assertEqual(delta["within_window_deliveries_gained"], 2.0)
        self.assertEqual(delta["within_window_rate_gain"], 1.0)
        self.assertEqual(delta["guard_activations_avoided"], 2.0)
        self.assertEqual(delta["guard_forced_decisions_avoided"], 1.0)
        self.assertTrue(event["mechanism_flags"]["avoids_any_future_rehandle"])
        self.assertTrue(
            event["mechanism_flags"]["immediate_rehandle_cost_amortized"]
        )
        self.assertTrue(event["mechanism_flags"]["strict_net_physical_benefit"])
        # The first event ties on return, so it is still weakly dominated by R
        # on all operational axes and strictly better on several.
        self.assertEqual(event["pareto_verdict"], "R_dominates_comparator")
        self.assertEqual(event["value_verdict"], "return_neutral")
        self.assertFalse(event["operationally_productive"])
        positive = next(
            item
            for item in report["primary_R_vs_D_Q"]["events"]
            if item["deltas_positive_favors_R"]["dense_return_gain"] > 0.0
        )
        self.assertEqual(positive["value_verdict"], "return_beneficial")
        self.assertTrue(positive["operationally_productive"])

    def test_primary_aggregation_is_instance_then_equal_seed_not_event_weighted(self):
        report = build_relocation_amortization_report(
            _panel_rows(), target_window=4.0, bootstrap_samples=40, rng_seed=3
        )
        primary = report["primary_R_vs_D_Q"]["instance_weighted_primary"]
        seed0 = primary["per_model_seed"]["0"]
        self.assertEqual(seed0["eligible_instance_count"], 2)
        self.assertEqual(seed0["event_count"], 3)
        self.assertAlmostEqual(
            seed0["deltas_positive_favors_R"]["dense_return_gain"], 25.0
        )
        self.assertAlmostEqual(
            primary["equal_model_seed_mean"]["deltas_positive_favors_R"][
                "dense_return_gain"
            ],
            15.0,
        )
        self.assertAlmostEqual(
            primary["equal_model_seed_mean"]["operationally_productive_fraction"],
            0.75,
        )

        secondary = report["primary_R_vs_D_Q"]["event_weighted_secondary"]
        self.assertAlmostEqual(
            secondary["per_model_seed"]["0"]["deltas_positive_favors_R"][
                "dense_return_gain"
            ],
            100.0 / 3.0,
        )
        self.assertAlmostEqual(
            secondary["equal_model_seed_mean"]["deltas_positive_favors_R"][
                "dense_return_gain"
            ],
            (100.0 / 3.0 + 10.0 + 10.0) / 3.0,
        )
        seed_values = report["primary_R_vs_D_Q"]["three_seed_descriptive_t95"][
            "dense_return_gain"
        ]["values_by_model_seed"]
        self.assertEqual(seed_values, {"0": 25.0, "1": 10.0, "2": 10.0})

    def test_bootstrap_is_reproducible_and_keeps_models_fixed(self):
        kwargs = dict(target_window=4.0, bootstrap_samples=75, rng_seed=99)
        first = build_relocation_amortization_report(_panel_rows(), **kwargs)
        second = build_relocation_amortization_report(_panel_rows(), **kwargs)
        first_boot = first["primary_R_vs_D_Q"][
            "conditional_instance_cluster_bootstrap"
        ]
        second_boot = second["primary_R_vs_D_Q"][
            "conditional_instance_cluster_bootstrap"
        ]
        self.assertEqual(first_boot, second_boot)
        self.assertEqual(first_boot["conditional_on_model_seed_count"], 3)
        self.assertIn("not resampled", first_boot["resampling"])
        self.assertAlmostEqual(
            first_boot["metrics"]["dense_return_gain"]["estimate"], 15.0
        )

    def test_return_harmful_verdict_prevents_productive_label_despite_mechanism_gain(self):
        rows = _panel_rows()
        factual = next(
            row
            for row in rows
            if row["event_id"] == "seed0-instance0-event0"
            and row["branch_id"] == R_BRANCH
        )
        factual["dense_return"] = 35.0
        factual["discounted_dense_return"] = 30.0
        report = build_relocation_amortization_report(
            rows, target_window=4.0, bootstrap_samples=10
        )
        event = next(
            item
            for item in report["primary_R_vs_D_Q"]["events"]
            if item["event_id"] == factual["event_id"]
        )
        self.assertEqual(event["value_verdict"], "return_harmful")
        self.assertTrue(event["mechanism_flags"]["strict_net_physical_benefit"])
        self.assertFalse(event["operationally_productive"])

    def test_runner_aliases_are_accepted_but_conflicts_fail(self):
        rows = _panel_rows()
        for row in rows:
            row["branch"] = row.pop("branch_id")
            row["total_dense_return_raw"] = row.pop("dense_return")
            row["total_dense_discounted_return"] = row.pop(
                "discounted_dense_return"
            )
            row["initial_macro_physical_relocations"] = row.pop(
                "initial_physical_rehandles"
            )
            row["future_only_physical_relocations"] = row.pop(
                "post_initial_physical_rehandles"
            )
            row["total_physical_relocations"] = row.pop(
                "total_physical_rehandles"
            )
            row["total_steps"] = row.pop("steps_from_boundary")
            row["success"] = row.pop("complete")
            row["labeled_errors"] = row.pop("signed_delivery_errors")
            row["guard_activations_delta"] = row.pop("guard_activations")
            row["guard_forced_decisions_delta"] = row.pop(
                "guard_forced_decisions"
            )
            row["all_executed_candidates_exact_safe"] = row.pop(
                "exact_safe_execution"
            )
        audit = validate_branch_rows(rows, target_window=4.0)
        self.assertEqual(audit["event_count"], 7)

        conflict = _panel_rows()
        conflict[0]["branch"] = D_Q_BRANCH
        with self.assertRaisesRegex(
            RelocationAmortizationAnalysisError, "conflicting aliases"
        ):
            validate_branch_rows(conflict, target_window=4.0)

    def test_validation_fails_closed_on_pair_safety_and_timing_violations(self):
        cases = {}

        missing_pair = _panel_rows()
        missing_pair = [
            row
            for row in missing_pair
            if not (
                row["event_id"] == "seed0-instance0-event0"
                and row["branch_id"] == D_Q_BRANCH
            )
        ]
        cases["must contain exactly one R"] = missing_pair

        incomplete_factual = _panel_rows()
        incomplete_factual[0]["complete"] = False
        incomplete_factual[0]["strict_method_success"] = False
        cases["factual R branch is not strict"] = incomplete_factual

        nonfinite = _panel_rows()
        nonfinite[0]["signed_delivery_errors"]["A"] = math.nan
        cases["must be finite"] = nonfinite

        missing_label = _panel_rows()
        missing_label[0]["signed_delivery_errors"].pop("B")
        cases["completed_remaining_delivery_count"] = missing_label

        different_expected = _panel_rows()
        different_expected[1]["expected_remaining_labels"] = ["B", "A"]
        cases["identical expected labels"] = different_expected

        bad_decomposition = _panel_rows()
        bad_decomposition[0]["total_physical_rehandles"] += 1
        cases["decomposition"] = bad_decomposition

        unsafe = _panel_rows()
        unsafe[0]["exact_safe_execution"] = False
        cases["factual R branch is not strict"] = unsafe

        for expected, rows in cases.items():
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(
                    RelocationAmortizationAnalysisError, expected
                ):
                    validate_branch_rows(rows, target_window=4.0)

    def test_failed_comparator_is_retained_in_safety_ledger_not_survivor_filtered(self):
        rows = _panel_rows()
        comparator = next(
            row
            for row in rows
            if row["event_id"] == "seed0-instance0-event0"
            and row["branch_id"] == D_Q_BRANCH
        )
        comparator.update(
            {
                "strict_method_success": False,
                "complete": False,
                "terminal": False,
                "method_failure_reason": "episode_step_limit",
                "expected_remaining_labels": ["A", "B"],
                "signed_delivery_errors": {"A": 5.0},
                "completed_remaining_delivery_count": 1,
            }
        )
        report = build_relocation_amortization_report(
            rows, target_window=4.0, bootstrap_samples=20
        )
        primary = report["primary_R_vs_D_Q"]
        self.assertEqual(primary["event_count"], 7)
        self.assertEqual(primary["numeric_complete_pair_count"], 6)
        self.assertEqual(primary["numeric_complete_pair_denominator"], 7)
        ledger = primary["safety_ledger"]
        self.assertEqual(ledger["R_safety_beneficial_count"], 1)
        self.assertEqual(ledger["numeric_complete_pair_numerator"], 6)
        failed = next(
            event
            for event in primary["events"]
            if event["event_id"] == "seed0-instance0-event0"
        )
        self.assertFalse(failed["numeric_comparison_eligible"])
        self.assertEqual(failed["safety_verdict"], "R_safety_beneficial")
        self.assertEqual(failed["value_verdict"], "safety_beneficial")
        self.assertFalse(failed["operationally_productive"])
        self.assertIsNone(failed["pareto_verdict"])
        self.assertTrue(
            all(
                value is None
                for value in failed["deltas_positive_favors_R"].values()
            )
        )
        self.assertEqual(
            ledger["value_verdict_counts"]["safety_beneficial"], 1
        )

    def test_requires_exactly_three_models_and_valid_bootstrap_configuration(self):
        two_seed_rows = [row for row in _panel_rows() if row["model_seed"] != 2]
        with self.assertRaisesRegex(
            RelocationAmortizationAnalysisError, "exactly three model seeds"
        ):
            validate_branch_rows(two_seed_rows, target_window=4.0)
        smoke = validate_branch_rows(
            [row for row in _panel_rows() if row["model_seed"] == 0],
            target_window=4.0,
            allow_partial_smoke=True,
        )
        self.assertTrue(smoke["partial_smoke_validation_only"])
        self.assertFalse(smoke["full_three_model_analysis_authorized"])

        with self.assertRaisesRegex(
            RelocationAmortizationAnalysisError, "positive integer"
        ):
            build_relocation_amortization_report(
                _panel_rows(), target_window=4.0, bootstrap_samples=0
            )


if __name__ == "__main__":
    unittest.main()
