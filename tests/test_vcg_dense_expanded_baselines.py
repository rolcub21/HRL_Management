import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from compare_vcg_dense_expanded_baselines import (
    DURATION_AWARE_GA_METHOD,
    NEW_ONLINE_METHODS,
    NEW_ONLINE_METHOD_TO_SOURCE,
    OFFLINE_GA_REFERENCE,
    OPERATIONAL_GA_METHOD,
    OUTCOME_EQUIVALENCE_FIELDS,
    PROTOCOL,
    SOURCE_EXPECTED_METHOD_COUNTS,
    SOURCE_PROTOCOL,
    VCG_FINAL_DIAGNOSTIC_GROUP,
    VCG_SELECTED_GROUP,
    _load_or_execute_new_run,
    _rank_summaries,
    _validate_source_rows,
    build_expanded_report,
)
from compare_vcg_dense_pareto import EVALUATION_SEEDS
from PSLAP.track_a import (
    TRACK_A_GA_OFFLINE,
    TRACK_A_GA_ROLLING_DURATION_AWARE,
    TRACK_A_GA_ROLLING_OPERATIONAL,
)


def _instance_manifest():
    return {
        "instances": {
            str(seed): {
                "instance_id": f"instance-{seed}",
                "schedule_id": f"schedule-{seed}",
            }
            for seed in EVALUATION_SEEDS
        }
    }


def _row(method, seed, *, strict=1.0):
    if "selected_best" in method:
        policy_group = "selected_best"
    elif "episode500_final" in method:
        policy_group = "episode500_final_diagnostic"
    else:
        policy_group = "baseline"
    model_seed = None
    if method.startswith("vcg_dense_seed"):
        model_seed = int(method[len("vcg_dense_seed")])
    return {
        "protocol": SOURCE_PROTOCOL,
        "method": method,
        "method_id": method,
        "policy_group": policy_group,
        "model_seed": model_seed,
        "instance_seed": seed,
        "instance_id": f"instance-{seed}",
        "schedule_id": f"schedule-{seed}",
        "strict_method_success": strict,
        "completion_rate": strict,
        "delivery_count": 8,
        "method_failure_reason": None if strict else "synthetic_failure",
        "dense_objective_return": 100.0,
        "legacy_rescored_return": 120.0,
        "mean_absolute_error": 10.0,
        "first_two_mean_absolute_error": 12.0,
        "positions_three_plus_mean_absolute_error": 9.0,
        "mean_tardiness": 3.0,
        "mean_earliness": 7.0,
        "within_target_window_rate": 0.9,
        "relocations_per_100_deliveries": 20.0,
        "steps": 175,
        "planning_seconds": 0.1,
    }


def _source_rows():
    return [
        _row(method, seed)
        for method in SOURCE_EXPECTED_METHOD_COUNTS
        for seed in EVALUATION_SEEDS
    ]


def _expanded_rows(*, fail_dynamic=False):
    rows = _source_rows()
    if fail_dynamic:
        target = next(
            row
            for row in rows
            if row["method_id"] == "duration_aware_dynamic_pslap"
        )
        target["strict_method_success"] = 0.0
        target["completion_rate"] = 0.5
        target["method_failure_reason"] = "synthetic_failure"
    for method in NEW_ONLINE_METHODS:
        for seed in EVALUATION_SEEDS:
            row = _row(method, seed)
            row["protocol"] = PROTOCOL
            if method == DURATION_AWARE_GA_METHOD:
                row["mean_absolute_error"] = 8.0
                row["relocations_per_100_deliveries"] = 15.0
            else:
                row["mean_absolute_error"] = 9.0
                row["relocations_per_100_deliveries"] = 12.0
            rows.append(row)
    return rows


class ExpandedBaselineProtocolTests(unittest.TestCase):
    def test_only_missing_online_variants_are_added_and_offline_is_separate(self):
        self.assertEqual(
            NEW_ONLINE_METHOD_TO_SOURCE,
            {
                DURATION_AWARE_GA_METHOD: TRACK_A_GA_ROLLING_DURATION_AWARE,
                OPERATIONAL_GA_METHOD: TRACK_A_GA_ROLLING_OPERATIONAL,
            },
        )
        self.assertNotIn(OFFLINE_GA_REFERENCE, NEW_ONLINE_METHODS)
        self.assertEqual(TRACK_A_GA_OFFLINE, "pslap_ga_2009_offline")

    def test_source_grid_validation_accepts_exact_grid_and_rejects_loss(self):
        rows = _source_rows()
        manifest = _instance_manifest()
        _validate_source_rows(rows, manifest)
        with self.assertRaisesRegex(ValueError, "row count mismatch"):
            _validate_source_rows(rows[:-1], manifest)

    def test_source_grid_validation_rejects_provenance_tampering(self):
        rows = _source_rows()
        rows[0]["schedule_id"] = "tampered"
        with self.assertRaisesRegex(ValueError, "schedule id mismatch"):
            _validate_source_rows(rows, _instance_manifest())

    def test_failure_is_retained_but_excluded_as_a_whole_method(self):
        report = build_expanded_report(_expanded_rows(fail_dynamic=True))
        summaries = {
            item["method_group"]: item for item in report["method_summaries"]
        }
        dynamic = summaries["duration_aware_dynamic_pslap"]
        self.assertFalse(dynamic["whole_method_numeric_eligible"])
        self.assertEqual(len(dynamic["method_failures"]), 1)
        self.assertIsNone(dynamic["mean_mean_absolute_error"])
        self.assertIn(
            "duration_aware_dynamic_pslap",
            report["safety_ineligible_methods"],
        )
        every_ranked = {
            item["method_group"]
            for ranking in report["rankings"].values()
            for item in ranking
        }
        self.assertNotIn("duration_aware_dynamic_pslap", every_ranked)

    def test_report_has_no_scalar_overall_rank_and_keeps_diagnostic_separate(self):
        report = build_expanded_report(_expanded_rows())
        self.assertIn(VCG_SELECTED_GROUP, report["online_primary_groups"])
        self.assertNotIn(
            VCG_FINAL_DIAGNOSTIC_GROUP, report["online_primary_groups"]
        )
        self.assertEqual(
            report["diagnostic_groups"], [VCG_FINAL_DIAGNOSTIC_GROUP]
        )
        self.assertFalse(report["offline_reference"]["executed"])
        self.assertFalse(
            report["offline_reference"]["included_in_online_ranking"]
        )
        self.assertNotIn("overall_rank", report)
        pending = {
            item["method_group"]
            for item in report["pending_matched_online_baselines"]
        }
        self.assertIn("reg_selector_v5", pending)
        self.assertIn("kim2020_a3c_spatial_adapted__stochastic", pending)

    def test_metric_ranking_uses_competition_ranks_for_exact_ties(self):
        summaries = []
        for name, dense in (("a", 3.0), ("b", 2.0), ("c", 2.0), ("d", 1.0)):
            summaries.append(
                {
                    "method_group": name,
                    "category": "deterministic_online_primary",
                    "whole_method_numeric_eligible": True,
                    "mean_dense_objective_return": dense,
                    "mean_mean_absolute_error": 1.0,
                    "mean_mean_tardiness": 1.0,
                    "mean_within_target_window_rate": 1.0,
                    "mean_relocations_per_100_deliveries": 1.0,
                    "mean_steps": 1.0,
                }
            )
        ranked = _rank_summaries(summaries)[
            "dense_objective_return_higher_is_better"
        ]
        self.assertEqual([item["rank"] for item in ranked], [1, 2, 2, 4])

    def test_reports_exact_aggregate_and_per_instance_outcome_equivalence(self):
        rows = _expanded_rows()
        for seed in EVALUATION_SEEDS:
            reference = next(
                row
                for row in rows
                if row["method_id"]
                == "duration_aware_enhanced_complete_rolling_ga"
                and row["instance_seed"] == seed
            )
            candidate = next(
                row
                for row in rows
                if row["method_id"] == DURATION_AWARE_GA_METHOD
                and row["instance_seed"] == seed
            )
            for field in OUTCOME_EQUIVALENCE_FIELDS:
                candidate[field] = copy.deepcopy(reference.get(field))
            # Runtime is implementation cost, not a realized control outcome.
            candidate["planning_seconds"] = reference["planning_seconds"] + 1.0
        report = build_expanded_report(rows)
        pair = next(
            item
            for item in report["deterministic_outcome_equivalence"]["pairs"]
            if {item["left"], item["right"]}
            == {
                "duration_aware_enhanced_complete_rolling_ga",
                DURATION_AWARE_GA_METHOD,
            }
        )
        self.assertTrue(pair["reported_aggregates_exactly_equal"])
        self.assertTrue(pair["outcome_identical_on_all_30_instances"])
        self.assertTrue(pair["exact_aggregate_and_row_equivalent"])
        self.assertEqual(pair["outcome_identical_instance_count"], 30)

        candidate = next(
            row
            for row in rows
            if row["method_id"] == DURATION_AWARE_GA_METHOD
        )
        candidate["steps"] += 1
        report = build_expanded_report(rows)
        pair = next(
            item
            for item in report["deterministic_outcome_equivalence"]["pairs"]
            if {item["left"], item["right"]}
            == {
                "duration_aware_enhanced_complete_rolling_ga",
                DURATION_AWARE_GA_METHOD,
            }
        )
        self.assertFalse(pair["outcome_identical_on_all_30_instances"])
        self.assertFalse(pair["reported_aggregates_exactly_equal"])
        self.assertEqual(pair["per_instance_differences"][0]["differing_fields"], ["steps"])

    def test_unexpected_delivery_count_excludes_whole_method_before_ranking(self):
        rows = _expanded_rows()
        target = next(
            row for row in rows if row["method_id"] == OPERATIONAL_GA_METHOD
        )
        target["delivery_count"] = 7
        report = build_expanded_report(rows)
        summary = next(
            item
            for item in report["method_summaries"]
            if item["method_group"] == OPERATIONAL_GA_METHOD
        )
        self.assertFalse(summary["whole_method_numeric_eligible"])
        self.assertTrue(
            any(
                issue.startswith("unexpected_delivery_count:7")
                for issue in summary["row_safety_issues"][0]["issues"]
            )
        )
        self.assertTrue(report["numeric_ranking_invariants"]["passed"])
        ranked = {
            item["method_group"]
            for ranking in report["rankings"].values()
            for item in ranking
        }
        self.assertNotIn(OPERATIONAL_GA_METHOD, ranked)

    def test_nonfinite_or_failure_reason_on_strict_full_row_fails_closed(self):
        rows = _expanded_rows()
        target = next(
            row for row in rows if row["method_id"] == DURATION_AWARE_GA_METHOD
        )
        target["mean_absolute_error"] = float("nan")
        target["method_failure_reason"] = "hidden_failure"
        report = build_expanded_report(rows)
        summary = next(
            item
            for item in report["method_summaries"]
            if item["method_group"] == DURATION_AWARE_GA_METHOD
        )
        self.assertFalse(summary["whole_method_numeric_eligible"])
        issues = summary["row_safety_issues"][0]["issues"]
        self.assertIn("missing_or_nonfinite:mean_absolute_error", issues)
        self.assertIn("method_failure_reason_present", issues)

    def test_unbalanced_learned_seed_grid_fails_closed(self):
        rows = _expanded_rows()
        selected = [
            row for row in rows if row["policy_group"] == "selected_best"
        ]
        removed = next(row for row in selected if row["model_seed"] == 2)
        rows.remove(removed)
        duplicate = copy.deepcopy(
            next(row for row in selected if row["model_seed"] == 1)
        )
        rows.append(duplicate)
        report = build_expanded_report(rows)
        summary = next(
            item
            for item in report["method_summaries"]
            if item["method_group"] == VCG_SELECTED_GROUP
        )
        self.assertFalse(summary["whole_method_numeric_eligible"])
        self.assertTrue(summary["grid_issues"])

    def test_transactional_ledger_executes_once_and_rejects_drift(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            calls = []
            contract = {
                "instance_id": "instance-80000",
                "schedule_id": "schedule-80000",
                "frozen": True,
            }

            def execute():
                calls.append(1)
                return {
                    "method_id": DURATION_AWARE_GA_METHOD,
                    "instance_seed": 80000,
                    "instance_id": "instance-80000",
                    "schedule_id": "schedule-80000",
                }

            first, _ = _load_or_execute_new_run(
                output_dir=output,
                method=DURATION_AWARE_GA_METHOD,
                seed=80000,
                input_contract=contract,
                execute=execute,
            )
            second, _ = _load_or_execute_new_run(
                output_dir=output,
                method=DURATION_AWARE_GA_METHOD,
                seed=80000,
                input_contract=contract,
                execute=execute,
            )
            self.assertEqual(first, second)
            self.assertEqual(calls, [1])

            changed = copy.deepcopy(contract)
            changed["frozen"] = False
            with self.assertRaisesRegex(ValueError, "input contract mismatch"):
                _load_or_execute_new_run(
                    output_dir=output,
                    method=DURATION_AWARE_GA_METHOD,
                    seed=80000,
                    input_contract=changed,
                    execute=execute,
                )


if __name__ == "__main__":
    unittest.main()
