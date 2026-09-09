import copy
import unittest

from reanalyze_vcg_contention_metrics import (
    ContentionAuditError,
    DIRECT_SELF,
    NO_DIRECT,
    OTHER_WITH_DIRECT,
    VCG_SELECTED_GROUP,
    build_report,
    preserve_baseline_contention,
    reconstruct_vcg_contention,
)


def _certificate(name):
    return {
        "status": "SAFE",
        "reason": name,
        "witness": [],
        "explored_nodes": 1,
    }


def _candidate(action, target, certificate, *, key=None, destination=None):
    return {
        "key": key or f"{action}:{target}",
        "action_type": action,
        "target_label": target,
        "source": [1, 1],
        "destination": destination or [2, 2],
        "certificate": certificate,
    }


def _frontier(epoch, current, candidates):
    return {
        "decision_epoch": epoch,
        "current_certificate": current,
        "candidates": candidates,
    }


def _three_class_fixture():
    initial = _certificate("initial")
    after_self = _certificate("after-self")
    after_other = _certificate("after-other")
    after_recovery = _certificate("after-recovery")
    unrelated = _certificate("unrelated")
    final = _certificate("terminal")
    frontiers = [
        _frontier(
            0,
            initial,
            [
                _candidate(
                    "reconfigure",
                    "B1",
                    after_self,
                    key="reconfigure:B1:2:2",
                ),
                _candidate("deliver", "B1", unrelated, key="deliver:B1:4:1"),
            ],
        ),
        _frontier(
            4,
            after_self,
            [
                _candidate(
                    "reconfigure",
                    "B2",
                    after_other,
                    key="reconfigure:B2:2:2",
                ),
                _candidate("deliver", "B1", unrelated, key="deliver:B1:4:1"),
            ],
        ),
        _frontier(
            8,
            after_other,
            [
                _candidate(
                    "reconfigure",
                    "B3",
                    after_recovery,
                    key="reconfigure:B3:2:2",
                )
            ],
        ),
        _frontier(
            12,
            after_recovery,
            [
                _candidate(
                    "deliver",
                    "B3",
                    final,
                    key="deliver:B3:4:1",
                    destination=[4, 1],
                ),
                _candidate(
                    "deliver",
                    "B3",
                    final,
                    key="deliver:B3:4:3",
                    destination=[4, 3],
                ),
                _candidate("reconfigure", "B3", unrelated),
            ],
        ),
    ]
    run = {
        "strict_method_success": 1.0,
        "completion_rate": 1.0,
        "delivery_count": 1,
        "relocations": 3,
    }
    audit = {
        "frontiers": frontiers,
        "complete_frontier_exactly_verified": True,
        "exact_verifier_authoritative": True,
        "macro_decisions": 4,
    }
    return run, audit


class ContentionMetricReanalysisTests(unittest.TestCase):
    def test_reconstruction_separates_all_three_vcg_causal_classes(self):
        run, audit = _three_class_fixture()
        result = reconstruct_vcg_contention(run, audit)
        self.assertEqual(result["total_physical_reconfigurations"], 3)
        self.assertEqual(result[DIRECT_SELF], 1)
        self.assertEqual(result[OTHER_WITH_DIRECT], 1)
        self.assertEqual(result[NO_DIRECT], 1)
        self.assertEqual(result["exact_frontier_count"], 4)
        self.assertEqual(result["frontiers_with_any_direct_delivery"], 3)
        self.assertEqual(result["frontiers_with_no_direct_delivery"], 1)
        self.assertEqual(result["reconstructed_semantic_transitions"], 4)
        self.assertEqual(result["equivalent_destination_alias_transitions"], 1)
        self.assertTrue(result["invariants_passed"])

    def test_same_successor_may_alias_delivery_exit_but_not_action_semantics(self):
        run, audit = _three_class_fixture()
        next_certificate = audit["frontiers"][1]["current_certificate"]
        audit["frontiers"][0]["candidates"].append(
            _candidate("deliver", "B1", next_certificate)
        )
        with self.assertRaisesRegex(ContentionAuditError, "ambiguous action semantics"):
            reconstruct_vcg_contention(run, audit)

    def test_reconfiguration_total_must_match_stored_physical_counter(self):
        run, audit = _three_class_fixture()
        run["relocations"] = 2
        with self.assertRaisesRegex(ContentionAuditError, "invariant failed"):
            reconstruct_vcg_contention(run, audit)

    def test_incomplete_run_cannot_use_terminal_completion_inference(self):
        run, audit = _three_class_fixture()
        run["strict_method_success"] = 0.0
        with self.assertRaisesRegex(ContentionAuditError, "strict, complete"):
            reconstruct_vcg_contention(run, audit)

    def test_baseline_counter_is_preserved_without_vcg_causal_labels(self):
        result = preserve_baseline_contention(
            {"relocations": 4, "obstructive_moves": 4},
            {"retrieve_relocations": 4},
        )
        self.assertEqual(result["retrieval_executor_obstruction_clearances"], 4)
        self.assertEqual(result["total_physical_reconfigurations"], 4)
        self.assertTrue(result["retrieval_clearances_match_source_counter"])

    def test_baseline_alias_disagreement_fails_closed(self):
        with self.assertRaisesRegex(ContentionAuditError, "does not match"):
            preserve_baseline_contention(
                {"relocations": 4, "obstructive_moves": 3},
                {"retrieve_relocations": 4},
            )

    def test_report_does_not_create_a_common_causal_count(self):
        base = {
            "method_group": VCG_SELECTED_GROUP,
            "model_seed": 0,
            "contention_metric_schema": (
                "vcg_physical_reconfiguration_causal_decomposition_v2"
            ),
            "strict_method_success": 1.0,
            "completion_rate": 1.0,
            "delivery_count": 8,
            "exact_frontier_count": 10,
            "frontiers_with_no_direct_delivery": 2,
            "physical_storage_relocations": 3,
            "target_bound_obstruction_clearances": 0,
            "standalone_reconfigurations": 3,
            "standalone_with_direct_delivery_available": 2,
            "standalone_without_direct_delivery_available": 1,
            "total_physical_reconfigurations": 3,
            DIRECT_SELF: 2,
            OTHER_WITH_DIRECT: 0,
            NO_DIRECT: 1,
            "retrieval_executor_obstruction_clearances": None,
            "relocations_per_100_deliveries": 37.5,
            "invariants_passed": True,
        }
        baseline = copy.deepcopy(base)
        baseline.update(
            {
                "method_group": "duration_aware_dynamic_pslap",
                "model_seed": None,
                "contention_metric_schema": (
                    "baseline_retrieval_executor_obstruction_clearances_v1"
                ),
                "exact_frontier_count": None,
                "frontiers_with_no_direct_delivery": None,
                "physical_storage_relocations": 2,
                "target_bound_obstruction_clearances": 2,
                "standalone_reconfigurations": 0,
                "standalone_with_direct_delivery_available": 0,
                "standalone_without_direct_delivery_available": 0,
                "total_physical_reconfigurations": 2,
                DIRECT_SELF: 0,
                OTHER_WITH_DIRECT: None,
                NO_DIRECT: None,
                "retrieval_executor_obstruction_clearances": 2,
                "relocations_per_100_deliveries": 25.0,
            }
        )
        report = build_report([base, baseline])
        self.assertFalse(
            report["cross_family_interpretation"][
                "common_causal_obstruction_count_available"
            ]
        )
        summaries = {item["method_group"]: item for item in report["method_summaries"]}
        self.assertEqual(
            summaries[VCG_SELECTED_GROUP][
                "total_physical_reconfigurations_per_100_deliveries"
            ],
            37.5,
        )
        self.assertIsNone(
            summaries[VCG_SELECTED_GROUP][
                "retrieval_executor_obstruction_clearances_per_100_deliveries"
            ]
        )
        self.assertEqual(
            summaries["duration_aware_dynamic_pslap"][
                "retrieval_executor_obstruction_clearances_per_100_deliveries"
            ],
            25.0,
        )
        self.assertEqual(
            summaries["duration_aware_dynamic_pslap"][
                "total_physical_reconfigurations_per_100_deliveries"
            ],
            25.0,
        )

    def test_failed_method_is_retained_but_primary_rate_fails_closed(self):
        row = {
            "method_group": "duration_aware_dynamic_pslap",
            "method_id": "duration_aware_dynamic_pslap",
            "model_seed": None,
            "instance_seed": 80002,
            "contention_metric_schema": (
                "baseline_retrieval_executor_obstruction_clearances_v1"
            ),
            "strict_method_success": 0.0,
            "completion_rate": 0.5,
            "delivery_count": 4,
            "method_failure_reason": "no_proposal",
            "exact_frontier_count": None,
            "frontiers_with_no_direct_delivery": None,
            "physical_storage_relocations": 2,
            "target_bound_obstruction_clearances": 2,
            "standalone_reconfigurations": 0,
            "standalone_with_direct_delivery_available": 0,
            "standalone_without_direct_delivery_available": 0,
            "total_physical_reconfigurations": 2,
            DIRECT_SELF: 0,
            OTHER_WITH_DIRECT: None,
            NO_DIRECT: None,
            "retrieval_executor_obstruction_clearances": 2,
            "relocations_per_100_deliveries": 50.0,
            "invariants_passed": True,
        }
        summary = build_report([row])["method_summaries"][0]
        self.assertFalse(summary["numeric_comparison_eligible"])
        self.assertIsNone(
            summary["total_physical_reconfigurations_per_100_deliveries"]
        )
        self.assertIsNone(
            summary[
                "retrieval_executor_obstruction_clearances_per_100_deliveries"
            ]
        )
        self.assertEqual(
            summary["conditional_on_completed_deliveries_diagnostic"][
                "total_physical_reconfigurations_per_100_completed_deliveries"
            ],
            50.0,
        )
        self.assertEqual(
            summary["failed_or_incomplete_rows"][0]["method_failure_reason"],
            "no_proposal",
        )


if __name__ == "__main__":
    unittest.main()
