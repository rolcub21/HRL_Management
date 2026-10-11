from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PSLAP.dynamic_yard import BlockView
from PSLAP.viability import ViabilityStatus, apply_recovery_action
from PSLAP.viability_candidates import ViabilityActionType
import run_vcg_dynamic_budgeted_robust_filter_panel_5x5 as runner
from vcg_bounded_macro_execution import BoundedMacroRealization
from vcg_robust_recovery_snapshot_5x5 import (
    SolveStatus,
    canonical_actions,
    make_two_block_5x5_fixture,
    recovery_action_key,
)


def _candidate(action):
    action_type = (
        ViabilityActionType.DELIVER
        if action.kind.value == "DELIVERY"
        else ViabilityActionType.RECONFIGURE
    )
    return SimpleNamespace(
        key=recovery_action_key(action),
        action_type=action_type,
        recovery_action=action,
    )


def _snapshot(state, candidates, *, witness_steps=14, witness_macros=2):
    return SimpleNamespace(
        recovery_state=state,
        candidates=tuple(candidates),
        current_certificate=SimpleNamespace(
            status=ViabilityStatus.SAFE,
            witness_primitive_steps=witness_steps,
            witness_macro_count=witness_macros,
        ),
    )


def _method_certificate(*keys):
    return SimpleNamespace(
        admitted_action_keys=tuple(keys),
        state=SimpleNamespace(status=SolveStatus.WINNING),
    )


class TimingProjectionAndActionAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.state = make_two_block_5x5_fixture()
        self.action = canonical_actions(self.state)[0]

    def test_timing_erasure_canonicalizes_only_remaining_time(self):
        aged = replace(
            self.state,
            blocks=tuple(
                replace(block, remaining_time=block.remaining_time - 17.5)
                for block in self.state.blocks
            ),
        )

        erased_original = runner._timing_erased_state(self.state)
        erased_aged = runner._timing_erased_state(aged)

        self.assertEqual(erased_original, erased_aged)
        self.assertTrue(
            all(block.remaining_time == 0.0 for block in erased_aged.blocks)
        )
        self.assertEqual(
            runner._recovery_physical_key(self.state),
            runner._recovery_physical_key(aged),
        )

    def test_full_action_equality_rejects_same_short_key_with_changed_path(self):
        changed = replace(
            self.action,
            approach_path=(
                self.action.approach_path
                + (self.action.approach_path[-1],)
            ),
        )
        snapshot = _snapshot(self.state, (_candidate(changed),))

        with self.assertRaisesRegex(
            runner.BudgetedDynamicFilterError,
            "differs from canonical",
        ):
            runner._authenticated_recovery_actions(snapshot)

    def test_physical_comparison_catches_any_extra_stored_label(self):
        expected = runner._recovery_physical_key(self.state)
        empty = next(
            cell
            for cell in sorted(self.state.storage_cells)
            if cell not in self.state.occupancy()
        )
        extra = replace(
            self.state,
            blocks=self.state.blocks
            + (BlockView("EXTERNAL", empty, remaining_time=0.0),),
        )

        comparison = runner._physical_comparison(
            expected,
            runner._recovery_physical_key(extra),
        )

        self.assertFalse(comparison["matched"])
        self.assertEqual(comparison["mismatched_fields"], ["blocks"])


class CommittedSegmentTests(unittest.TestCase):
    def setUp(self):
        self.state = runner._timing_erased_state(make_two_block_5x5_fixture())

    def test_budget_is_activated_once_carried_and_never_replenished(self):
        segment = runner.CommittedRecoverySegment()
        activation = segment.activate(
            self.state,
            primitive_budget=20,
            nominal_witness_primitive_steps=14,
            nominal_witness_macro_count=2,
            remaining_episode_horizon=50,
        )
        first = segment.consume(6, action_type="reconfigure")
        successor = apply_recovery_action(
            self.state,
            next(
                action
                for action in canonical_actions(self.state)
                if action.kind.value == "DELIVERY"
            ),
        )
        boundary = segment.boundary(successor)

        self.assertEqual(activation["initial_budget"], 20)
        self.assertEqual(first["before"], 20)
        self.assertEqual(first["after"], 14)
        self.assertEqual(boundary["remaining_budget"], 14)
        self.assertEqual(segment.activation_count, 1)

    def test_new_stored_label_during_active_segment_fails_closed(self):
        segment = runner.CommittedRecoverySegment()
        segment.activate(
            self.state,
            primitive_budget=20,
            nominal_witness_primitive_steps=14,
            nominal_witness_macro_count=2,
            remaining_episode_horizon=20,
        )
        empty = next(
            cell
            for cell in sorted(self.state.storage_cells)
            if cell not in self.state.occupancy()
        )
        changed = replace(
            self.state,
            blocks=self.state.blocks + (BlockView("NEW", empty, 0.0),),
        )

        with self.assertRaisesRegex(
            runner.BudgetedDynamicFilterError,
            "unexpected stored label",
        ):
            segment.boundary(changed)

    def test_completion_discards_leftover_and_reopens_inactive_state(self):
        segment = runner.CommittedRecoverySegment()
        segment.activate(
            self.state,
            primitive_budget=20,
            nominal_witness_primitive_steps=14,
            nominal_witness_macro_count=2,
            remaining_episode_horizon=20,
        )
        segment.consume(9, action_type="deliver")

        completion = segment.close_completed()

        self.assertEqual(completion["discarded_leftover_budget"], 11)
        self.assertFalse(segment.active)
        self.assertEqual(segment.completion_count, 1)


class BudgetAndFilteringTests(unittest.TestCase):
    def setUp(self):
        self.state = make_two_block_5x5_fixture()
        self.action = canonical_actions(self.state)[0]
        self.recovery = _candidate(self.action)
        self.accept = SimpleNamespace(
            key="accept:X:1:1",
            action_type=ViabilityActionType.ACCEPT,
            recovery_action=None,
        )
        self.defer = SimpleNamespace(
            key="defer:10",
            action_type=ViabilityActionType.DEFER,
            recovery_action=None,
        )
        self.snapshot = _snapshot(
            self.state,
            (self.accept, self.recovery, self.defer),
        )
        method = _method_certificate(self.recovery.key)
        self.certificate = SimpleNamespace(
            semantic_digest="semantic",
            nominal=method,
            one_step=method,
            recursive=method,
        )

    def test_provisional_budget_is_capped_by_remaining_live_horizon(self):
        budget = runner._provisional_budget(
            self.snapshot,
            remaining_episode_horizon=17,
        )

        self.assertEqual(budget["uncapped_budget"], 20)
        self.assertEqual(budget["primitive_budget"], 17)

    def test_inactive_frontier_retains_accept_and_defer_before_commitment(self):
        with (
            patch.object(runner, "certify_snapshot", return_value=self.certificate),
            patch.object(runner, "method_to_dict", return_value={"test": True}),
        ):
            admitted, audit = runner._filter_recovery_frontier(
                self.snapshot,
                method="recursive",
                primitive_budget=20,
                segment_active=False,
                memo={},
                memo_namespace=1,
            )

        self.assertEqual(
            admitted,
            frozenset({self.accept.key, self.recovery.key, self.defer.key}),
        )
        self.assertEqual(audit["pass_through_keys"], [
            self.accept.key,
            self.defer.key,
        ])

    def test_active_frontier_masks_accept_and_defer_and_uses_carried_budget(self):
        with (
            patch.object(runner, "certify_snapshot", return_value=self.certificate) as solve,
            patch.object(runner, "method_to_dict", return_value={"test": True}),
        ):
            admitted, audit = runner._filter_recovery_frontier(
                self.snapshot,
                method="recursive",
                primitive_budget=11,
                segment_active=True,
                memo={},
                memo_namespace=1,
            )

        self.assertEqual(admitted, frozenset({self.recovery.key}))
        self.assertEqual(audit["pass_through_keys"], [])
        self.assertEqual(
            audit["masked_nonrecovery_keys"],
            [self.accept.key, self.defer.key],
        )
        self.assertEqual(solve.call_args.kwargs["primitive_budget"], 11)
        certified_state = solve.call_args.args[0]
        self.assertTrue(
            all(block.remaining_time == 0.0 for block in certified_state.blocks)
        )

    def test_unknown_certificate_admits_no_recovery(self):
        unknown = SimpleNamespace(
            admitted_action_keys=(),
            state=SimpleNamespace(status=SolveStatus.UNKNOWN),
        )
        certificate = SimpleNamespace(
            semantic_digest="unknown",
            nominal=unknown,
            one_step=unknown,
            recursive=unknown,
        )
        with (
            patch.object(runner, "certify_snapshot", return_value=certificate),
            patch.object(runner, "method_to_dict", return_value={"test": True}),
        ):
            admitted, audit = runner._filter_recovery_frontier(
                self.snapshot,
                method="recursive",
                primitive_budget=11,
                segment_active=True,
                memo={},
                memo_namespace=1,
            )

        self.assertEqual(admitted, frozenset())
        self.assertEqual(audit["status"], SolveStatus.UNKNOWN.value)


class DeclaredOutcomeAndDurationTests(unittest.TestCase):
    def setUp(self):
        self.state = runner._timing_erased_state(make_two_block_5x5_fixture())
        self.action = canonical_actions(self.state)[0]
        self.candidate = _candidate(self.action)

    def test_declared_outcome_cost_and_endpoint_match_requested_realization(self):
        nominal = BoundedMacroRealization(delay_steps=1)

        outcome = runner._declared_recovery_outcome(
            self.state,
            self.action,
            realization=nominal,
            primitive_budget=50,
        )

        self.assertEqual(outcome["delay_steps"], 1)
        self.assertEqual(outcome["stop_cell"], list(self.action.destination))
        self.assertEqual(outcome["primitive_steps"], self.action.steps + 1)
        self.assertEqual(
            outcome["expected_physical_key"]["agent_position"],
            list(self.action.destination),
        )

    def test_duration_verification_requires_base_injection_total_and_no_replan(self):
        realization = BoundedMacroRealization(delay_steps=1)
        total = self.action.steps + 1
        execution = SimpleNamespace(
            total_primitive_steps=total,
            audit={
                "base_primitive_steps": self.action.steps,
                "injected_primitive_steps": 1,
                "total_primitive_steps": total,
                "base_replan_count": 0,
                "exact_path": {
                    "trace_matches_certified_action": True,
                    "planner_calls": 0,
                    "replan_count": 0,
                },
            },
        )

        matched = runner._duration_verification(
            self.candidate,
            realization=realization,
            execution=execution,
        )
        changed = SimpleNamespace(
            total_primitive_steps=total + 1,
            audit=execution.audit,
        )
        mismatched = runner._duration_verification(
            self.candidate,
            realization=realization,
            execution=changed,
        )

        self.assertTrue(matched["matched"])
        self.assertFalse(mismatched["matched"])


def _valid_segment_replay_row():
    expected_nonempty = {"blocks": [{"label": "A", "position": [2, 2]}]}
    expected_empty = {"blocks": []}
    decisions = [
        {
            "decision_index": 0,
            "segment_boundary": {
                "event": "inactive_boundary",
                "active": False,
                "segment_index": None,
                "observed_recovery_labels": ["A", "B"],
            },
            "budget": {"primitive_budget": 10},
            "filter": {
                "segment_active": False,
                "primitive_budget": 10,
                "pass_through_keys": ["accept:X:1:1"],
            },
            "selection": {"selected_key": "reconfigure:B:2:3"},
            "execution": {
                "candidate_key": "reconfigure:B:2:3",
                "action_type": "reconfigure",
                "total_primitive_steps": 4,
                "realization_complete": True,
                "exact_path": {
                    "trace_matches_certified_action": True,
                    "planner_calls": 0,
                    "replan_count": 0,
                    "expected_action_names": ["PICKUP", "RIGHT", "RIGHT", "PUTDOWN"],
                    "emitted_action_names": ["PICKUP", "RIGHT", "RIGHT", "PUTDOWN"],
                    "expected_cell_trace": [[2, 2], [2, 2], [2, 3], [2, 4], [2, 4]],
                    "observed_cell_trace": [[2, 2], [2, 2], [2, 3], [2, 4], [2, 4]],
                    "recovery_action": {"steps": 4},
                },
            },
            "segment_activation": {
                "event": "committed_recovery_segment_activated",
                "segment_index": 0,
                "initial_budget": 10,
                "remaining_budget": 10,
                "cohort_labels": ["A", "B"],
                "nominal_witness_primitive_steps": 4,
                "nominal_witness_macro_count": 2,
                "remaining_episode_horizon": 50,
                "uncapped_budget": 10,
            },
            "recovery_duration_verification": {
                "matched": True,
                "exact_trace_matches_certified_action": True,
                "planner_calls": 0,
                "exact_executor_replan_count": 0,
            },
            "immediate_successor_verification": {
                "matched": True,
                "expected_key": expected_nonempty,
            },
            "next_boundary_successor_verification": {
                "matched": True,
                "expected_key": expected_nonempty,
                "actual_key": expected_nonempty,
            },
            "budget_charge": {
                "charged": True,
                "segment_index": 0,
                "action_type": "reconfigure",
                "duration": 4,
                "before": 10,
                "after": 6,
            },
            "segment_completion": None,
        },
        {
            "decision_index": 1,
            "segment_boundary": {
                "event": "active_continuity_verified",
                "active": True,
                "segment_index": 0,
                "initial_budget": 10,
                "remaining_budget": 6,
                "cohort_labels": ["A", "B"],
                "observed_recovery_labels": ["A"],
            },
            "budget": {"primitive_budget": 6},
            "filter": {
                "segment_active": True,
                "primitive_budget": 6,
                "pass_through_keys": [],
            },
            "selection": {"selected_key": "deliver:A:4:3"},
            "execution": {
                "candidate_key": "deliver:A:4:3",
                "action_type": "deliver",
                "total_primitive_steps": 3,
                "realization_complete": True,
                "exact_path": {
                    "trace_matches_certified_action": True,
                    "planner_calls": 0,
                    "replan_count": 0,
                    "expected_action_names": ["PICKUP", "DOWN", "PUTDOWN"],
                    "emitted_action_names": ["PICKUP", "DOWN", "PUTDOWN"],
                    "expected_cell_trace": [[3, 3], [3, 3], [4, 3], [4, 3]],
                    "observed_cell_trace": [[3, 3], [3, 3], [4, 3], [4, 3]],
                    "recovery_action": {"steps": 3},
                },
            },
            "segment_activation": None,
            "recovery_duration_verification": {
                "matched": True,
                "exact_trace_matches_certified_action": True,
                "planner_calls": 0,
                "exact_executor_replan_count": 0,
            },
            "immediate_successor_verification": {
                "matched": True,
                "expected_key": expected_empty,
            },
            "next_boundary_successor_verification": {
                "matched": True,
                "status": "terminal_boundary_vacuous",
                "expected_key": expected_empty,
                "actual_key": expected_empty,
            },
            "budget_charge": {
                "charged": True,
                "segment_index": 0,
                "action_type": "deliver",
                "duration": 3,
                "before": 6,
                "after": 3,
            },
            "segment_completion": {
                "event": "committed_recovery_segment_completed",
                "completed_segment_index": 0,
                "discarded_leftover_budget": 3,
                "initial_budget": 10,
                "cohort_labels": ["A", "B"],
            },
        },
    ]
    return {
        "decisions": decisions,
        "committed_segment_activation_count": 1,
        "committed_segment_completion_count": 1,
        "committed_segment_open_at_end": False,
        "committed_segment_total_consumed_steps": 7,
        "recovery_duration_verification_count": 2,
        "recovery_duration_mismatch_count": 0,
        "immediate_successor_verification_count": 2,
        "boundary_successor_verification_count": 1,
        "physical_successor_mismatch_count": 0,
    }


class SegmentLedgerReplayTests(unittest.TestCase):
    def test_complete_chain_replays(self):
        row = _valid_segment_replay_row()
        observed = runner._replay_committed_segment_audit(row)
        self.assertEqual(observed["committed_segment_total_consumed_steps"], 7)

    def test_budget_reset_or_replenishment_is_rejected(self):
        row = _valid_segment_replay_row()
        row["decisions"][1]["budget"]["primitive_budget"] = 10
        with self.assertRaisesRegex(
            runner.BudgetedDynamicFilterError,
            "decision budget",
        ):
            runner._replay_committed_segment_audit(row)

    def test_skipped_next_boundary_check_is_rejected(self):
        row = _valid_segment_replay_row()
        row["decisions"][0].pop("next_boundary_successor_verification")
        with self.assertRaisesRegex(
            runner.BudgetedDynamicFilterError,
            "next-boundary",
        ):
            runner._replay_committed_segment_audit(row)

    def test_tampered_aggregate_count_is_rejected(self):
        row = _valid_segment_replay_row()
        row["committed_segment_activation_count"] = 2
        with self.assertRaisesRegex(
            runner.BudgetedDynamicFilterError,
            "activation_count",
        ):
            runner._replay_committed_segment_audit(row)

    def test_failed_realization_cannot_forge_segment_completion(self):
        row = _valid_segment_replay_row()
        row["decisions"][1]["execution"]["realization_complete"] = False
        with self.assertRaisesRegex(
            runner.BudgetedDynamicFilterError,
            "cannot complete",
        ):
            runner._replay_committed_segment_audit(row)


class NullableFailureMetricTests(unittest.TestCase):
    def test_zero_delivery_summary_contains_null_not_nan(self):
        timing = runner._nullable_timing_summary(())
        self.assertEqual(timing["delivery_count"], 0)
        self.assertTrue(
            all(timing[key] is None for key in runner.TIMING_METRIC_FIELDS)
        )
        runner.v1._canonical(timing)

    def test_zero_delivery_failure_ledger_validates_and_aggregation_suppresses(self):
        timing = runner._nullable_timing_summary(())
        legacy_return, dense_return = runner.v1._dual_rescore_from_legacy_return(
            0.0,
            [],
            runner.v1.v11.FROZEN_OBJECTIVE_SPEC,
        )
        row = {
            "instance_seed": 86_001,
            "episode_instance_id": "instance",
            "schedule_id": "schedule",
            "method": "recursive",
            "handling_lambda": 0.2,
            "device": "cpu",
            "strict_safe_complete": False,
            "terminal": False,
            "success": False,
            "method_failure_reason": "empty_active_recovery_frontier",
            "delivery_deviations": [],
            "raw_environment_return": 0.0,
            "return": legacy_return,
            "dense_return": dense_return,
            "steps": 0,
            "physical_rehandles": 0,
            "physical_rehandles_per_100": 0.0,
            **timing,
            "illegal_drops": 0,
            "macro_failures": 0,
            "complete_nominal_frontier_exactly_verified": True,
            "recovery_filter_decision_count": 0,
            "recovery_filter_status_counts": {},
            "recovery_candidates_removed": 0,
            "logical_recovery_expanded_nodes": 0,
            "recovery_decisions_hitting_compute_cutoff": 0,
            "recovery_certification_wall_seconds": 0.0,
            "recovery_certification_memo_hits": 0,
            "committed_segment_activation_count": 0,
            "committed_segment_completion_count": 0,
            "committed_segment_open_at_end": False,
            "committed_segment_total_consumed_steps": 0,
            "recovery_duration_verification_count": 0,
            "recovery_duration_mismatch_count": 0,
            "immediate_successor_verification_count": 0,
            "boundary_successor_verification_count": 0,
            "physical_successor_mismatch_count": 0,
            "timing_erased_recovery_certification": True,
            "episode_wall_seconds": 0.0,
            "final_environment": None,
            "decisions": [],
            "training_or_learning": False,
        }
        behavior = {
            "instance_seed": row["instance_seed"],
            "episode_instance_id": row["episode_instance_id"],
            "method": row["method"],
            "handling_lambda": row["handling_lambda"],
            "decisions": row["decisions"],
            "final_environment": row["final_environment"],
            "return": row["raw_environment_return"],
            "steps": row["steps"],
            "terminal": row["terminal"],
            "failure": row["method_failure_reason"],
        }
        row["behavior_digest"] = runner.v1._digest(
            runner.v1._behavior_semantics(behavior)
        )
        contract = {"contract_sha256": "contract", "device": "cpu"}
        ledger = {
            "schema_version": runner.SCHEMA_VERSION,
            "protocol": runner.PROTOCOL,
            "contract_sha256": "contract",
            "method": "recursive",
            "handling_lambda": 0.2,
            "instance_seed": 86_001,
            "episode_instance_id": "instance",
            "schedule_id": "schedule",
            "checkpoint_sha256": runner.v1.source89.SELECTED_CHECKPOINT_SHA256[
                runner.MODEL_SEED
            ],
            "cost_head_sha256": runner.v1.source89.COST_HEAD_SHA256[
                runner.MODEL_SEED
            ],
            "row": row,
        }
        ledger["ledger_sha256"] = runner.v1._digest(ledger)
        record = {
            "instance_seed": 86_001,
            "episode_instance_id": "instance",
            "schedule_id": "schedule",
        }

        validated = runner._validate_ledger(
            ledger,
            contract=contract,
            method="recursive",
            handling_lambda=0.2,
            record=record,
        )
        summary = runner._cell_summary((validated,))

        self.assertIsNone(validated["mean_absolute_error"])
        self.assertFalse(summary["whole_cell_performance_eligible"])
        self.assertIsNone(summary["metrics"])


class ContractSourceBindingTests(unittest.TestCase):
    def test_every_declared_v2_semantic_source_changes_contract_digest(self):
        baseline = runner._v2_semantic_source_sha256()
        declared_paths = {
            "exact_recovery_executor": (
                runner.HERE / "vcg_exact_recovery_execution.py"
            ),
            "macro_execution_core": runner.HERE / "train_viability_graph_smdp.py",
            "frozen_selector_hierarchy": (
                runner.HERE / "viability_graph_hierarchy.py"
            ),
            "recovery_model": runner.HERE / "PSLAP/viability.py",
            "live_candidate_interface": (
                runner.HERE / "PSLAP/viability_candidates.py"
            ),
            "live_viability_filter": runner.HERE / "PSLAP/viability_filter.py",
            "dynamic_yard_projection": runner.HERE / "PSLAP/dynamic_yard.py",
            "reported_timing_metrics": (
                runner.HERE / "example/helper/timing_metrics.py"
            ),
        }
        self.assertEqual(set(baseline), set(declared_paths))
        original_sha = runner.v1._sha

        for name, target_path in declared_paths.items():
            with self.subTest(source=name):
                def changed_sha(path, *, target=target_path):
                    if path.resolve() == target.resolve():
                        return "0" * 64
                    return original_sha(path)

                with patch.object(runner.v1, "_sha", side_effect=changed_sha):
                    changed = runner._v2_semantic_source_sha256()

                self.assertNotEqual(
                    runner.v1._digest({"source_sha256": baseline}),
                    runner.v1._digest({"source_sha256": changed}),
                )
                self.assertNotEqual(baseline[name], changed[name])


if __name__ == "__main__":
    unittest.main()
