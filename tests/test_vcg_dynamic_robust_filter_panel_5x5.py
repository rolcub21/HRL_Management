from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from PSLAP.viability import RecoveryAction, RecoveryActionKind, ViabilityStatus
from PSLAP.viability_candidates import ViabilityActionType
import run_vcg_dynamic_robust_filter_panel_5x5 as runner
from vcg_robust_recovery_snapshot_5x5 import SolveStatus
from viability_graph_hierarchy import MODE_TO_ID


def _candidate(key, action_type, *, blocks=("B",)):
    recovery_action = None
    if action_type in (
        ViabilityActionType.DELIVER,
        ViabilityActionType.RECONFIGURE,
    ):
        _, label, row, column = key.split(":")
        destination = (int(row), int(column))
        source = (0, 0)
        recovery_action = RecoveryAction(
            kind=(
                RecoveryActionKind.DELIVERY
                if action_type is ViabilityActionType.DELIVER
                else RecoveryActionKind.RELOCATION
            ),
            block_label=label,
            source=source,
            destination=destination,
            approach_path=(source,),
            transport_path=(source, destination),
        )
    return SimpleNamespace(
        key=key,
        action_type=action_type,
        successor_state=SimpleNamespace(blocks=tuple(blocks)),
        recovery_action=recovery_action,
        target_label=(
            None
            if action_type is ViabilityActionType.DEFER
            else key.split(":")[1]
        ),
    )


def _snapshot(*, current_status=ViabilityStatus.SAFE):
    candidates = (
        _candidate("accept:A:1:1", ViabilityActionType.ACCEPT),
        _candidate("deliver:B:4:3", ViabilityActionType.DELIVER),
        _candidate("reconfigure:B:2:2", ViabilityActionType.RECONFIGURE),
        _candidate("defer:3", ViabilityActionType.DEFER),
    )
    current = SimpleNamespace(
        status=current_status,
        witness_primitive_steps=(
            7 if current_status is ViabilityStatus.SAFE else None
        ),
        witness_macro_count=(
            2 if current_status is ViabilityStatus.SAFE else None
        ),
    )
    return SimpleNamespace(
        candidates=candidates,
        current_certificate=current,
        recovery_state="hashable-recovery-state",
    )


def _method_certificate(*keys):
    return SimpleNamespace(
        admitted_action_keys=tuple(keys),
        state=SimpleNamespace(status=SolveStatus.WINNING),
    )


class RecoveryFilterTests(unittest.TestCase):
    def test_recovery_keys_are_intersected_while_accept_and_defer_pass_through(self):
        certificate = SimpleNamespace(
            semantic_digest="semantic",
            nominal=_method_certificate(
                "deliver:B:4:3",
                "reconfigure:B:2:2",
                "deliver:solver-only:0:0",
            ),
            one_step=_method_certificate("deliver:B:4:3"),
            recursive=_method_certificate(),
        )
        expected = {
            "nominal": {
                "accept:A:1:1",
                "deliver:B:4:3",
                "reconfigure:B:2:2",
                "defer:3",
            },
            "one_step": {
                "accept:A:1:1",
                "deliver:B:4:3",
                "defer:3",
            },
            "recursive": {"accept:A:1:1", "defer:3"},
        }
        memo = {}
        with (
            patch.object(runner, "certify_snapshot", return_value=certificate) as solve,
            patch.object(runner, "method_to_dict", return_value={"test": True}),
        ):
            observed = {}
            audits = {}
            for method in runner.METHODS:
                observed[method], audits[method] = runner._filter_recovery_frontier(
                    _snapshot(),
                    method=method,
                    memo=memo,
                    memo_namespace=86_001,
                )

        self.assertEqual(
            {key: set(value) for key, value in observed.items()},
            expected,
        )
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(solve.call_args.kwargs["primitive_budget"], 13)
        self.assertEqual(
            solve.call_args.kwargs["max_expansions"],
            runner.MAX_EXPANSIONS,
        )
        self.assertFalse(audits["nominal"]["memo_hit"])
        self.assertTrue(audits["one_step"]["memo_hit"])
        self.assertTrue(audits["recursive"]["memo_hit"])
        self.assertNotIn(
            "deliver:solver-only:0:0",
            audits["nominal"]["admitted_recovery_keys"],
        )
        self.assertEqual(
            audits["recursive"]["rejected_recovery_keys"],
            ["deliver:B:4:3", "reconfigure:B:2:2"],
        )

    def test_missing_finite_current_witness_fails_recovery_closed_only(self):
        with patch.object(runner, "certify_snapshot") as solve:
            admitted, audit = runner._filter_recovery_frontier(
                _snapshot(current_status=ViabilityStatus.UNKNOWN),
                method="recursive",
                memo={},
                memo_namespace=86_001,
            )

        solve.assert_not_called()
        self.assertEqual(admitted, frozenset({"accept:A:1:1", "defer:3"}))
        self.assertEqual(
            audit["status"], "UNKNOWN_NO_FINITE_CURRENT_NOMINAL_WITNESS"
        )


class RealizationAndSelectionTests(unittest.TestCase):
    def test_disturbance_is_nominal_at_terminal_and_bounded_when_nonterminal(self):
        final_delivery = _candidate(
            "deliver:B:4:3",
            ViabilityActionType.DELIVER,
            blocks=(),
        )
        with patch.object(
            runner,
            "declared_clear_adjacent_stops",
            side_effect=AssertionError("terminal delivery must not request a stop"),
        ):
            terminal = runner._realization(
                SimpleNamespace(
                    blocks=[SimpleNamespace(label="B", delivered=False)]
                ),
                final_delivery,
            )

        nonterminal = _candidate(
            "reconfigure:B:2:2",
            ViabilityActionType.RECONFIGURE,
        )
        with patch.object(
            runner,
            "declared_clear_adjacent_stops",
            return_value=((2, 1), (1, 2)),
        ):
            disturbed = runner._realization(
                SimpleNamespace(
                    blocks=[
                        SimpleNamespace(label="B", delivered=False),
                        SimpleNamespace(label="C", delivered=False),
                    ]
                ),
                nonterminal,
            )

        accept = runner._realization(
            SimpleNamespace(blocks=[]),
            _candidate("accept:A:1:1", ViabilityActionType.ACCEPT)
        )
        self.assertEqual(terminal.delay_steps, 0)
        self.assertIsNone(terminal.adjacent_stop)
        self.assertEqual(accept.delay_steps, 0)
        self.assertEqual(disturbed.delay_steps, 1)
        self.assertEqual(disturbed.adjacent_stop, (1, 2))

    def test_same_frozen_selector_moves_only_when_handling_merit_changes(self):
        first = _candidate("deliver:A:4:3", ViabilityActionType.DELIVER)
        second = _candidate("deliver:B:4:3", ViabilityActionType.DELIVER)
        snapshot = SimpleNamespace(candidates=(first, second))
        recover_mode = MODE_TO_ID["recover"]
        prepared = SimpleNamespace(
            records=(
                SimpleNamespace(key=first.key, mode_id=recover_mode),
                SimpleNamespace(key=second.key, mode_id=recover_mode),
            ),
            source_indices=(0, 1),
        )

        class BaseAgent:
            Q_local = object()
            within_temperatures = torch.ones(3)

            @staticmethod
            def _score_records(records):
                if len(records) != 2:
                    raise AssertionError("unexpected retained frontier")
                return torch.tensor([10.0, 9.0])

        with (
            patch.object(runner, "prepare_viability_snapshot", return_value=prepared),
            patch.object(
                runner,
                "score_cost_records",
                return_value=torch.tensor([10.0, 0.0]),
            ) as cost_score,
        ):
            unconstrained = runner._select_frozen(
                snapshot,
                admitted_keys=frozenset({first.key, second.key}),
                base_agent=BaseAgent(),
                cost_network=object(),
                handling_lambda=0.0,
            )
            constrained = runner._select_frozen(
                snapshot,
                admitted_keys=frozenset({first.key, second.key}),
                base_agent=BaseAgent(),
                cost_network=object(),
                handling_lambda=0.2,
            )

        self.assertEqual(unconstrained.candidate.key, first.key)
        self.assertFalse(unconstrained.audit["cost_head_called"])
        self.assertEqual(constrained.candidate.key, second.key)
        self.assertTrue(constrained.audit["cost_head_called"])
        cost_score.assert_called_once()


def _summary_row(*, strict, failure=None, dense=100.0, rehandles=1):
    return {
        "strict_safe_complete": strict,
        "terminal": strict,
        "method_failure_reason": failure,
        "dense_return": dense,
        "mean_absolute_error": 4.0,
        "steps": 20,
        "physical_rehandles": rehandles,
        "within_target_window_rate": 0.75,
        "recovery_filter_decision_count": 2,
        "recovery_candidates_removed": 1,
        "macro_failures": 0 if strict else 1,
        "illegal_drops": 0,
        "recovery_certification_wall_seconds": 0.1,
        "logical_recovery_expanded_nodes": 3,
        "recovery_decisions_hitting_compute_cutoff": 0,
        "episode_wall_seconds": 0.2,
    }


class AggregationAndLedgerTests(unittest.TestCase):
    def test_one_incomplete_row_suppresses_whole_cell_performance(self):
        summary = runner._cell_summary(
            (
                _summary_row(strict=True, dense=200.0),
                _summary_row(strict=False, failure="deadlock", dense=-999.0),
            )
        )

        self.assertEqual(summary["strict_safe_complete_count"], 1)
        self.assertFalse(summary["whole_cell_performance_eligible"])
        self.assertIsNone(summary["metrics"])
        self.assertEqual(
            summary["failure_reason_counts"],
            {"deadlock": 1, "none": 1},
        )

    def test_complete_cell_reports_declared_metrics(self):
        summary = runner._cell_summary(
            (
                _summary_row(strict=True, dense=100.0, rehandles=1),
                _summary_row(strict=True, dense=200.0, rehandles=3),
            )
        )

        self.assertTrue(summary["whole_cell_performance_eligible"])
        self.assertEqual(summary["metrics"]["mean_dense_return"], 150.0)
        self.assertEqual(
            summary["metrics"]["physical_rehandles_per_100"],
            25.0,
        )

    def test_ledger_identity_is_checked_after_self_hash(self):
        contract = {"contract_sha256": "contract", "device": "cpu"}
        record = {
            "instance_seed": 86_001,
            "episode_instance_id": "instance",
            "schedule_id": "schedule",
        }
        legacy_return, dense_return = runner._dual_rescore_from_legacy_return(
            0.0,
            [],
            runner.v11.FROZEN_OBJECTIVE_SPEC,
        )
        row = {
            **record,
            "method": "recursive",
            "handling_lambda": 0.2,
            "device": "cpu",
            "strict_safe_complete": False,
            "terminal": False,
            "success": False,
            "method_failure_reason": "test_incomplete",
            "delivery_count": 0,
            "delivery_deviations": [],
            "raw_environment_return": 0.0,
            "return": legacy_return,
            "dense_return": dense_return,
            "mean_absolute_error": 0.0,
            "steps": 0,
            "physical_rehandles": 0,
            "physical_rehandles_per_100": 0.0,
            "illegal_drops": 0,
            "macro_failures": 0,
            "complete_nominal_frontier_exactly_verified": False,
            "decisions": [],
        }
        behavior = {
            "instance_seed": row["instance_seed"],
            "episode_instance_id": row["episode_instance_id"],
            "method": row["method"],
            "handling_lambda": row["handling_lambda"],
            "decisions": row["decisions"],
            "final_environment": None,
            "return": row["raw_environment_return"],
            "steps": row["steps"],
            "terminal": row["terminal"],
            "failure": row["method_failure_reason"],
        }
        row["behavior_digest"] = runner._digest(
            runner._behavior_semantics(behavior)
        )
        ledger = {
            "schema_version": runner.SCHEMA_VERSION,
            "protocol": runner.PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "method": "recursive",
            "handling_lambda": 0.2,
            **record,
            "checkpoint_sha256": runner.source89.SELECTED_CHECKPOINT_SHA256[
                runner.MODEL_SEED
            ],
            "cost_head_sha256": runner.source89.COST_HEAD_SHA256[
                runner.MODEL_SEED
            ],
            "row": row,
        }
        ledger["ledger_sha256"] = runner._digest(ledger)

        self.assertEqual(
            runner._validate_ledger(
                ledger,
                contract=contract,
                method="recursive",
                handling_lambda=0.2,
                record=record,
            ),
            row,
        )

        changed = deepcopy(ledger)
        changed["row"]["instance_seed"] += 1
        changed["ledger_sha256"] = runner._digest(changed, drop="ledger_sha256")
        with self.assertRaisesRegex(runner.DynamicFilterError, "row identity"):
            runner._validate_ledger(
                changed,
                contract=contract,
                method="recursive",
                handling_lambda=0.2,
                record=record,
            )


if __name__ == "__main__":
    unittest.main()
