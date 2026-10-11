from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_vcg_exhaustive_live_recovery_tree_5x5 as runner
from vcg_robust_recovery_snapshot_5x5 import (
    canonical_actions,
    make_two_block_5x5_fixture,
)


def _execution_metrics(
    *,
    raw_return=0.0,
    live_steps=0,
    physical_rehandles=0,
    absolute_error_sum=0.0,
    delivery_count=0,
):
    return {
        "raw_return": float(raw_return),
        "live_steps": int(live_steps),
        "physical_rehandles": int(physical_rehandles),
        "absolute_error_sum": float(absolute_error_sum),
        "delivery_count": int(delivery_count),
        "delivery_deviations": [],
    }


def _edge(disturbance_id, child_summary, **metrics):
    return {
        "disturbance_id": disturbance_id,
        "child_summary": child_summary,
        "execution_metrics": _execution_metrics(**metrics),
    }


def _self_hashed(payload, field):
    value = dict(payload)
    value[field] = runner.snapshot_bridge._digest(value)
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _valid_row(*, status=runner.STATUS_PASS, cutoff_count=0):
    complete = status == runner.STATUS_PASS
    return {
        "instance_seed": 89_000,
        "method": "recursive",
        "handling_lambda": 0.2,
        "root_provenance": {
            "root_environment_digest": "root-digest",
            "source_root_replay_authenticated": True,
            "source_decision_index": 4,
            "source_decision_epoch": 9,
            "source_behavior_digest": "behavior-digest",
            "source_ledger_sha256": "source-ledger-digest",
            "observed_occupancy": 2,
        },
        "episode_instance_id": "episode-instance-89000",
        "schedule_id": "schedule-89000",
        "initial_budget": 20,
        "cohort_labels": ["B1", "B2"],
        "cohort_size": 2,
        "universal_status": status,
        "universal_pass": complete,
        "coverage": {
            "all_expanded_nodes_cover_every_declared_id": True,
            "abstract_outcome_count": 0,
            "node_cutoff_count": int(cutoff_count),
        },
        "graph": {
            "unique_node_count": 1,
            "live_edge_count": 0,
            "nodes": [{"status": status}],
            "edges": [],
            "node_status_counts": {status: 1},
            "edge_status_counts": {},
            "maximum_depth": 0,
            "maximum_fanout": 0,
        },
        "path_summary": runner._empty_path_summary(
            status=status,
            reason="synthetic-test-leaf",
        ),
        "tree_wall_seconds": 0.25,
        "training_or_learning": False,
    }


class TreeLimitTests(unittest.TestCase):
    def test_positive_integer_is_retained(self):
        limits = runner.TreeLimits(max_unique_nodes_per_root=17)

        self.assertEqual(limits.max_unique_nodes_per_root, 17)

    def test_non_positive_non_integer_and_bool_limits_are_rejected(self):
        for value in (0, -1, 1.5, True, False):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    runner.TreeLimits(max_unique_nodes_per_root=value)


class OutcomeRealizationTests(unittest.TestCase):
    def test_every_structural_outcome_has_one_distinct_live_realization(self):
        state = make_two_block_5x5_fixture()
        action = canonical_actions(state)[0]
        outcomes = runner._structural_envelope(state, action)

        realizations = tuple(
            runner.outcome_realization(action, outcome) for outcome in outcomes
        )

        self.assertGreater(len(outcomes), 1)
        self.assertEqual(len(realizations), len(set(realizations)))
        self.assertEqual(
            {
                (
                    realization.delay_steps,
                    action.destination
                    if realization.adjacent_stop is None
                    else realization.adjacent_stop,
                )
                for realization in realizations
            },
            {(outcome.delay_steps, outcome.stop_cell) for outcome in outcomes},
        )
        self.assertEqual(
            sum(realization.adjacent_stop is None for realization in realizations),
            len({outcome.delay_steps for outcome in outcomes}),
        )


class TerminalAliasBudgetTests(unittest.TestCase):
    def test_over_budget_terminal_alias_fails_without_recursing_to_empty_pass(self):
        state = make_two_block_5x5_fixture()
        action = canonical_actions(state)[0]
        empty_successor = replace(state, blocks=(), agent_position=action.destination)
        candidate = SimpleNamespace(successor_state=empty_successor)
        outcome = SimpleNamespace(
            delay_steps=1,
            stop_cell=action.destination,
            primitive_steps=action.steps + 1,
            successor=empty_successor,
        )
        env = SimpleNamespace(time_steps=0, current_state=(1, 1))
        env.is_state_terminal = lambda current_state: True
        execution = SimpleNamespace(
            success=True,
            total_primitive_steps=action.steps,
            illegal_drops=0,
        )

        explorer = object.__new__(runner.SegmentTreeExplorer)
        explorer.base_agent = SimpleNamespace(config=SimpleNamespace(gamma=0.99))
        explorer.search_config = None
        explorer.live_execution_count = 0
        explorer.terminal_alias_count = 0
        explorer._node_key = lambda environment, budget: ("same-digest", {})
        explorer._exact_summary = lambda observed: {
            "trace_matches_certified_action": True,
            "planner_calls": 0,
            "exact_executor_replan_count": 0,
        }
        explorer._execution_metrics = lambda observed: _execution_metrics()
        explorer._visit = Mock(
            side_effect=AssertionError(
                "over-budget terminal aliases must not recurse"
            )
        )

        with (
            patch.object(runner.v1, "_would_complete_live_episode", return_value=True),
            patch.object(runner.v2, "_execute_exact_selected", return_value=execution),
            patch.object(
                runner.v2,
                "_live_recovery_state",
                return_value=empty_successor,
            ),
        ):
            result = explorer._execute_branch(
                env=env,
                state=state,
                candidate=candidate,
                action=action,
                outcome=outcome,
                remaining_budget=action.steps,
                depth=0,
            )

        self.assertEqual(result["status"], runner.STATUS_FAIL)
        self.assertTrue(result["terminal_alias"])
        self.assertIsNone(result["child_id"])
        self.assertEqual(
            result["child_summary"]["leaf_reason"],
            "carried_recovery_budget_exhausted_by_declared_outcome",
        )
        self.assertEqual(
            result["audit"]["remaining_budget_after_abstract_charge"],
            -1,
        )
        explorer._visit.assert_not_called()


class LiveCausalStateTests(unittest.TestCase):
    @staticmethod
    def _environment():
        block = SimpleNamespace(
            label="B1",
            position=(2, 2),
            storage_location=(2, 2),
            carrying=False,
            stored=True,
            delivered=False,
            picked=True,
            cleared=False,
            hold=False,
            hold_counter=0,
            storage_steps_needed=30,
            storage_steps_elapsed=7,
            arrival_step=2,
            stored_time_step=5,
            delivered_time_step=None,
            delivery_error_time=None,
            last_position=(2, 1),
        )
        env = SimpleNamespace(
            current_episode_instance=SimpleNamespace(instance_id="instance-1"),
            time_steps=12,
            current_state=(1, 1),
            blocks=[block],
            storage_counts={(2, 2): 1, (2, 3): 0},
            store_events=[{"label": "B1", "time": 5}],
            delivery_error_times=[],
        )
        env.is_state_terminal = lambda state: False
        return env

    def _digest(self, env, *, budget=20, method="recursive", value=0.2):
        causal = runner.live_causal_state(
            env,
            remaining_budget=budget,
            cohort_labels=("B1",),
            method=method,
            handling_lambda=value,
        )
        return runner.v1._digest(causal)

    def test_budget_method_lambda_and_live_clock_are_distinct_memo_keys(self):
        env = self._environment()
        later = deepcopy(env)
        later.time_steps += 1

        digests = {
            self._digest(env),
            self._digest(env, budget=19),
            self._digest(env, method="one_step"),
            self._digest(env, value=0.0),
            self._digest(later),
        }

        self.assertEqual(len(digests), 5)

    def test_block_clock_and_event_history_are_causal(self):
        env = self._environment()
        aged = deepcopy(env)
        aged.blocks[0].storage_steps_elapsed += 1
        new_event = deepcopy(env)
        new_event.store_events.append({"label": "B1", "time": 6})

        self.assertNotEqual(self._digest(env), self._digest(aged))
        self.assertNotEqual(self._digest(env), self._digest(new_event))


class PathSummaryTests(unittest.TestCase):
    def test_status_precedence_is_invalid_fail_unknown_pass(self):
        self.assertEqual(
            runner._status_from_children([runner.STATUS_PASS, runner.STATUS_FAIL]),
            runner.STATUS_FAIL,
        )
        self.assertEqual(
            runner._status_from_children([runner.STATUS_FAIL, runner.STATUS_UNKNOWN]),
            runner.STATUS_FAIL,
        )
        self.assertEqual(
            runner._status_from_children(
                [runner.STATUS_PASS, runner.STATUS_UNKNOWN, runner.STATUS_INVALID]
            ),
            runner.STATUS_INVALID,
        )
        self.assertEqual(runner._status_from_children([]), runner.STATUS_INVALID)

    def test_recursive_summary_prepends_paths_and_adds_edge_metrics(self):
        completed_leaf = runner._empty_path_summary(
            status=runner.STATUS_PASS,
            reason="complete",
        )
        child = runner._combine_edge_summaries(
            [
                _edge(
                    "child",
                    completed_leaf,
                    raw_return=3,
                    live_steps=4,
                    physical_rehandles=1,
                    absolute_error_sum=6,
                    delivery_count=1,
                )
            ]
        )
        root = runner._combine_edge_summaries(
            [
                _edge(
                    "root",
                    child,
                    raw_return=-2,
                    live_steps=5,
                    physical_rehandles=2,
                    absolute_error_sum=4,
                    delivery_count=1,
                )
            ]
        )

        self.assertEqual(root["total_path_count"], 1)
        self.assertEqual(root["completed_path_count"], 1)
        extrema = root["completed_extrema"]
        self.assertEqual(extrema["minimum_raw_return"], {
            "value": 1.0,
            "path": ["root", "child"],
        })
        self.assertEqual(extrema["maximum_live_steps"]["value"], 9)
        self.assertEqual(extrema["maximum_physical_rehandles"]["value"], 3)
        self.assertEqual(
            extrema["maximum_absolute_error_sum"],
            {
                "value": 10.0,
                "delivery_count": 2,
                "mean_absolute_error": 5.0,
                "path": ["root", "child"],
            },
        )

    def test_failure_witness_is_prefixed_without_becoming_completed_extremum(self):
        passed = runner._empty_path_summary(
            status=runner.STATUS_PASS,
            reason="complete",
        )
        failed = runner._empty_path_summary(
            status=runner.STATUS_FAIL,
            reason="counterexample",
        )

        summary = runner._combine_edge_summaries(
            [
                _edge("good", passed, raw_return=4),
                _edge("bad", failed, raw_return=100),
            ]
        )

        self.assertEqual(summary["total_path_count"], 2)
        self.assertEqual(summary["completed_path_count"], 1)
        self.assertEqual(summary["status_path_counts"], {
            runner.STATUS_FAIL: 1,
            runner.STATUS_PASS: 1,
        })
        self.assertEqual(summary["witnesses"], {runner.STATUS_FAIL: ["bad"]})
        self.assertEqual(
            summary["completed_extrema"]["minimum_raw_return"]["value"],
            4.0,
        )


class ParentArtifactAuthenticationTests(unittest.TestCase):
    def _artifacts(self, directory: Path):
        contract = _self_hashed(
            {"protocol": runner.snapshot_bridge.PROTOCOL},
            "contract_sha256",
        )
        snapshots = [
            {
                "instance_seed": 89_000,
                "observed_occupancy": 2,
            },
            *[
                {
                    "instance_seed": 90_000 + index,
                    "observed_occupancy": 2,
                }
                for index in range(29)
            ],
        ]
        panel = _self_hashed(
            {
                "protocol": runner.snapshot_bridge.PROTOCOL,
                "contract_sha256": contract["contract_sha256"],
                "snapshot_count": 30,
                "snapshots": snapshots,
            },
            "panel_sha256",
        )
        report = _self_hashed(
            {
                "protocol": runner.snapshot_bridge.PROTOCOL,
                "contract_sha256": contract["contract_sha256"],
                "panel_sha256": panel["panel_sha256"],
                "status": "passed",
            },
            "report_sha256",
        )
        _write_json(directory / runner.snapshot_bridge.CONTRACT_NAME, contract)
        _write_json(directory / runner.snapshot_bridge.PANEL_NAME, panel)
        _write_json(directory / runner.snapshot_bridge.REPORT_NAME, report)
        return contract, panel, report

    def test_smoke_parent_requires_and_accepts_bound_self_hashed_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            expected = self._artifacts(parent)

            with patch.object(runner, "PARENT_SNAPSHOT_OUTPUT", parent):
                observed = runner._read_authenticated_parent(parent, "smoke")

        self.assertEqual(observed, expected)

    def test_rehashed_but_differently_bound_panel_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            _, panel, _ = self._artifacts(parent)
            panel["contract_sha256"] = "different-contract"
            panel["panel_sha256"] = runner.snapshot_bridge._digest(
                panel,
                drop="panel_sha256",
            )
            _write_json(parent / runner.snapshot_bridge.PANEL_NAME, panel)

            with patch.object(runner, "PARENT_SNAPSHOT_OUTPUT", parent):
                with self.assertRaisesRegex(
                    runner.ExhaustiveLiveTreeError,
                    "panel/contract binding changed",
                ):
                    runner._read_authenticated_parent(parent, "smoke")

    def test_rehashed_report_bound_to_different_panel_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            _, _, report = self._artifacts(parent)
            report["panel_sha256"] = "different-panel"
            report["report_sha256"] = runner.snapshot_bridge._digest(
                report,
                drop="report_sha256",
            )
            _write_json(parent / runner.snapshot_bridge.REPORT_NAME, report)

            with patch.object(runner, "PARENT_SNAPSHOT_OUTPUT", parent):
                with self.assertRaisesRegex(
                    runner.ExhaustiveLiveTreeError,
                    "report/panel binding changed",
                ):
                    runner._read_authenticated_parent(parent, "smoke")


class RowAndLedgerValidationTests(unittest.TestCase):
    def setUp(self):
        self.contract = {"contract_sha256": "contract-digest"}

    def _validate_row(self, row):
        return runner._validate_row(
            row,
            contract=self.contract,
            instance_seed=89_000,
            method="recursive",
            handling_lambda=0.2,
            expected_root_digest="root-digest",
        )

    def test_universal_pass_row_satisfies_closed_tree_invariants(self):
        row = _valid_row()

        self.assertEqual(self._validate_row(row), row)

    def test_row_without_authenticated_source_root_is_rejected(self):
        row = _valid_row()
        row["root_provenance"].pop("source_root_replay_authenticated")

        with self.assertRaisesRegex(
            runner.ExhaustiveLiveTreeError,
            "source root was not authenticated",
        ):
            self._validate_row(row)

    def test_pass_cannot_hide_a_resource_cutoff(self):
        row = _valid_row(cutoff_count=1)

        with self.assertRaisesRegex(
            runner.ExhaustiveLiveTreeError,
            "PASS tree violates universal invariants",
        ):
            self._validate_row(row)

    def test_explicit_unknown_cutoff_is_not_misclassified_as_failure_or_pass(self):
        row = _valid_row(status=runner.STATUS_UNKNOWN, cutoff_count=1)

        observed = self._validate_row(row)

        self.assertFalse(observed["universal_pass"])
        self.assertEqual(observed["universal_status"], runner.STATUS_UNKNOWN)
        self.assertEqual(observed["path_summary"]["completed_path_count"], 0)

    def test_ledger_hash_and_contract_binding_are_authenticated(self):
        row = _valid_row()
        ledger = {
            "schema_version": runner.SCHEMA_VERSION,
            "protocol": runner.PROTOCOL,
            "contract_sha256": self.contract["contract_sha256"],
            "instance_seed": 89_000,
            "method": "recursive",
            "handling_lambda": 0.2,
            "row": row,
        }
        ledger["ledger_sha256"] = runner.v1._digest(ledger)

        observed = runner._validate_ledger(
            ledger,
            contract=self.contract,
            instance_seed=89_000,
            method="recursive",
            handling_lambda=0.2,
            expected_root_digest="root-digest",
        )
        self.assertEqual(observed, row)

        tampered = deepcopy(ledger)
        tampered["contract_sha256"] = "different-contract"
        tampered["ledger_sha256"] = runner.v1._digest(
            tampered,
            drop="ledger_sha256",
        )
        with self.assertRaisesRegex(
            runner.ExhaustiveLiveTreeError,
            "contract_sha256 changed",
        ):
            runner._validate_ledger(
                tampered,
                contract=self.contract,
                instance_seed=89_000,
                method="recursive",
                handling_lambda=0.2,
                expected_root_digest="root-digest",
            )


class CompactReportRowTests(unittest.TestCase):
    def test_compact_row_references_ledger_and_omits_full_dag_payload(self):
        row = _valid_row()
        row["graph"]["nodes"][0]["large_node_audit"] = [1, 2, 3]
        ledger = {"ledger_sha256": "ledger-digest"}

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary).resolve()
            ledger_path = (
                output_dir
                / "root-ledger"
                / "instance-89000"
                / "recursive"
                / "lambda-0p2.json"
            )
            compact = runner._compact_report_row(
                row,
                ledger=ledger,
                ledger_path=ledger_path,
                output_dir=output_dir,
            )

        self.assertNotIn("nodes", compact["graph"])
        self.assertNotIn("edges", compact["graph"])
        self.assertEqual(compact["graph"]["unique_node_count"], 1)
        self.assertEqual(compact["graph"]["live_edge_count"], 0)
        self.assertEqual(compact["graph"]["maximum_depth"], 0)
        self.assertEqual(compact["graph"]["maximum_fanout"], 0)
        self.assertEqual(compact["path_summary"], row["path_summary"])
        self.assertEqual(
            compact["ledger_path"],
            "root-ledger/instance-89000/recursive/lambda-0p2.json",
        )
        self.assertEqual(compact["ledger_sha256"], "ledger-digest")


if __name__ == "__main__":
    unittest.main()
