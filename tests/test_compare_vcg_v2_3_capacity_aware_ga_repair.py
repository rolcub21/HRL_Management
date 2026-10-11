import copy
import json
from pathlib import Path
import tempfile
import unittest

import compare_vcg_v2_3_capacity_aware_ga_repair as repair
import compare_vcg_v2_3_matched_baselines as base
from PSLAP.ga_capacity_aware import CapacityAwareRollingGAStorageAssigner
from PSLAP.ga_optimizer import GAConfig
from PSLAP.neutral_protocol import shared_candidate_mask
from PSLAP.track_a import TRACK_A_GA_ROLLING
from tests.test_capacity_aware_rolling_ga import _make_env, _view, _yard


def _identity(seed=85_000):
    return {
        "episode_instance_id": f"instance-{seed}",
        "schedule_id": f"schedule-{seed}",
        "episode_instance_sha256": f"{seed:064x}"[-64:],
    }


def _audited_raw(seed=85_000):
    env = _make_env(5)
    yard = _yard(env)
    block = _view(env)
    candidates = shared_candidate_mask(yard, block, env.pickup_cell)
    config = GAConfig(
        population_size=base.ROLLING_POPULATION,
        generations=base.ROLLING_GENERATIONS,
        elite_count=2,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=0.1,
        seed=base.GA_SEED_BASE + seed,
    )
    source = CapacityAwareRollingGAStorageAssigner(env, config)
    chosen = source.preview(yard, block, env.pickup_cell, candidates)
    mask_id = repair._candidate_mask_sha(candidates)[:16]
    metadata = {
        "proposal_id": "audited-proposal",
        "block_label": block.label,
        "chosen_cell": chosen,
        "selection_time_step": env.time_steps,
        "candidate_count": len(candidates),
        "candidate_mask_id": mask_id,
    }
    source.on_preview_token_minted(**metadata)
    source.on_preview_reserved(**metadata)
    commit = {**metadata, "commit_time_step": env.time_steps + 2}
    source.validate_bound_assignment_commit(**commit)
    source.on_bound_assignment_committed(**commit)
    source_audit = source.audit()
    selector_decision = {
        "proposal_id": metadata["proposal_id"],
        "valid": True,
        "block_label": block.label,
        "chosen_cell": chosen,
        "selection_time_step": env.time_steps,
        "commit_time_step": env.time_steps + 2,
        "candidate_count": len(candidates),
        "candidate_mask_id": mask_id,
    }
    selector_audit = {
        "assignment_source": repair.REPAIRED_METHOD_TO_SOURCE[
            repair.REPAIRED_2009_METHOD
        ],
        "assignment_source_version": repair.REPAIRED_METHOD_TO_VERSION[
            repair.REPAIRED_2009_METHOD
        ],
        "valid_assignment_count": 1,
        "reserved_commit_count": 1,
        "decisions": [selector_decision],
        "source_audit": source_audit,
    }
    option_audit = {
        "inbound_successes": 1,
        "reservation_bound_count": 1,
        "reservation_commit_count": 1,
        "reservation_execution_match_count": 1,
        "reservation_invalidation_count": 0,
        "bound_proposal_ids": [metadata["proposal_id"]],
        "committed_proposal_ids": [metadata["proposal_id"]],
    }
    return {
        "assignment_source": selector_audit["assignment_source"],
        "assignment_source_version": selector_audit[
            "assignment_source_version"
        ],
        "reservation_integrity": True,
        "delivery_count": 1,
        "selector_audit": selector_audit,
        "scheduler_audit": option_audit,
    }


class CapacityAwareRepairProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.sources = base.authenticate_sources(
            v23_source_dir=cls.root
            / "results/vcg-constrained-v2-3-gamma1-ablation-seed10-200ep",
            v11_control_dir=cls.root
            / "results/vcg-dense-v1-1-v2-2-panel-control-12instance",
            v22_source_dir=cls.root
            / "results/vcg-constrained-v2-2-development-seed10-200ep",
        )
        cls.matched = repair.authenticate_completed_matched_comparison(
            cls.root / "results/vcg-v2-3-matched-baselines-85k",
            sources=cls.sources,
        )

    def test_completed_source_is_pinned_and_new_grid_is_exactly_four_by_twelve(self):
        self.assertEqual(len(self.matched.rows), 156)
        contract = repair.build_repair_contract(
            sources=self.sources, matched=self.matched, device="cpu"
        )
        self.assertEqual(
            contract["expected_new_run_grid"]["total_rows"], 48
        )
        self.assertEqual(
            tuple(contract["new_repaired_methods"]), repair.REPAIRED_METHODS
        )
        self.assertFalse(contract["final_86xxx_panel_opened"])
        self.assertTrue(
            contract["immutable_matched_source"][
                "historical_source_time_hashes_preserved"
            ]
        )
        self.assertNotIn(
            TRACK_A_GA_ROLLING,
            set(contract["new_assignment_sources"].values()),
        )

    def test_capacity_source_audit_authenticates_queue_and_reservation(self):
        raw = _audited_raw()
        self.assertEqual(
            repair._capacity_audit_issues(
                raw,
                method=repair.REPAIRED_2009_METHOD,
                seed=85_000,
                expected_pickup_cell=(0, 0),
                expected_delivery_count=1,
            ),
            [],
        )
        source = raw["selector_audit"]["source_audit"]
        decision = source["decisions"][0]
        self.assertGreater(decision["deferred_count"], 0)
        self.assertEqual(
            decision["observed_arrived_pending_labels"],
            tuple((*decision["planned_labels"], *decision["deferred_labels"])),
        )
        self.assertFalse(decision["scheduler_defer_action_introduced"])

    def test_capacity_audit_rejects_hidden_overflow_future_access_and_bad_reservation(self):
        for mutation, expected_fragment in (
            (
                lambda raw: raw["selector_audit"]["source_audit"][
                    "decisions"
                ][0].__setitem__("deferred_labels", ()),
                "pending_partition",
            ),
            (
                lambda raw: raw["selector_audit"]["source_audit"].__setitem__(
                    "future_schedule_accessed", True
                ),
                "future_schedule_accessed",
            ),
            (
                lambda raw: raw["scheduler_audit"].__setitem__(
                    "committed_proposal_ids", ["wrong"]
                ),
                "committed_proposal_order_mismatch",
            ),
            (
                lambda raw: raw["selector_audit"]["source_audit"][
                    "decisions"
                ][0].__setitem__("best_cost_scalar", -1),
                "cost_scalar",
            ),
        ):
            with self.subTest(expected=expected_fragment):
                raw = _audited_raw()
                mutation(raw)
                issues = repair._capacity_audit_issues(
                    raw,
                    method=repair.REPAIRED_2009_METHOD,
                    seed=85_000,
                    expected_pickup_cell=(0, 0),
                    expected_delivery_count=1,
                )
                self.assertTrue(
                    any(expected_fragment in issue for issue in issues), issues
                )

    def test_capacity_audit_rejects_schema_numeric_time_cell_and_label_tamper(self):
        mutations = (
            (
                lambda raw: raw["selector_audit"]["source_audit"].__setitem__(
                    "unexpected", 1
                ),
                "source_audit_schema_keys",
            ),
            (
                lambda raw: raw["selector_audit"]["source_audit"].__setitem__(
                    "pending_order", "reverse"
                ),
                "source_pending_order_mismatch",
            ),
            (
                lambda raw: raw["selector_audit"]["source_audit"].__setitem__(
                    "observable_fields_used", ("due_step",)
                ),
                "source_observable_fields_mismatch",
            ),
            (
                lambda raw: raw["selector_audit"]["source_audit"].__setitem__(
                    "preview_contract", "tampered"
                ),
                "source_preview_contract_mismatch",
            ),
            (
                lambda raw: raw["selector_audit"]["source_audit"].__setitem__(
                    "optimizer_seed_offset_contract", "tampered"
                ),
                "source_optimizer_seed_contract_mismatch",
            ),
            (
                lambda raw: raw["selector_audit"]["source_audit"].__setitem__(
                    "minted_unreserved_preview_count", -7
                ),
                "source_invalid_nonnegative_count:minted_unreserved_preview_count",
            ),
            (
                lambda raw: raw["selector_audit"]["source_audit"].__setitem__(
                    "minted_unreserved_preview_count", 7
                ),
                "minted_unreserved_count_id_mismatch",
            ),
            (
                lambda raw: (
                    raw["selector_audit"]["source_audit"].__setitem__(
                        "minted_unreserved_preview_count", 1
                    ),
                    raw["selector_audit"]["source_audit"].__setitem__(
                        "minted_unreserved_proposal_ids",
                        (
                            raw["selector_audit"]["source_audit"][
                                "decisions"
                            ][0]["proposal_id"],
                        ),
                    ),
                ),
                "discarded_preview_id_also_committed",
            ),
            (
                lambda raw: raw["selector_audit"]["source_audit"].__setitem__(
                    "decision_count", 1.9
                ),
                "source_invalid_nonnegative_count:decision_count",
            ),
            (
                lambda raw: raw["selector_audit"]["source_audit"][
                    "decisions"
                ][0].__setitem__("unexpected", 1),
                "schema_keys",
            ),
            (
                lambda raw: raw["selector_audit"]["source_audit"][
                    "decisions"
                ][0]["best_cost_components"].__setitem__(
                    "route_steps", True
                ),
                "cost_component_type_or_sign",
            ),
            (
                lambda raw: raw["selector_audit"]["source_audit"][
                    "decisions"
                ][0].__setitem__("source_cell", (9, 9)),
                "source_not_frozen_pickup_cell",
            ),
            (
                lambda raw: raw["selector_audit"]["source_audit"][
                    "decisions"
                ][0].__setitem__("commit_time_step", 1),
                "commit_precedes_selection",
            ),
            (
                lambda raw: raw["selector_audit"]["source_audit"][
                    "decisions"
                ][0].__setitem__(
                    "observed_arrived_pending_labels", ("b0", "b0")
                ),
                "duplicate_observed_label",
            ),
        )
        for mutation, expected_fragment in mutations:
            with self.subTest(expected=expected_fragment):
                raw = _audited_raw()
                mutation(raw)
                issues = repair._capacity_audit_issues(
                    raw,
                    method=repair.REPAIRED_2009_METHOD,
                    seed=85_000,
                    expected_pickup_cell=(0, 0),
                    expected_delivery_count=1,
                )
                self.assertTrue(
                    any(expected_fragment in issue for issue in issues), issues
                )

    def test_final_panel_is_refused(self):
        contract = repair.build_repair_contract(
            sources=self.sources, matched=self.matched, device="cpu"
        )
        with self.assertRaises(repair.CapacityRepairComparisonError):
            repair._repair_input_contract(
                repair.REPAIRED_2009_METHOD,
                86_000,
                sources=self.sources,
                repair_contract=contract,
            )

    def test_resume_rejects_self_rehashed_method_or_source_tamper(self):
        seed = 85_000
        identities = {seed: _identity(seed)}
        input_contract = {
            "episode_instance_sha256": identities[seed][
                "episode_instance_sha256"
            ],
            "unit": True,
        }
        method = repair.REPAIRED_2009_METHOD
        failed = repair._failed_repaired_row(
            method, seed, identities, RuntimeError("unit failure")
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            repair._load_or_execute_repaired(
                output_dir=output,
                method=method,
                seed=seed,
                input_contract=input_contract,
                identities=identities,
                execute=True,
                executor=lambda: failed,
            )
            path = repair._repair_ledger_path(output, method, seed)
            original = json.loads(path.read_text())
            for field, bad in (
                ("method_id", repair.REPAIRED_DURATION_METHOD),
                ("assignment_source", "historical_or_wrong_source"),
                ("assignment_source_version", "wrong-version"),
            ):
                with self.subTest(field=field):
                    ledger = copy.deepcopy(original)
                    ledger["run"][field] = bad
                    ledger.pop("ledger_sha256")
                    ledger["ledger_sha256"] = repair._digest_json(ledger)
                    path.write_text(json.dumps(ledger))
                    with self.assertRaises(
                        repair.CapacityRepairComparisonError
                    ):
                        repair._load_or_execute_repaired(
                            output_dir=output,
                            method=method,
                            seed=seed,
                            input_contract=input_contract,
                            identities=identities,
                            execute=False,
                            executor=lambda: failed,
                        )
                    path.write_text(json.dumps(original))

    def test_ledger_tree_rejects_extra_protected_temp_directory_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            canonical = repair._repair_ledger_path(
                output, repair.REPAIRED_2009_METHOD, 85_000
            )
            canonical.parent.mkdir(parents=True)
            canonical.write_text("{}")
            self.assertEqual(
                repair._scan_repair_ledger_tree(
                    output, require_complete=False
                ),
                (canonical.resolve(),),
            )
            with self.assertRaises(repair.CapacityRepairComparisonError):
                repair._scan_repair_ledger_tree(output, require_complete=True)

            bad_paths = (
                canonical.parent / "seed-86000.json",
                canonical.parent / ".seed-85001.json.tmp",
            )
            for bad in bad_paths:
                with self.subTest(path=bad.name):
                    bad.write_text("{}")
                    with self.assertRaises(
                        repair.CapacityRepairComparisonError
                    ):
                        repair._scan_repair_ledger_tree(
                            output, require_complete=False
                        )
                    bad.unlink()

            extra_directory = output / "run-ledger" / "unknown-method"
            extra_directory.mkdir()
            with self.assertRaises(repair.CapacityRepairComparisonError):
                repair._scan_repair_ledger_tree(
                    output, require_complete=False
                )
            extra_directory.rmdir()

            symlink = canonical.parent / "seed-85001.json"
            try:
                symlink.symlink_to(canonical)
            except (NotImplementedError, OSError):
                pass
            else:
                with self.assertRaises(
                    repair.CapacityRepairComparisonError
                ):
                    repair._scan_repair_ledger_tree(
                        output, require_complete=False
                    )

    def test_ledger_tree_accepts_only_exact_complete_48_regular_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for method in repair.REPAIRED_METHODS:
                for seed in repair.PANEL_SEEDS:
                    path = repair._repair_ledger_path(output, method, seed)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}")
            observed = repair._scan_repair_ledger_tree(
                output, require_complete=True
            )
            self.assertEqual(len(observed), 48)
            self.assertTrue(all(path.is_file() for path in observed))

    def test_ledger_tree_rejects_dangling_root_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            ledger_root = output / "run-ledger"
            try:
                ledger_root.symlink_to(Path(directory) / "does-not-exist")
            except (NotImplementedError, OSError):
                self.skipTest("symlinks unavailable")
            with self.assertRaises(repair.CapacityRepairComparisonError):
                repair._scan_repair_ledger_tree(
                    output, require_complete=False
                )

    def test_source_snapshot_recheck_detects_content_and_symlink_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.json"
            path.write_text("original")
            paths = {"source": path}
            hashes = {"source": repair._sha256_file(path)}
            self.assertEqual(
                repair._recheck_path_snapshot(
                    paths, hashes, label="unit source"
                ),
                hashes,
            )
            path.write_text("changed")
            with self.assertRaises(repair.CapacityRepairComparisonError):
                repair._recheck_path_snapshot(
                    paths, hashes, label="unit source"
                )
            path.unlink()
            target = root / "target.json"
            target.write_text("original")
            try:
                path.symlink_to(target)
            except (NotImplementedError, OSError):
                pass
            else:
                with self.assertRaises(
                    repair.CapacityRepairComparisonError
                ):
                    repair._recheck_path_snapshot(
                        paths, hashes, label="unit source"
                    )

    def test_whole_method_suppression_uses_no_complete_case_filtering(self):
        identities = {seed: _identity(seed) for seed in repair.PANEL_SEEDS}
        rows = tuple(
            repair._failed_repaired_row(
                repair.REPAIRED_2009_METHOD,
                seed,
                identities,
                RuntimeError("one unsafe row"),
            )
            for seed in repair.PANEL_SEEDS
        )
        summary = repair._summarize_repaired_method(
            repair.REPAIRED_2009_METHOD, rows, identities
        )
        self.assertFalse(summary["whole_method_numeric_eligible"])
        self.assertIsNone(summary["metrics"])
        self.assertFalse(summary["complete_case_filtering_used"])


if __name__ == "__main__":
    unittest.main()
