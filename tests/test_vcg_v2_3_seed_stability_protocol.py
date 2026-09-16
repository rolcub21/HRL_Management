import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import train_vcg_constrained_v2_3 as base
import train_vcg_constrained_v2_3_stability as trainer
import vcg_v2_3_seed_stability as workflow
from vcg_v2_3_seed_stability_protocol import (
    MODEL_SEEDS,
    PANEL_SEEDS,
    SEED_PROFILES,
    StabilityProtocolError,
    authenticate_freeze_spec,
    authenticate_repair_v2,
    expected_freeze_spec,
    load_json,
    verify_self_hash,
)


def binding():
    return trainer.StabilityRunBinding(
        freeze_spec_sha256="1" * 64,
        run_manifest_sha256="2" * 64,
        repair_v2_contract_sha256="3" * 64,
        repair_v2_report_sha256="4" * 64,
        implementation_source_sha256={"source.py": "5" * 64},
    )


class FrozenProfileTests(unittest.TestCase):
    def test_frozen_source_scope_includes_transitive_algorithm_dependencies(self):
        paths = workflow._source_paths()
        for name in (
            "train_vcg_constrained_v2.py",
            "train_vcg_constrained_v2_1.py",
            "viability_graph_constrained_v2.py",
            "vcg_objective_audit.py",
            "PSLAP/viability.py",
            "PSLAP/viability_candidates_hold_v2.py",
            "example/small_rooms_env.py",
            "GA/helper_functions.py",
        ):
            self.assertIn(name, paths)
            self.assertTrue(paths[name].is_file())
        self.assertEqual(len(paths), 36)

    def test_exact_seed_and_rng_namespaces(self):
        expected = {
            11: (61_001_000, 61_001_199, 610_001_000, 610_001_199, 610_101_010),
            12: (61_002_000, 61_002_199, 610_002_000, 610_002_199, 610_102_010),
            13: (61_003_000, 61_003_199, 610_003_000, 610_003_199, 610_103_010),
        }
        self.assertEqual(tuple(SEED_PROFILES), MODEL_SEEDS)
        for seed, values in expected.items():
            profile = SEED_PROFILES[seed]
            self.assertEqual(
                (
                    profile.train_seeds[0], profile.train_seeds[-1],
                    profile.behavior_rng_seeds[0], profile.behavior_rng_seeds[-1],
                    profile.replay_rng_seed,
                ),
                values,
            )

    def test_reviewed_freeze_spec_is_exact_and_self_hashed(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "experiments/vcg_v2_3_seed_stability_85k/freeze-spec.json"
        )
        self.assertEqual(
            json.dumps(authenticate_freeze_spec(path), sort_keys=True),
            json.dumps(expected_freeze_spec(), sort_keys=True),
        )

    def test_seed_contracts_are_exact_and_independently_reconstructed(self):
        for seed in MODEL_SEEDS:
            profile = SEED_PROFILES[seed]
            contract = trainer.build_stability_training_contract(
                seed, binding=binding(), output_dir=Path("unused"), device="cuda"
            )
            self.assertEqual(contract["model_seed"], seed)
            self.assertEqual(contract["train_seed_base"], profile.train_seed_base)
            self.assertEqual(contract["replay_rng_seed"], profile.replay_rng_seed)
            self.assertEqual(
                contract["training_policy_rng_formula"],
                f"{profile.behavior_rng_base} + episode_number - 1",
            )
            self.assertFalse(contract["training_instances_reused_after_v2_2"])
            authenticated, path, digest = trainer.authenticate_seed_training_contract(
                seed, contract, binding=binding()
            )
            self.assertIsNone(path)
            self.assertEqual(authenticated, contract)
            self.assertEqual(digest, contract["contract_sha256"])

    def test_profile_activation_restores_seed10_module_behavior(self):
        before_contract = base._independently_reconstructed_contract(device="cpu")
        before_globals = (
            base.DEFAULT_MODEL_SEED,
            base.FRESH_TRAIN_SEED_BASE,
            base.TRAINING_POLICY_RNG_BASE,
            base.REPLAY_RNG_SEED,
            base.TRAINING_PROTOCOL,
            base.CHECKPOINT_FAMILY,
        )
        with trainer.activated_seed_profile(SEED_PROFILES[12], binding()):
            self.assertEqual(base.DEFAULT_MODEL_SEED, 12)
            self.assertEqual(base.REPLAY_RNG_SEED, 610_102_010)
        self.assertEqual(
            (
                base.DEFAULT_MODEL_SEED,
                base.FRESH_TRAIN_SEED_BASE,
                base.TRAINING_POLICY_RNG_BASE,
                base.REPLAY_RNG_SEED,
                base.TRAINING_PROTOCOL,
                base.CHECKPOINT_FAMILY,
            ),
            before_globals,
        )
        self.assertEqual(
            base._independently_reconstructed_contract(device="cpu"),
            before_contract,
        )

    def test_fresh_process_core_import_is_restored_to_seed10(self):
        program = r'''
import sys
import train_vcg_constrained_v2_3 as base
import train_vcg_constrained_v2_3_stability as stability
assert "viability_graph_constrained_v2_3" not in sys.modules
binding = stability.StabilityRunBinding(
    "1" * 64, "2" * 64, "3" * 64, "4" * 64,
    {"source.py": "5" * 64},
)
with stability.activated_seed_profile(stability.SEED_PROFILES[12], binding):
    import viability_graph_constrained_v2_3 as core
    assert core.REPLAY_RNG_SEED == 610_102_010
import viability_graph_constrained_v2_3 as core
assert base.REPLAY_RNG_SEED == 610_100_010
assert core.REPLAY_RNG_SEED == 610_100_010
assert core.CHECKPOINT_FAMILY == base.CHECKPOINT_FAMILY
'''
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_only_frozen_model_seeds_are_accepted(self):
        for rejected in (10, 11.0, "11", True):
            with self.subTest(rejected=rejected), self.assertRaises(
                StabilityProtocolError
            ):
                trainer.build_stability_training_contract(
                    rejected, binding=binding(), output_dir=Path("unused")
                )

    def test_completed_repair_v2_exact_tree_authenticates(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "results/vcg-v2-3-capacity-aware-ga-repair-v2-85k"
        )
        repair = authenticate_repair_v2(root)
        self.assertEqual(len(repair["repair_ledger_raw_sha256"]), 48)

    def test_repair_v2_rejects_unbound_top_level_file_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name in (
                "repair-contract.json", "preflight.json", "expanded-report.json",
                "expanded-runs.csv", "expanded-audit.json",
            ):
                (root / name).write_text("{}\n", encoding="utf-8")
            (root / "run-ledger").mkdir()
            (root / "unbound.tmp").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(
                StabilityProtocolError, "root file set"
            ):
                authenticate_repair_v2(root)


class AggregationTests(unittest.TestCase):
    @staticmethod
    def rows():
        rows = []
        for instance_index, instance_seed in enumerate(PANEL_SEEDS):
            for rng_index in range(4):
                value = float(instance_index + rng_index)
                rows.append(
                    {
                        "instance_seed": instance_seed,
                        "instance_index": instance_index,
                        "policy_rng_index": rng_index,
                        "policy_rng_seed": 620_000_000 + 4 * instance_index + rng_index,
                        "strict_method_success": True,
                        "required_deliveries": 8,
                        "delivery_count": 8,
                        "method_failure_reason": None,
                        "all_selected_candidates_exact_safe": True,
                        "illegal_drops": 0,
                        "fallbacks": 0,
                        "witness_mismatches": 0,
                        "evaluation_learning": False,
                        "training_agent_unchanged": True,
                        "validation_batch_state_unchanged": True,
                        "fresh_evaluation_clone": True,
                        "stochastic_selection_only": True,
                        "map_selection_used": False,
                        "dense_return": 100.0 - value,
                        "mean_absolute_error": value,
                        "mean_tardiness": value / 2.0,
                        "mean_earliness": value / 2.0,
                        "steps": 200 + value,
                        "within_target_window_rate": 0.5,
                        "physical_rehandles": rng_index % 2,
                    }
                )
        return rows

    def test_equal_rng_then_equal_instance_summary(self):
        summary = workflow.summarize_selected_rows(11, self.rows())
        self.assertTrue(summary["whole_seed_strict_safe_complete"])
        self.assertEqual(summary["row_count"], 48)
        self.assertEqual(summary["instance_count"], 12)
        self.assertAlmostEqual(
            summary["metrics"]["physical_rehandles_per_100_required_deliveries"],
            6.25,
        )
        self.assertAlmostEqual(summary["metrics"]["mean_absolute_error"], 7.0)

    def test_any_unsafe_row_fails_whole_seed(self):
        rows = self.rows()
        rows[0]["strict_method_success"] = False
        summary = workflow.summarize_selected_rows(11, rows)
        self.assertFalse(summary["whole_seed_strict_safe_complete"])
        self.assertEqual(len(summary["safety_issues"]), 1)

    def test_duplicate_or_incomplete_grid_fails_closed(self):
        rows = self.rows()
        rows[-1] = copy.deepcopy(rows[0])
        with self.assertRaises(StabilityProtocolError):
            workflow.summarize_selected_rows(11, rows)

    def test_selected_row_identity_does_not_coerce_numeric_strings(self):
        rows = self.rows()
        rows[0]["instance_seed"] = str(rows[0]["instance_seed"])
        with self.assertRaisesRegex(StabilityProtocolError, "identity types"):
            workflow.summarize_selected_rows(11, rows)

    def test_metric_ranking_uses_tie_aware_dense_ranks(self):
        methods = (
            {"method_id": "b", "metrics": {"m": 2.0}},
            {"method_id": "a", "metrics": {"m": 2.0}},
            {"method_id": "c", "metrics": {"m": 1.0}},
            {"method_id": "d", "metrics": {"m": 0.0}},
        )
        ranking = workflow._dense_metric_ranking(
            methods, metric="m", higher_is_better=True,
        )
        self.assertEqual(
            [(item["method_id"], item["rank"]) for item in ranking],
            [("a", 1), ("b", 1), ("c", 2), ("d", 3)],
        )

    def test_output_tree_rejects_extra_and_symlink_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / workflow.RUN_MANIFEST_NAME).write_text("{}\n", encoding="utf-8")
            (root / "unexpected.tmp").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(StabilityProtocolError, "unexpected entries"):
                workflow._validate_output_tree(root)
            (root / "unexpected.tmp").unlink()
            (root / "seed-11").symlink_to(root / "escaped", target_is_directory=True)
            with self.assertRaisesRegex(StabilityProtocolError, "forbidden symlink"):
                workflow._validate_output_tree(root)

    def test_seed_tree_rejects_extra_and_candidate_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            seed_dir = Path(temporary).resolve()
            for name in workflow._SEED_ROOT_REQUIRED_FILES:
                (seed_dir / name).write_bytes(b"x")
            candidate_dir = seed_dir / "candidate-look-checkpoints"
            ledger_dir = seed_dir / "validation-ledger"
            candidate_dir.mkdir()
            ledger_dir.mkdir()
            for episode in workflow.CANDIDATE_LOOKS:
                (candidate_dir / f"episode-{episode:04d}-model-only.pth").write_bytes(b"x")
            for episode in range(20, 201, 20):
                (ledger_dir / f"episode-{episode:04d}.json").write_bytes(b"x")
            self.assertEqual(
                len(workflow._seed_tree_raw_sha256(seed_dir, allow_completion=False)),
                22,
            )
            (seed_dir / "scratch.tmp").write_bytes(b"x")
            with self.assertRaisesRegex(StabilityProtocolError, "unexpected entries"):
                workflow._seed_tree_raw_sha256(seed_dir, allow_completion=False)
            (seed_dir / "scratch.tmp").unlink()
            target = candidate_dir / "episode-0080-model-only.pth"
            target.unlink()
            target.symlink_to(candidate_dir / "episode-0100-model-only.pth")
            with self.assertRaisesRegex(StabilityProtocolError, "non-symlink"):
                workflow._seed_tree_raw_sha256(seed_dir, allow_completion=False)

    def test_validation_ledger_rejects_boolean_episode_without_coercion(self):
        with tempfile.TemporaryDirectory() as temporary:
            seed_dir = Path(temporary).resolve()
            root = seed_dir / "validation-ledger"
            root.mkdir()
            for episode in range(20, 201, 20):
                ledger = {
                    "schema_version": 1,
                    "validation_protocol": base.VALIDATION_PROTOCOL,
                    "checkpoint_episode": True if episode == 20 else episode,
                    "validated_lambda": 0.0,
                    "schedule_state": {},
                    "row_count": 48,
                    "complete_case_filtering_used": False,
                    "validation_batch_audit": {},
                    "rows": [{} for _ in range(48)],
                }
                ledger["ledger_sha256"] = workflow.digest_json(ledger)
                workflow.atomic_json(
                    root / f"episode-{episode:04d}.json", ledger,
                )
            with self.assertRaisesRegex(StabilityProtocolError, "episode is invalid"):
                workflow._validation_ledger_manifest(seed_dir)

    def test_run_seed_does_not_complete_after_post_training_trust_failure(self):
        run_manifest = {"manifest_sha256": "2" * 64}
        repair = {
            "root": Path("/tmp/fake-repair"),
            "raw_sha256": {}, "contract_sha256": "3" * 64,
            "report_sha256": "4" * 64, "audit_sha256": "5" * 64,
            "repair_ledger_raw_sha256": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()

            def fake_long_runner(_seed, *, output_dir, **_kwargs):
                output_dir.mkdir()

            with mock.patch.object(
                workflow, "authenticate_run_manifest",
                return_value=(run_manifest, binding(), repair),
            ), mock.patch.object(
                workflow, "run_frozen_stability_seed", side_effect=fake_long_runner,
            ), mock.patch.object(
                workflow, "_reauthenticate_run_trust",
                side_effect=StabilityProtocolError("source drift"),
            ), mock.patch.object(workflow, "reconstruct_seed_completion") as reconstruct:
                with self.assertRaisesRegex(StabilityProtocolError, "source drift"):
                    workflow.run_seed(11, output_dir=output)
                reconstruct.assert_not_called()
                self.assertFalse(
                    (output / "seed-11" / workflow.COMPLETION_NAME).exists()
                )

    def test_prepare_never_invokes_long_runner(self):
        freeze = expected_freeze_spec()
        repair = {
            "root": Path("/tmp/fake-repair"),
            "raw_sha256": {"contract": "a" * 64},
            "contract_sha256": "b" * 64,
            "report_sha256": "c" * 64,
            "audit_sha256": "d" * 64,
            "repair_ledger_raw_sha256": {"method:85000": "9" * 64},
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            with mock.patch.object(workflow, "authenticate_freeze_spec", return_value=freeze), mock.patch.object(
                workflow, "authenticate_repair_v2", return_value=repair
            ), mock.patch.object(
                workflow, "sha256_file", return_value="e" * 64
            ), mock.patch.object(
                workflow, "_source_hashes", return_value={"source.py": "f" * 64}
            ), mock.patch.object(
                workflow, "run_frozen_stability_seed"
            ) as long_runner:
                workflow.prepare_protocol(
                    output_dir=output,
                    freeze_spec_path=Path("freeze.json"),
                    repair_dir=Path("repair"),
                )
                long_runner.assert_not_called()

    def test_completed_ineligible_seed_writes_failed_nonfiltered_analysis(self):
        rows_by_seed = {
            seed: copy.deepcopy(self.rows()) for seed in MODEL_SEEDS[:2]
        }
        selected_by_seed = {}
        completions = {}
        for seed in MODEL_SEEDS:
            eligible = seed != 13
            completions[seed] = {
                "schema_version": 1,
                "protocol": "synthetic",
                "status": "complete",
                "model_seed": seed,
                "development_candidate_eligible": eligible,
                "seed_tree_raw_sha256": {},
                "completion_sha256": str(seed) * 64,
            }
            if eligible:
                metrics = workflow.summarize_selected_rows(
                    seed, rows_by_seed[seed]
                )["metrics"]
                selected_by_seed[seed] = {
                    "selected_episode": 160,
                    "selected_diagnostic": {
                        "validation_ledger": {"rows": rows_by_seed[seed]}
                    },
                    "validation_summary": {
                        "mean_absolute_error": metrics["mean_absolute_error"],
                        "expected_physical_rehandles_per_100_required_deliveries": metrics[
                            "physical_rehandles_per_100_required_deliveries"
                        ],
                        "mean_dense_return": metrics[
                            "mean_dense_objective_return"
                        ],
                    },
                }
        source_hashes = {"bound-source.py": "a" * 64}
        run_manifest = {
            "manifest_sha256": "b" * 64,
            "implementation_source_sha256": source_hashes,
        }
        repair = {
            "root": Path("/tmp/fake-repair"),
            "contract_sha256": "9" * 64,
            "report_sha256": "c" * 64,
            "audit_sha256": "8" * 64,
            "raw_sha256": {"report": "d" * 64},
            "repair_ledger_raw_sha256": {"method:85000": "7" * 64},
            "baseline_summaries": (
                {
                    "method_id": "safe-baseline",
                    "whole_method_numeric_eligible": True,
                    "metrics": {
                        "mean_dense_objective_return": 0.0,
                        "mean_absolute_error": 50.0,
                        "physical_rehandles_per_100_required_deliveries": 0.0,
                        "mean_steps": 100.0,
                    },
                },
            ),
        }

        def authenticate_completion(seed, **_kwargs):
            return completions[seed], selected_by_seed.get(seed)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            for seed in MODEL_SEEDS:
                seed_dir = output / f"seed-{seed}"
                seed_dir.mkdir()
                (seed_dir / workflow.COMPLETION_NAME).write_text(
                    "{}\n", encoding="utf-8"
                )
            with mock.patch.object(
                workflow,
                "authenticate_run_manifest",
                return_value=(run_manifest, object(), repair),
            ) as authenticate_run, mock.patch.object(
                workflow,
                "authenticate_seed_completion",
                side_effect=authenticate_completion,
            ) as authenticate_completion_mock, mock.patch.object(
                workflow, "_seed_tree_raw_sha256", return_value={},
            ), mock.patch.object(
                workflow, "_source_hashes", return_value=source_hashes,
            ), mock.patch.object(
                workflow, "run_frozen_stability_seed"
            ) as long_runner:
                report = workflow.analyze(output_dir=output)
                long_runner.assert_not_called()
                self.assertEqual(authenticate_run.call_count, 3)
                self.assertEqual(authenticate_completion_mock.call_count, 6)

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["completed_but_ineligible_model_seeds"], (13,))
            self.assertEqual(report["eligible_training_seed_count"], 2)
            self.assertEqual(report["strict_safe_complete_selected_row_count"], 96)
            self.assertFalse(report["equal_seed_aggregate"]["available"])
            self.assertIsNone(report["equal_seed_aggregate"]["metrics"])
            self.assertIsNone(
                report["equal_seed_aggregate"][
                    "dominators_on_MAE_and_total_physical_rehandles"
                ]
            )
            self.assertIsNone(report["metric_rankings"])
            self.assertIsNone(report["non_dominated_individual_seed_count"])
            self.assertTrue(
                all(item["non_dominated"] is None for item in report["individual_seed_screen"])
            )
            self.assertFalse(report["criterion"]["three_of_three_eligible"])
            self.assertFalse(report["criterion"]["aggregate_non_dominated"])
            self.assertFalse(report["criterion"]["passed"])
            self.assertFalse(report["complete_case_filtering_used"])
            self.assertTrue(report["incomplete_three_seed_estimand_is_never_aggregated"])

            report_path = output / workflow.REPORT_NAME
            audit_path = output / workflow.AUDIT_NAME
            stored_report = load_json(report_path, name="synthetic stability report")
            verify_self_hash(stored_report, "report_sha256", name="synthetic report")
            audit = load_json(audit_path, name="synthetic stability audit")
            verify_self_hash(audit, "audit_sha256", name="synthetic audit")
            self.assertEqual(audit["completed_but_ineligible_model_seeds"], [13])
            self.assertFalse(audit["equal_seed_estimand_available"])
            self.assertFalse(audit["all_144_rows_authenticated"])
            self.assertEqual(audit["environment_rollouts_executed_during_analysis"], 0)

    def test_safe_completed_no_best_is_derived_from_all_seven_ledgers(self):
        from tests.test_train_vcg_constrained_v2_3_protocol import _FakeRuntime

        class SafeButTimingIneligibleRuntime(_FakeRuntime):
            def run_episode(self, **kwargs):
                row = super().run_episode(**kwargs)
                if not kwargs["training"]:
                    row["mean_absolute_error"] = 100.0
                return row

        frozen_binding = binding()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            seed_dir = root / "seed-11"
            profile = SEED_PROFILES[11]
            with trainer.activated_seed_profile(profile, frozen_binding):
                args = trainer._profile_args(
                    profile, output_dir=seed_dir, device="cpu"
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = base.run_development_calibration(
                        args,
                        runtime=SafeButTimingIneligibleRuntime(args),
                    )
            self.assertEqual(summary["status"], "complete")
            self.assertFalse(summary["development_candidate_eligible"])
            self.assertIsNone(summary["best_validation"])
            self.assertIsNone(summary["best_development_candidate"])
            self.assertFalse((seed_dir / "best-development-candidate.pth").exists())

            run_manifest = {
                "output_root": str(root),
                "manifest_sha256": frozen_binding.run_manifest_sha256,
            }
            completion, selected = workflow.reconstruct_seed_completion(
                11,
                seed_dir=seed_dir,
                run_manifest=run_manifest,
                binding=frozen_binding,
            )
            self.assertIsNone(selected)
            self.assertFalse(completion["development_candidate_eligible"])
            self.assertIsNone(completion["selected_best"])
            self.assertEqual(len(completion["candidate_look_artifacts"]), 7)
            self.assertTrue(
                all(
                    not item["validation_development_candidate_eligible"]
                    for item in completion["candidate_look_artifacts"]
                )
            )
            self.assertTrue(
                all(
                    item["validation_ledger_sha256"]
                    for item in completion["candidate_look_artifacts"]
                )
            )
            workflow.atomic_json(seed_dir / workflow.COMPLETION_NAME, completion)
            observed, reloaded_selected = workflow.authenticate_seed_completion(
                11,
                output_dir=root,
                run_manifest=run_manifest,
                binding=frozen_binding,
            )
            self.assertEqual(
                json.dumps(observed, sort_keys=True),
                json.dumps(completion, sort_keys=True),
            )
            self.assertIsNone(reloaded_selected)

            # Even a self-consistent rewrite cannot replace an artifact after
            # completion: the loader must use the raw hashes pinned in that
            # completion, not recompute a new expected hash from current bytes.
            manifest_path = seed_dir / "candidate-look-checkpoint-manifest.json"
            rewritten = load_json(manifest_path, name="candidate manifest")
            rewritten["complete"] = False
            rewritten.pop("manifest_sha256")
            rewritten["manifest_sha256"] = workflow.digest_json(rewritten)
            workflow.atomic_json(manifest_path, rewritten)
            with self.assertRaisesRegex(StabilityProtocolError, "pinned seed artifact tree"):
                workflow.authenticate_seed_completion(
                    11,
                    output_dir=root,
                    run_manifest=run_manifest,
                    binding=frozen_binding,
                )


if __name__ == "__main__":
    unittest.main()
