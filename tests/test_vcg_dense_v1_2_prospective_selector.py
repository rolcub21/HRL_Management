import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
import select_vcg_dense_v1_2_prospective as selector
import train_vcg_dense_proper as base
import train_vcg_dense_v1_2_prospective as trainer


def _summary(mae, relocation, dense=100.0, *, strict=True):
    physical_relocations = int(round(float(relocation) * 2.4))
    return {
        "episodes": 30,
        "strict_full": bool(strict),
        "strict_method_success_rate": 1.0 if strict else 0.9,
        "completion_rate": 1.0 if strict else 0.9,
        "delivery_count": 240,
        "mean_absolute_error": float(mae),
        "physical_storage_relocations": physical_relocations,
        "physical_storage_relocations_per_100_deliveries": float(relocation),
        "relocations": physical_relocations,
        "relocations_per_100_deliveries": float(relocation),
        "mean_dense_rescored_return": float(dense),
        "mean_tardiness": 1.0,
        "within_target_window_rate": 0.9,
        "mean_steps": 170.0,
    }


def _record(episode, score):
    return {
        "checkpoint_episode": int(episode),
        "selection_score": tuple(float(value) for value in score),
        "selection_score_fields": (
            "strict_success_rate",
            "completion_rate",
            "selected_objective_return_per_block",
            "negative_mean_absolute_error",
            "negative_relocations_per_100_deliveries",
            "negative_checkpoint_episode",
        ),
        "deployment_eligible": True,
        "summary": {
            "strict_method_success_rate": 1.0,
            "completion_rate": 1.0,
            "mean_dense_rescored_return": 160.0,
            "mean_absolute_error": 10.0,
            "relocations_per_100_deliveries": 15.0,
        },
        "runs": (),
    }


class ProspectiveSelectorProtocolTests(unittest.TestCase):
    def test_frozen_namespaces_are_disjoint_and_validation_does_not_open_panels(self):
        self.assertEqual(selector.MODEL_SEEDS, (3, 4, 5))
        self.assertEqual(selector.PANEL_A_SEEDS, tuple(range(83_000, 83_030)))
        self.assertEqual(selector.PANEL_B_SEEDS, tuple(range(83_030, 83_060)))
        self.assertFalse(set(selector.PANEL_A_SEEDS) & set(selector.PANEL_B_SEEDS))
        self.assertEqual(selector.NAMESPACE_AUDIT["repo_and_results_literal_matches"], 0)
        with patch.object(
            selector, "_make_env", side_effect=AssertionError("must not open panel")
        ):
            selector._validate_protocol()

    def test_panel_a_rank_uses_physical_relocation_then_dense_mae_episode(self):
        summaries = {
            25: _summary(10.0, 20.0, 120.0),
            50: _summary(11.5, 10.0, 100.0),
            75: _summary(11.5, 10.0, 110.0),
            100: _summary(12.1, 0.0, 999.0),
            125: _summary(9.0, 5.0, 999.0, strict=False),
        }
        rank = selector._rank_panel_a(summaries, 25)
        self.assertEqual(
            rank["ranked_nonreference_candidate_episodes"], (75, 50)
        )
        self.assertEqual(
            rank["selection_order_with_mandatory_fallback"], (75, 50, 25)
        )
        self.assertEqual(
            rank["candidate_table"][0][
                "physical_storage_relocations_per_100_deliveries"
            ],
            20.0,
        )

    def test_panel_b_gate_requires_all_three_conditions_and_is_sequential(self):
        rank = {
            "reference_episode": 25,
            "selection_order_with_mandatory_fallback": (50, 75, 100, 25),
        }
        summaries = {
            25: _summary(10.0, 20.0),
            50: _summary(12.1, 5.0),       # MAE gate fails
            75: _summary(11.0, 20.0),      # positive saving gate fails
            100: _summary(11.9, 19.0),     # first complete pass
        }
        result = selector._gate_panel_b(rank, summaries)
        self.assertEqual(result["chosen_episode"], 100)
        self.assertTrue(result["selected_nonreference"])
        self.assertEqual(
            [trial["accepted"] for trial in result["panel_B_trials"]],
            [False, False, True],
        )

    def test_panel_b_mandatory_reference_fallback_needs_no_relocation_gain(self):
        rank = {
            "reference_episode": 25,
            "selection_order_with_mandatory_fallback": (50, 25),
        }
        result = selector._gate_panel_b(
            rank,
            {25: _summary(10.0, 20.0), 50: _summary(11.0, 21.0)},
        )
        self.assertEqual(result["chosen_episode"], 25)
        self.assertFalse(result["selected_nonreference"])
        self.assertEqual(
            result["panel_B_trials"][-1]["reason"],
            "mandatory_authenticated_v1_1_reference_fallback",
        )

    def test_strict_full_requires_canonical_partition_not_false_obstructive_alias(self):
        row = {
            "strict_method_success": 1.0,
            "completion_rate": 1.0,
            "delivery_count": 8,
            "contention_metric_schema_version": (
                "physical_storage_relocation_decomposition_v1"
            ),
            "physical_storage_relocations": 2,
            "target_bound_obstruction_clearances": 0,
            "standalone_reconfigurations": 2,
            "standalone_with_direct_delivery_available": 1,
            "standalone_without_direct_delivery_available": 1,
            "directly_deliverable_self_reconfigurations": 1,
            "relocations": 2,
            "obstructive_moves": 0,
        }
        self.assertTrue(selector._strict_full((row,), 1))
        self.assertFalse(
            selector._strict_full(
                ({**row, "obstructive_moves": 2},), 1
            )
        )
        self.assertFalse(
            selector._strict_full(
                ({**row, "standalone_without_direct_delivery_available": 0},),
                1,
            )
        )

    def test_unsafe_snapshot_with_undefined_timing_is_retained_but_ineligible(self):
        reference = _summary(10.0, 20.0)
        unsafe = _summary(0.0, 0.0, strict=False)
        unsafe["mean_absolute_error"] = None
        unsafe["mean_dense_rescored_return"] = None
        rank = selector._rank_panel_a({25: reference, 50: unsafe}, 25)
        self.assertEqual(rank["ranked_nonreference_candidate_episodes"], ())
        row = next(
            item for item in rank["candidate_table"]
            if item["checkpoint_episode"] == 50
        )
        self.assertFalse(row["panel_A_eligible"])
        self.assertIsNone(row["mae_cost_vs_reference"])

    def test_panel_b_execution_plan_contains_every_snapshot_not_only_a_eligible(self):
        arms = tuple(
            selector.SnapshotArm(
                model_seed=seed,
                episode=episode,
                method_id=f"m-{seed}-{episode}",
                path=Path(f"/tmp/m-{seed}-{episode}.pth"),
                sha256="a" * 64,
                policy_digest="b" * 64,
                payload={},
                validation_record={},
            )
            for seed in selector.MODEL_SEEDS
            for episode in selector.SNAPSHOT_EPISODES
        )
        plan = selector._execution_plan(arms, "B")
        self.assertEqual(len(plan), 3 * 20 * 30)
        self.assertEqual(len(set(plan)), len(plan))

    def test_transactional_ledger_resumes_once_and_rejects_contract_drift(self):
        arm = selector.SnapshotArm(
            model_seed=3,
            episode=25,
            method_id="fixture-arm",
            path=Path("/tmp/fixture.pth"),
            sha256="a" * 64,
            policy_digest="b" * 64,
            payload={},
            validation_record={},
        )
        row = {
            "method_id": arm.method_id,
            "instance_seed": 83_000,
            "instance_id": "instance",
            "schedule_id": "schedule",
            "panel": "A",
        }
        contract = {
            "instance_id": "instance",
            "schedule_id": "schedule",
            "checkpoint_sha256": arm.sha256,
        }
        calls = []
        with TemporaryDirectory() as directory:
            output = Path(directory)
            first, _ = selector._load_or_execute_run(
                output_dir=output,
                panel="A",
                arm=arm,
                seed=83_000,
                input_contract=contract,
                execute=lambda: calls.append(1) or row,
            )
            second, _ = selector._load_or_execute_run(
                output_dir=output,
                panel="A",
                arm=arm,
                seed=83_000,
                input_contract=contract,
                execute=lambda: calls.append(2) or row,
            )
            self.assertEqual(first, second)
            self.assertEqual(calls, [1])
            drift = dict(contract, checkpoint_sha256="c" * 64)
            with self.assertRaisesRegex(ValueError, "input contract"):
                selector._load_or_execute_run(
                    output_dir=output,
                    panel="A",
                    arm=arm,
                    seed=83_000,
                    input_contract=drift,
                    execute=lambda: row,
                )


class ProspectiveSelectorTrainerIntegrationTests(unittest.TestCase):
    def _write_two_snapshot_bundle(self, root: Path):
        runtime = trainer._build_parser().parse_args(
            ("--output-dir", str(root), "--model-seed", "3", "--device", "cpu")
        )
        prospective_contract, base_contract, _ = trainer._prospective_contract(runtime)
        contract_document = {
            "prospective_training_protocol": trainer.PROSPECTIVE_TRAINING_PROTOCOL,
            "prospective_method_version": trainer.PROSPECTIVE_METHOD_VERSION,
            "prospective_training_contract_sha256": audit._contract_hash(
                prospective_contract
            ),
            "contract": prospective_contract,
        }
        root.mkdir(parents=True)
        audit._atomic_json_save(
            contract_document, root / trainer.CONTRACT_FILENAME
        )
        source = {
            "checkpoint_family": "episodic_viability_graph_hierarchy_v1",
            "training_protocol": base.TRAINING_PROTOCOL,
            "method_version": base.METHOD_VERSION,
            "model_seed": 3,
            "completed_training_episodes": 500,
            "protocol_training_complete": True,
            "timing_objective": "dense_piecewise_v1",
            "timing_objective_spec": base.FROZEN_OBJECTIVE_SPEC.to_dict(),
            "gamma": base.FROZEN_GAMMA,
            "resume_contract": base_contract,
            "config": base_contract["graph_config"],
            "environment": base_contract["environment"],
            "viability_search": base_contract["search_config"],
            "liveness_rule": base_contract["liveness_rule"],
            "future_schedule_visible_to_policy": False,
            "exact_full": True,
            "exact_safe_mask_authoritative": True,
            "baseline_teacher": False,
            "viability_critic_enabled": False,
            "baseline_policy_query": False,
            "agent_state": {
                "Q_local": {"weight": torch.tensor([1.0, 2.0])},
                "Q_target": {"weight": torch.tensor([1.0, 2.0])},
                "optimizer": {},
                "replay": {},
                "rng_state": {},
            },
        }
        scores = {
            25: (1, 1, 20, -10, -15, -25),
            50: (1, 1, 19, -9, -10, -50),
        }
        base_latest_path = root / trainer.BASE_STATE_DIRECTORY / "latest.pth"
        audit._atomic_torch_save(source, base_latest_path)
        base_latest_sha = audit._sha256(base_latest_path)
        entries = []
        snapshot_payloads = {}
        for episode in (25, 50):
            record = _record(episode, scores[episode])
            payload = trainer._snapshot_payload(
                source,
                episode=episode,
                source_latest_sha256=(
                    "a" * 64 if episode == 25 else base_latest_sha
                ),
                prospective_contract=prospective_contract,
                validation_record=record,
            )
            relative = Path("snapshots") / f"episode-{episode:04d}.pth"
            path = root / relative
            audit._atomic_torch_save(payload, path)
            entries.append(
                {
                    "episode": episode,
                    "relative_path": relative.as_posix(),
                    "sha256": audit._sha256(path),
                    "source_latest_relative_path": "v1_1_training/latest.pth",
                    "source_latest_sha256": payload["source_latest_checkpoint_sha256"],
                    "base_resume_contract_sha256": prospective_contract[
                        "base_resume_contract_sha256"
                    ],
                    "validation_record_sha256": audit._contract_hash(record),
                    "validation_selection_score": list(record["selection_score"]),
                    "validation_selection_score_fields": list(
                        record["selection_score_fields"]
                    ),
                }
            )
            snapshot_payloads[episode] = payload

        best = copy.deepcopy(snapshot_payloads[25])
        best["agent_state"] = copy.deepcopy(snapshot_payloads[25]["agent_state"])
        best.update(
            {
                "checkpoint_role": "best_deployment_finalized",
                "trainer_resumable": False,
                "completed_training_episodes": 500,
                "protocol_training_complete": True,
                "selection_finalized_after_total_episodes": True,
                "deployment_checkpoint_eligible": True,
                "selected_checkpoint_episode": 25,
                "best_validation_record": _record(25, scores[25]),
            }
        )
        best_path = root / trainer.BASE_STATE_DIRECTORY / "best.pth"
        audit._atomic_torch_save(best, best_path)
        manifest = trainer._initial_manifest(prospective_contract)
        manifest.update(
            {
                "snapshot_episodes": [25, 50],
                "entries": entries,
                "completed_training_episodes": 500,
                "candidate_pool_complete": True,
                "v1_1_reference_checkpoint_episode": 25,
                "v1_1_reference_checkpoint_sha256": entries[0]["sha256"],
                "v1_1_best_checkpoint_sha256": audit._sha256(best_path),
            }
        )
        audit._atomic_json_save(
            manifest, root / trainer.SNAPSHOT_MANIFEST_FILENAME
        )

    def test_authenticator_consumes_exact_trainer_schema_and_derives_same_reference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "seed3"
            self._write_two_snapshot_bundle(root)
            with patch.object(
                selector, "SNAPSHOT_EPISODES", (25, 50)
            ), patch.object(selector.base_training, "_validate_resume"):
                bundle = selector._authenticate_snapshot_bundle(root)
            self.assertEqual(bundle.model_seed, 3)
            self.assertEqual(bundle.reference_episode, 25)
            self.assertEqual(len(bundle.snapshots), 2)

    def test_authenticator_rejects_manifest_checkpoint_hash_tampering(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "seed3"
            self._write_two_snapshot_bundle(root)
            manifest_path = root / trainer.SNAPSHOT_MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text())
            manifest["entries"][0]["sha256"] = "f" * 64
            audit._atomic_json_save(manifest, manifest_path)
            with patch.object(selector, "SNAPSHOT_EPISODES", (25, 50)):
                with self.assertRaisesRegex(ValueError, "file SHA-256"):
                    selector._authenticate_snapshot_bundle(root)

    def test_full_contract_rejects_consistent_frozen_recipe_drift(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "seed3"
            runtime = trainer._build_parser().parse_args(
                (
                    "--output-dir",
                    str(root),
                    "--model-seed",
                    "3",
                    "--device",
                    "cpu",
                )
            )
            contract, _, _ = trainer._prospective_contract(runtime)
            tampered = copy.deepcopy(contract)
            tampered["base_resume_contract"]["graph_config"]["gamma"] = 1.0
            tampered["base_resume_contract_sha256"] = audit._contract_hash(
                tampered["base_resume_contract"]
            )
            with self.assertRaisesRegex(
                ValueError, "full prospective/base training contract"
            ):
                selector._require_exact_prospective_contract(root, 3, tampered)


class ProspectiveSelectorOrderingTests(unittest.TestCase):
    def test_panel_a_rank_failure_occurs_before_panel_b_materialization(self):
        arm_by_seed = {}
        bundles = []
        for seed in selector.MODEL_SEEDS:
            arm = selector.SnapshotArm(
                model_seed=seed,
                episode=25,
                method_id=f"m-{seed}",
                path=Path(f"/tmp/m-{seed}.pth"),
                sha256="a" * 64,
                policy_digest="b" * 64,
                payload={"environment": {}},
                validation_record={},
            )
            arm_by_seed[seed] = arm
            bundles.append(
                selector.SnapshotBundle(
                    training_dir=Path(f"/tmp/t-{seed}"),
                    model_seed=seed,
                    manifest_path=Path(f"/tmp/t-{seed}/manifest.json"),
                    manifest_sha256="c" * 64,
                    manifest={},
                    snapshots=(arm,),
                    reference_episode=25,
                    reference_sha256="a" * 64,
                    shared_execution_contract={},
                )
            )
        rows = [
            {
                "method_id": arm_by_seed[seed].method_id,
                "instance_seed": instance_seed,
                "panel": "A",
            }
            for seed in selector.MODEL_SEEDS
            for instance_seed in selector.PANEL_A_SEEDS
        ]
        opened = []
        with TemporaryDirectory() as directory, patch.object(
            selector, "_load_bundles", return_value=tuple(bundles)
        ), patch.object(
            selector, "resolve_device", return_value=torch.device("cpu")
        ), patch.object(
            selector,
            "_load_or_create_panel_instances",
            side_effect=lambda _out, panel, _payload: opened.append(panel)
            or ({}, {"instances": {}}),
        ), patch.object(
            selector, "_run_panel", return_value=rows
        ), patch.object(
            selector, "_write_rows"
        ), patch.object(
            selector,
            "_compute_panel_a_ranks",
            side_effect=ValueError("unsafe reference"),
        ), patch.object(selector, "SNAPSHOT_EPISODES", (25,)):
            with self.assertRaisesRegex(ValueError, "unsafe reference"):
                selector.main(
                    (
                        "--training-dirs",
                        "a",
                        "b",
                        "c",
                        "--output-dir",
                        directory,
                        "--device",
                        "cpu",
                    )
                )
        self.assertEqual(opened, ["A"])


if __name__ == "__main__":
    unittest.main()
