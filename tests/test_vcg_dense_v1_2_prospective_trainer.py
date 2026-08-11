import copy
from contextlib import redirect_stderr
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

import run_vcg_objective_gamma_audit as audit
import train_vcg_dense_proper as base
from train_vcg_dense_v1_2_prospective import (
    BASE_STATE_DIRECTORY,
    CONTRACT_FILENAME,
    FRESH_MODEL_SEEDS,
    PROSPECTIVE_METHOD_VERSION,
    PROSPECTIVE_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
    PROSPECTIVE_TRAINING_PROTOCOL,
    SNAPSHOT_EPISODES,
    SNAPSHOT_MANIFEST_FILENAME,
    SNAPSHOT_ROLE,
    _build_parser,
    _commit_snapshot,
    _initial_manifest,
    _prepare_root,
    _prospective_contract,
    _snapshot_payload,
    _snapshot_relative_path,
    _same_policy_weights,
    _validate_manifest,
    _validate_snapshot_payload,
)
from viability_graph_episodic_audit import (
    EpisodicViabilityGraphHierarchyAgent,
)


def _runtime(root: Path, seed: int = 3, *, resume: bool = False):
    argv = [
        "--output-dir",
        str(root),
        "--model-seed",
        str(seed),
        "--device",
        "cpu",
    ]
    if resume:
        argv.append("--resume-existing")
    return _build_parser().parse_args(argv)


def _record(episode: int) -> dict:
    return {
        "checkpoint_episode": int(episode),
        "selection_score": (
            1.0,
            1.0,
            20.0,
            -10.0,
            -15.0,
            -float(episode),
        ),
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
        },
        "runs": (),
    }


def _source_checkpoint(root: Path, episode: int = 25):
    runtime = _runtime(root)
    prospective, base_contract, _ = _prospective_contract(runtime)
    agent = EpisodicViabilityGraphHierarchyAgent(
        config=audit._graph_config(
            base._frozen_audit_args(
                type(
                    "Runtime",
                    (),
                    {
                        "output_dir": root / BASE_STATE_DIRECTORY,
                        "model_seed": 3,
                        "device": "cpu",
                        "stop_after_episode": episode,
                        "log_every": 5,
                    },
                )()
            ),
            base.FROZEN_GAMMA,
        ),
        seed=3,
        device="cpu",
        epsilon=base.FROZEN_EPSILON_START,
    )
    payload = base._checkpoint(
        agent,
        include_replay=True,
        resumable=True,
        checkpoint_role="latest_resumable",
        completed_episodes=episode,
        contract=base_contract,
        train_history=tuple({"episode": n + 1} for n in range(episode)),
        validation_history=(_record(episode),),
        best_record=None,
        best_candidate_filename=None,
        best_candidate_sha256=None,
        best_checkpoint_sha256=None,
    )
    return prospective, base_contract, payload


class ProspectiveFrozenInterfaceTests(unittest.TestCase):
    def test_fresh_seed_and_snapshot_schedule_are_closed(self):
        self.assertEqual(FRESH_MODEL_SEEDS, (3, 4, 5))
        self.assertEqual(SNAPSHOT_EPISODES, tuple(range(25, 501, 25)))
        self.assertEqual(len(SNAPSHOT_EPISODES), 20)
        self.assertEqual(
            _snapshot_relative_path(25).as_posix(),
            "snapshots/episode-0025.pth",
        )
        self.assertEqual(
            _snapshot_relative_path(500).as_posix(),
            "snapshots/episode-0500.pth",
        )
        with self.assertRaisesRegex(ValueError, "not a snapshot boundary"):
            _snapshot_relative_path(26)

    def test_public_cli_exposes_no_recipe_or_panel_override(self):
        for invalid_seed in (0, 1, 2, 6):
            with self.subTest(seed=invalid_seed), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    _build_parser().parse_args(
                        ("--output-dir", "unused", "--model-seed", str(invalid_seed))
                    )
        for forbidden in ("--gamma", "--validation-seeds", "--lambda-abs"):
            with self.subTest(forbidden=forbidden), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    _build_parser().parse_args(
                        (
                            "--output-dir",
                            "unused",
                            "--model-seed",
                            "3",
                            forbidden,
                            "1",
                        )
                    )

    def test_contract_is_exact_v1_1_recipe_in_fresh_seed_namespace(self):
        with TemporaryDirectory() as directory:
            prospective, base_contract, train_seed_base = _prospective_contract(
                _runtime(Path(directory), seed=3)
            )
        self.assertEqual(train_seed_base, 53_000_000)
        self.assertEqual(
            prospective["prospective_training_protocol"],
            PROSPECTIVE_TRAINING_PROTOCOL,
        )
        self.assertEqual(
            prospective["prospective_method_version"],
            PROSPECTIVE_METHOD_VERSION,
        )
        self.assertEqual(base_contract["training_protocol"], base.TRAINING_PROTOCOL)
        self.assertEqual(base_contract["method_version"], base.METHOD_VERSION)
        self.assertEqual(base_contract["objective_spec"], base.FROZEN_OBJECTIVE_SPEC.to_dict())
        self.assertEqual(base_contract["gamma"], 0.99)
        self.assertEqual(base_contract["total_episodes"], 500)
        self.assertEqual(
            tuple(base_contract["validation_instance_seeds"]),
            base.FROZEN_VALIDATION_SEEDS,
        )
        self.assertFalse(prospective["future_a_b_selection_panels_opened"])
        self.assertFalse(prospective["sealed_test_panels_opened"])


class ProspectiveSnapshotTests(unittest.TestCase):
    def test_snapshot_is_model_only_directly_deployable_and_unselected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prospective, _, source = _source_checkpoint(root)
            snapshot = _snapshot_payload(
                source,
                episode=25,
                source_latest_sha256="a" * 64,
                prospective_contract=prospective,
                validation_record=_record(25),
            )
            _validate_snapshot_payload(
                snapshot,
                episode=25,
                prospective_contract=prospective,
                source_latest_sha256="a" * 64,
            )
            self.assertEqual(snapshot["checkpoint_role"], SNAPSHOT_ROLE)
            self.assertFalse(snapshot["trainer_resumable"])
            self.assertFalse(snapshot["candidate_selection_performed"])
            self.assertFalse(snapshot["deployment_checkpoint_eligible"])
            self.assertEqual(
                tuple(sorted(snapshot["agent_state"])),
                ("Q_local", "Q_target"),
            )
            loaded = EpisodicViabilityGraphHierarchyAgent.from_checkpoint(
                snapshot, device="cpu", resumable=False, seed=3
            )
            for name, tensor in source["agent_state"]["Q_local"].items():
                self.assertTrue(
                    torch.equal(tensor, loaded.Q_local.state_dict()[name])
                )
            self.assertTrue(_same_policy_weights(source, snapshot))
            drift = copy.deepcopy(snapshot)
            name = next(iter(drift["agent_state"]["Q_local"]))
            drift["agent_state"]["Q_local"][name].add_(1.0)
            self.assertFalse(_same_policy_weights(source, drift))

    def test_snapshot_validation_rejects_training_state_and_contract_drift(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prospective, _, source = _source_checkpoint(root)
            snapshot = _snapshot_payload(
                source,
                episode=25,
                source_latest_sha256="b" * 64,
                prospective_contract=prospective,
                validation_record=_record(25),
            )
            replay_drift = copy.deepcopy(snapshot)
            replay_drift["agent_state"]["replay"] = {}
            with self.assertRaisesRegex(ValueError, "model_only_agent_state"):
                _validate_snapshot_payload(
                    replay_drift,
                    episode=25,
                    prospective_contract=prospective,
                )
            seed_drift = copy.deepcopy(snapshot)
            seed_drift["model_seed"] = 4
            with self.assertRaisesRegex(ValueError, "model_seed"):
                _validate_snapshot_payload(
                    seed_drift,
                    episode=25,
                    prospective_contract=prospective,
                )


class ProspectiveManifestTests(unittest.TestCase):
    def test_root_resume_contract_fails_closed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            runtime = _runtime(root)
            prospective, _, _ = _prospective_contract(runtime)
            manifest = _prepare_root(runtime, prospective)
            self.assertEqual(manifest["entries"], [])
            self.assertTrue((root / CONTRACT_FILENAME).is_file())
            self.assertTrue((root / SNAPSHOT_MANIFEST_FILENAME).is_file())

            resumed = _prepare_root(_runtime(root, resume=True), prospective)
            self.assertEqual(resumed, manifest)
            contract_path = root / CONTRACT_FILENAME
            document = audit._json_safe(
                __import__("json").loads(contract_path.read_text())
            )
            document["contract"]["model_seed"] = 4
            audit._atomic_json_save(document, contract_path)
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                _prepare_root(_runtime(root, resume=True), prospective)

    def test_contract_only_initialization_gap_is_the_only_recoverable_gap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            runtime = _runtime(root)
            prospective, _, _ = _prospective_contract(runtime)
            _prepare_root(runtime, prospective)
            manifest_path = root / SNAPSHOT_MANIFEST_FILENAME
            manifest_path.unlink()
            recovered = _prepare_root(
                _runtime(root, resume=True), prospective
            )
            self.assertEqual(recovered["entries"], [])
            self.assertTrue(manifest_path.is_file())

            manifest_path.unlink()
            (root / "unexpected.txt").write_text("evidence of work")
            with self.assertRaisesRegex(ValueError, "non-pristine"):
                _prepare_root(_runtime(root, resume=True), prospective)

    def test_commit_creates_authenticated_immutable_manifest_entry(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            runtime = _runtime(root)
            prospective, base_contract, source = _source_checkpoint(root)
            manifest = _prepare_root(runtime, prospective)
            latest = root / BASE_STATE_DIRECTORY / "latest.pth"
            audit._atomic_torch_save(source, latest)
            committed = _commit_snapshot(
                runtime,
                manifest,
                episode=25,
                prospective_contract=prospective,
                base_contract=base_contract,
            )
            self.assertEqual(committed["completed_training_episodes"], 25)
            self.assertEqual(len(committed["entries"]), 1)
            self.assertFalse(committed["candidate_pool_complete"])
            self.assertIsNone(committed["v1_1_reference_checkpoint_episode"])
            self.assertEqual(
                committed["entries"][0]["relative_path"],
                "snapshots/episode-0025.pth",
            )
            _validate_manifest(root, committed, prospective)

            # File hashes make post-commit weight mutation fail closed.
            path = root / "snapshots/episode-0025.pth"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            first_name = next(iter(payload["agent_state"]["Q_local"]))
            payload["agent_state"]["Q_local"][first_name].add_(1.0)
            audit._atomic_torch_save(payload, path)
            with self.assertRaisesRegex(ValueError, "sha256"):
                _validate_manifest(root, committed, prospective)

    def test_initial_manifest_cannot_claim_reference_or_selection(self):
        with TemporaryDirectory() as directory:
            prospective, _, _ = _prospective_contract(
                _runtime(Path(directory), seed=3)
            )
            manifest = audit._json_safe(_initial_manifest(prospective))
        self.assertEqual(
            manifest["manifest_schema_version"],
            PROSPECTIVE_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        )
        self.assertFalse(manifest["candidate_selection_performed"])
        self.assertFalse(manifest["deployment_checkpoint_eligible"])
        self.assertIsNone(manifest["v1_1_reference_checkpoint_episode"])


if __name__ == "__main__":
    unittest.main()
