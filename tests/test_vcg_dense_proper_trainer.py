import copy
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as objective_audit
from benchmark_viability_critic_priority import (
    _freeze_agent,
    _load_controller_checkpoint,
)
from compare_viability_graph_baselines import _checkpoint_readiness
from train_viability_graph_smdp import SEALED_STRESS_V1_HOLDOUT_SEEDS
from train_viability_graph_smdp_proper import SEALED_IN_REGIME_TEST_SEEDS
from train_vcg_dense_proper import (
    ALLOWED_MODEL_SEEDS,
    CANDIDATE_FILENAMES,
    FROZEN_GAMMA,
    FROZEN_OBJECTIVE_SPEC,
    FROZEN_TOTAL_EPISODES,
    FROZEN_TRAIN_SEED_ORIGIN,
    FROZEN_TRAIN_SEED_STRIDE,
    FROZEN_VALIDATION_SEEDS,
    METHOD_VERSION,
    TRAINING_PROTOCOL,
    _assert_frozen_recipe,
    _build_parser,
    _checkpoint,
    _candidate_path,
    _derived_train_seed_base,
    _frozen_audit_args,
    _finalize_best_candidate,
    _next_candidate_filename,
    _result_payload,
    _resume_contract,
    _run,
    _validate_resume,
)
from vcg_objective_audit import (
    DENSE_PIECEWISE,
    TimingObjectiveSpec,
    delivery_reward,
)
from viability_graph_episodic_audit import (
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
    EpisodicViabilityGraphHierarchyAgent,
)


EXPECTED_DENSE_SPEC = TimingObjectiveSpec.dense(
    dense_b=40.0,
    lambda_abs=1.5,
    lambda_outside=0.5,
    window=20.0,
).to_dict()


def _runtime(output_dir: Path, model_seed: int):
    return _build_parser().parse_args(
        (
            "--output-dir",
            str(output_dir),
            "--model-seed",
            str(model_seed),
            "--device",
            "cpu",
        )
    )


def _frozen_context(output_dir: Path, model_seed: int):
    runtime = _runtime(output_dir, model_seed)
    args = _frozen_audit_args(runtime)
    train_seed_base = _assert_frozen_recipe(args)
    contract = _resume_contract(args, train_seed_base)
    agent = EpisodicViabilityGraphHierarchyAgent(
        config=objective_audit._graph_config(args, FROZEN_GAMMA),
        seed=model_seed,
        device="cpu",
        epsilon=0.9,
    )
    return args, train_seed_base, contract, agent


def _eligible_record(checkpoint_episode: int = 25) -> dict:
    return {
        "checkpoint_episode": int(checkpoint_episode),
        "selection_score": (1.0, 1.0, 10.0, -2.0, -1.0, -checkpoint_episode),
        "selection_score_fields": (
            "strict_success_rate",
            "completion_rate",
            "selected_objective_return_per_block",
            "negative_mean_absolute_error",
            "negative_relocations_per_100_deliveries",
            "negative_checkpoint_episode",
        ),
        "deployment_eligible": True,
        "development_selection_only": False,
        "summary": {
            "strict_method_success_rate": 1.0,
            "completion_rate": 1.0,
            "mean_dense_rescored_return": 80.0,
            "mean_absolute_error": 2.0,
        },
        "runs": (),
    }


class DenseProperFrozenContractTests(unittest.TestCase):
    def test_recipe_is_the_predeclared_500_episode_dense_g099_protocol(self):
        self.assertEqual(FROZEN_TOTAL_EPISODES, 500)
        self.assertEqual(FROZEN_VALIDATION_SEEDS, tuple(range(79_000, 79_020)))
        self.assertEqual(FROZEN_TRAIN_SEED_ORIGIN, 50_000_000)
        self.assertEqual(FROZEN_TRAIN_SEED_STRIDE, 1_000_000)
        self.assertEqual(ALLOWED_MODEL_SEEDS, (0, 1, 2))
        self.assertEqual(FROZEN_GAMMA, 0.99)
        self.assertEqual(FROZEN_OBJECTIVE_SPEC.to_dict(), EXPECTED_DENSE_SPEC)

        with TemporaryDirectory() as directory:
            args, base, contract, agent = _frozen_context(
                Path(directory), model_seed=1
            )
        self.assertEqual(base, 51_000_000)
        self.assertEqual(args.total_episodes, 500)
        self.assertEqual(tuple(args.validation_seeds), FROZEN_VALIDATION_SEEDS)
        self.assertEqual(contract["training_protocol"], TRAINING_PROTOCOL)
        self.assertEqual(contract["method_version"], METHOD_VERSION)
        self.assertEqual(contract["objective_spec"], EXPECTED_DENSE_SPEC)
        self.assertEqual(contract["gamma"], 0.99)
        self.assertEqual(contract["graph_config"]["gamma"], 0.99)
        self.assertEqual(agent.config.gamma, 0.99)

    def test_public_cli_exposes_no_objective_or_discount_override(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _build_parser().parse_args(
                    (
                        "--output-dir",
                        "unused",
                        "--model-seed",
                        "1",
                        "--gamma",
                        "1.0",
                    )
                )
            with self.assertRaises(SystemExit):
                _build_parser().parse_args(
                    (
                        "--output-dir",
                        "unused",
                        "--model-seed",
                        "1",
                        "--lambda-abs",
                        "1.4",
                    )
                )

    def test_training_environment_uses_the_authenticated_dense_objective(self):
        with TemporaryDirectory() as directory:
            args, _, contract, _ = _frozen_context(Path(directory), 1)
            env = objective_audit._new_env(args, FROZEN_OBJECTIVE_SPEC)
        self.assertEqual(env.timing_objective_spec.to_dict(), EXPECTED_DENSE_SPEC)
        self.assertEqual(contract["objective_spec"], env.timing_objective_spec.to_dict())
        # At |e|=30 the selected dense delivery component is -10, whereas
        # the frozen legacy component is +10.  This guards against a trainer
        # that authenticates dense metadata but accidentally learns legacy rewards.
        self.assertEqual(
            delivery_reward(-30.0, FROZEN_OBJECTIVE_SPEC), -10.0
        )


class DenseProperSeedIsolationTests(unittest.TestCase):
    def test_model_seed_namespaces_are_disjoint_and_validation_is_shared(self):
        ranges = []
        for model_seed, expected_base in enumerate(
            (50_000_000, 51_000_000, 52_000_000)
        ):
            self.assertEqual(_derived_train_seed_base(model_seed), expected_base)
            ranges.append(
                set(range(expected_base, expected_base + FROZEN_TOTAL_EPISODES))
            )
        for left in range(len(ranges)):
            for right in range(left + 1, len(ranges)):
                self.assertFalse(ranges[left] & ranges[right])
        for seeds in ranges:
            self.assertFalse(seeds & set(FROZEN_VALIDATION_SEEDS))
        self.assertFalse(
            set(FROZEN_VALIDATION_SEEDS)
            & (SEALED_STRESS_V1_HOLDOUT_SEEDS | SEALED_IN_REGIME_TEST_SEEDS)
        )
        # The dedicated panel is fresh relative to both development screens.
        used_development = set(range(76_000, 76_010)) | set(
            range(78_000, 78_020)
        )
        self.assertFalse(set(FROZEN_VALIDATION_SEEDS) & used_development)

    def test_recipe_validator_refuses_both_sealed_panels_in_either_split(self):
        with TemporaryDirectory() as directory:
            for protected_seed in (69_000, 77_000):
                with self.subTest(protected_seed=protected_seed, split="validation"):
                    args = _frozen_audit_args(_runtime(Path(directory), 1))
                    args.validation_seeds = (protected_seed,)
                    with self.assertRaisesRegex(ValueError, "sealed test seeds"):
                        _assert_frozen_recipe(args)
                with self.subTest(protected_seed=protected_seed, split="training"):
                    args = _frozen_audit_args(_runtime(Path(directory), 1))
                    args.train_instance_seed_base = protected_seed
                    with self.assertRaisesRegex(ValueError, "sealed test seeds"):
                        _assert_frozen_recipe(args)

    def test_seed_one_checkpoint_cannot_resume_seed_two(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, seed_one_contract, seed_one_agent = _frozen_context(
                root / "seed1", 1
            )
            _, _, seed_two_contract, seed_two_agent = _frozen_context(
                root / "seed2", 2
            )
            payload = _checkpoint(
                seed_one_agent,
                include_replay=True,
                resumable=True,
                checkpoint_role="latest_resumable",
                completed_episodes=1,
                contract=seed_one_contract,
                train_history=(),
                validation_history=(),
                best_record=None,
                best_candidate_filename=None,
                best_candidate_sha256=None,
                best_checkpoint_sha256=None,
            )
            _validate_resume(payload, seed_one_contract)
            with self.assertRaisesRegex(ValueError, "incompatible VCG-Dense"):
                _validate_resume(payload, seed_two_contract)

            seed_one_state = seed_one_agent.Q_local.state_dict()
            seed_two_state = seed_two_agent.Q_local.state_dict()
            self.assertTrue(
                any(
                    not torch.equal(seed_one_state[name], seed_two_state[name])
                    for name in seed_one_state
                )
            )


class DenseProperCheckpointTests(unittest.TestCase):
    def test_candidate_slots_alternate_and_paths_are_exactly_whitelisted(self):
        slot_a, slot_b = CANDIDATE_FILENAMES
        self.assertEqual(_next_candidate_filename(None), slot_a)
        self.assertEqual(_next_candidate_filename(slot_a), slot_b)
        self.assertEqual(_next_candidate_filename(slot_b), slot_a)

        with TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(_candidate_path(output, slot_a), output / slot_a)
            self.assertEqual(_candidate_path(output, slot_b), output / slot_b)
            for invalid in (
                None,
                "",
                "best.pth",
                f"../{slot_a}",
                f"nested/{slot_a}",
                str((output / slot_a).resolve()),
            ):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(
                        ValueError, "invalid candidate checkpoint filename"
                    ):
                        _candidate_path(output, invalid)
                    if invalid is not None:
                        with self.assertRaisesRegex(
                            ValueError, "invalid candidate checkpoint filename"
                        ):
                            _next_candidate_filename(invalid)

    def test_resume_requires_a_whitelisted_candidate_filename_and_sha256(self):
        with TemporaryDirectory() as directory:
            _, _, contract, agent = _frozen_context(Path(directory), 1)
            record = _eligible_record()
            payload = _checkpoint(
                agent,
                include_replay=True,
                resumable=True,
                checkpoint_role="latest_resumable",
                completed_episodes=25,
                contract=contract,
                train_history=(),
                validation_history=(record,),
                best_record=record,
                best_candidate_filename=CANDIDATE_FILENAMES[0],
                best_candidate_sha256="a" * 64,
                best_checkpoint_sha256=None,
            )
            _validate_resume(payload, contract)
            alternate_slot = copy.deepcopy(payload)
            alternate_slot["best_candidate_checkpoint_filename"] = (
                CANDIDATE_FILENAMES[1]
            )
            _validate_resume(alternate_slot, contract)

            invalid_pairs = (
                (f"../{CANDIDATE_FILENAMES[0]}", "a" * 64),
                ("best.pth", "a" * 64),
                (CANDIDATE_FILENAMES[0], None),
                (CANDIDATE_FILENAMES[0], ""),
                (CANDIDATE_FILENAMES[0], "not-a-sha256"),
                (CANDIDATE_FILENAMES[0], "a" * 63),
                (CANDIDATE_FILENAMES[0], "z" * 64),
            )
            for filename, sha256 in invalid_pairs:
                mutated = copy.deepcopy(payload)
                mutated["best_candidate_checkpoint_filename"] = filename
                mutated["best_candidate_checkpoint_sha256"] = sha256
                with self.subTest(filename=filename, sha256=sha256):
                    with self.assertRaisesRegex(
                        ValueError, "incompatible VCG-Dense"
                    ):
                        _validate_resume(mutated, contract)

            no_record = copy.deepcopy(payload)
            no_record["best_validation_record"] = None
            with self.assertRaisesRegex(ValueError, "incompatible VCG-Dense"):
                _validate_resume(no_record, contract)

    def test_manifest_only_resume_recovers_an_authenticated_episode_zero_latest(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            args, train_seed_base, contract, _ = _frozen_context(output, 1)
            args.resume_existing = True
            args.stop_after_episode = 1
            output.mkdir(parents=True, exist_ok=True)
            objective_audit._atomic_json_save(
                {
                    "training_protocol": TRAINING_PROTOCOL,
                    "method_version": METHOD_VERSION,
                    "resume_contract_sha256": objective_audit._contract_hash(
                        contract
                    ),
                    "contract": contract,
                },
                output / "training-contract.json",
            )
            with patch.object(
                objective_audit,
                "run_objective_audit_episode",
                side_effect=RuntimeError("stop-after-episode-zero-recovery"),
            ):
                with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(
                        RuntimeError, "stop-after-episode-zero-recovery"
                    ):
                        _run(
                            args,
                            device=torch.device("cpu"),
                            train_seed_base=train_seed_base,
                        )

            latest = torch.load(
                output / "latest.pth", map_location="cpu", weights_only=False
            )
            _validate_resume(latest, contract)
            self.assertEqual(latest["completed_training_episodes"], 0)
            self.assertEqual(latest["train_instance_seeds"], ())
            self.assertIsNone(latest["best_validation_record"])
            self.assertIsNone(latest["best_candidate_checkpoint_filename"])
            self.assertIsNone(latest["best_candidate_checkpoint_sha256"])
            self.assertFalse(latest["protocol_training_complete"])

    def test_missing_latest_with_orphan_artifacts_cannot_roll_back_to_episode_zero(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            args, train_seed_base, contract, _ = _frozen_context(output, 1)
            args.resume_existing = True
            args.stop_after_episode = 1
            output.mkdir(parents=True, exist_ok=True)
            objective_audit._atomic_json_save(
                {
                    "training_protocol": TRAINING_PROTOCOL,
                    "method_version": METHOD_VERSION,
                    "resume_contract_sha256": objective_audit._contract_hash(
                        contract
                    ),
                    "contract": contract,
                },
                output / "training-contract.json",
            )
            # Any second material artifact means this is not the sole safe
            # manifest-committed/episode-0-uncommitted crash state.
            (output / CANDIDATE_FILENAMES[0]).write_bytes(b"orphan")
            with patch.object(
                objective_audit,
                "run_objective_audit_episode",
                side_effect=AssertionError("training must not start"),
            ):
                with self.assertRaisesRegex(
                    ValueError, "episode-0|orphan|additional|manifest-only"
                ):
                    _run(
                        args,
                        device=torch.device("cpu"),
                        train_seed_base=train_seed_base,
                    )

    def test_resume_rejects_objective_gamma_and_contract_tampering(self):
        with TemporaryDirectory() as directory:
            _, _, contract, agent = _frozen_context(Path(directory), 1)
            payload = _checkpoint(
                agent,
                include_replay=True,
                resumable=True,
                checkpoint_role="latest_resumable",
                completed_episodes=1,
                contract=contract,
                train_history=(),
                validation_history=(),
                best_record=None,
                best_candidate_filename=None,
                best_candidate_sha256=None,
                best_checkpoint_sha256=None,
            )
            _validate_resume(payload, contract)

            mutations = []
            top_spec = copy.deepcopy(payload)
            top_spec["timing_objective_spec"]["lambda_abs"] = 1.4
            mutations.append(top_spec)
            top_objective = copy.deepcopy(payload)
            top_objective["timing_objective"] = "legacy_clipped_v1"
            mutations.append(top_objective)
            top_gamma = copy.deepcopy(payload)
            top_gamma["gamma"] = 1.0
            mutations.append(top_gamma)
            config_gamma = copy.deepcopy(payload)
            config_gamma["config"]["gamma"] = 1.0
            mutations.append(config_gamma)

            for index, mutated in enumerate(mutations):
                with self.subTest(mutation=index):
                    with self.assertRaisesRegex(
                        ValueError, "incompatible VCG-Dense"
                    ):
                        _validate_resume(mutated, contract)

            changed_contract = copy.deepcopy(contract)
            changed_contract["objective_spec"]["lambda_outside"] = 0.4
            with self.assertRaisesRegex(ValueError, "incompatible VCG-Dense"):
                _validate_resume(payload, changed_contract)

    def test_partial_run_cannot_publish_an_eligible_best_checkpoint(self):
        with TemporaryDirectory() as directory:
            args, _, contract, agent = _frozen_context(Path(directory), 1)
            record = _eligible_record()
            with self.assertRaisesRegex(ValueError, "full.*budget|final"):
                _checkpoint(
                    agent,
                    include_replay=False,
                    resumable=False,
                    checkpoint_role="best_candidate_unfinalized",
                    completed_episodes=FROZEN_TOTAL_EPISODES - 1,
                    contract=contract,
                    train_history=(),
                    validation_history=(record,),
                    best_record=record,
                    best_candidate_filename=CANDIDATE_FILENAMES[0],
                    best_candidate_sha256=None,
                    best_checkpoint_sha256=None,
                    protocol_training_complete=False,
                    selection_finalized_after_total_episodes=False,
                    deployment_checkpoint_eligible=True,
                )

            result = _result_payload(
                args,
                completed_episodes=FROZEN_TOTAL_EPISODES - 1,
                contract=contract,
                train_history=(),
                validation_history=(record,),
                best_record=record,
                best_candidate_filename=CANDIDATE_FILENAMES[0],
                best_candidate_sha256="candidate-sha",
                best_checkpoint_sha256=None,
                agent=agent,
            )
        self.assertEqual(result["status"], "paused")
        self.assertFalse(result["protocol_training_complete"])
        self.assertFalse(result["selection_finalized_after_total_episodes"])
        self.assertFalse(result["deployment_checkpoint_eligible"])
        self.assertIsNone(result["best_checkpoint"])
        self.assertIsNotNone(result["best_candidate_checkpoint"])

    def test_full_budget_best_is_finalized_and_evaluator_compatible(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args, _, contract, agent = _frozen_context(root, 1)
            record = _eligible_record(checkpoint_episode=100)
            # Frozen evaluation records legitimately contain NaN loss means.
            # Selection identity must not rely on recursive NaN equality.
            record["runs"] = ({"loss_mean": float("nan")},)
            candidate_payload = _checkpoint(
                agent,
                include_replay=False,
                resumable=False,
                checkpoint_role="best_candidate_unfinalized",
                completed_episodes=100,
                contract=contract,
                train_history=(),
                validation_history=(record,),
                best_record=record,
                best_candidate_filename=CANDIDATE_FILENAMES[0],
                best_candidate_sha256=None,
                best_checkpoint_sha256=None,
                protocol_training_complete=False,
                selection_finalized_after_total_episodes=False,
                deployment_checkpoint_eligible=False,
            )
            candidate_path = root / CANDIDATE_FILENAMES[0]
            best_path = root / "best.pth"
            torch.save(candidate_payload, candidate_path)
            candidate_sha = objective_audit._sha256(candidate_path)
            train_history = tuple({"episode": index + 1} for index in range(500))
            with self.assertRaisesRegex(ValueError, "all training episodes"):
                _finalize_best_candidate(
                    candidate_path,
                    best_path,
                    candidate_sha256=candidate_sha,
                    contract=contract,
                    train_history=train_history[:-1],
                    validation_history=(record,),
                    best_record=record,
                )
            best_sha = _finalize_best_candidate(
                candidate_path,
                best_path,
                candidate_sha256=candidate_sha,
                contract=contract,
                train_history=train_history,
                validation_history=(record,),
                best_record=record,
            )
            self.assertEqual(best_sha, objective_audit._sha256(best_path))
            payload = torch.load(best_path, map_location="cpu", weights_only=False)
            self.assertTrue(payload["protocol_training_complete"])
            self.assertTrue(payload["selection_finalized_after_total_episodes"])
            self.assertTrue(payload["deployment_checkpoint_eligible"])
            self.assertEqual(payload["timing_objective_spec"], EXPECTED_DENSE_SPEC)
            self.assertEqual(payload["gamma"], 0.99)
            self.assertEqual(
                payload["checkpoint_family"],
                EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
            )

            loaded = _load_controller_checkpoint(best_path)
            frozen = _freeze_agent(
                loaded, device=torch.device("cpu"), seed=1
            )
            readiness = _checkpoint_readiness(loaded)

            result = _result_payload(
                args,
                completed_episodes=FROZEN_TOTAL_EPISODES,
                contract=contract,
                train_history=(),
                validation_history=(record,),
                best_record=record,
                best_candidate_filename=CANDIDATE_FILENAMES[0],
                best_candidate_sha256=candidate_sha,
                best_checkpoint_sha256=best_sha,
                agent=agent,
            )

        self.assertIsInstance(frozen, EpisodicViabilityGraphHierarchyAgent)
        self.assertEqual(frozen.config.gamma, 0.99)
        self.assertEqual(frozen.epsilon, 0.0)
        self.assertFalse(frozen.Q_local.training)
        self.assertEqual(loaded["timing_objective"], DENSE_PIECEWISE)
        self.assertFalse(readiness["development_mechanism_screen_checkpoint"])
        self.assertTrue(readiness["deployment_checkpoint_eligible"])
        self.assertFalse(readiness["performance_claim_authorized"])
        self.assertTrue(result["protocol_training_complete"])
        self.assertTrue(result["selection_finalized_after_total_episodes"])
        self.assertTrue(result["deployment_checkpoint_eligible"])
        self.assertIsNotNone(result["best_checkpoint"])


if __name__ == "__main__":
    unittest.main()
