import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from PSLAP.viability_candidates_hold_v2 import (
    CertifiedHoldRuleV2,
    PrimitiveIdleBudgetStateV2,
    enumerate_viability_candidates_hold_v2,
)
from PSLAP.viability_filter import ViabilitySearchConfig
import train_vcg_constrained_v2_3 as trainer
import train_vcg_constrained_v2_2 as trainer22
import viability_graph_constrained_v2 as v2
import viability_graph_constrained_v2_1 as v21
import viability_graph_constrained_v2_2 as v22
from viability_graph_constrained_v2_3 import (
    ConstrainedV23Config,
    ConstrainedV23DevelopmentRuntime,
    ConstrainedV23HierarchyAgent,
    V2_3_DIAGNOSTIC_VERSION,
)
try:
    from tests.test_viability_graph_constrained_v2 import _record
except ImportError:  # Supports unittest discovery with ``-s tests``.
    from test_viability_graph_constrained_v2 import _record


def _small_config(**overrides):
    values = {
        "graph_hidden_dim": 8,
        "graph_embedding_dim": 8,
        "message_passing_steps": 1,
        "action_embedding_dim": 4,
        "head_hidden_dim": 8,
        "batch_size": 2,
        "replay_capacity": 8,
        "episode_horizon_steps": 12,
        "max_hold_steps": 10,
        "max_idle_steps": 20,
        "gamma_operational": 1.0,
        "gamma_rehandle": 1.0,
        "operational_reward_scale": 0.01,
    }
    values.update(overrides)
    return ConstrainedV23Config(**values)


def _snapshot():
    try:
        from tests.test_viability_candidates_hold_v2 import compact_env, make_stored
    except ImportError:  # Supports unittest discovery with ``-s tests``.
        from test_viability_candidates_hold_v2 import compact_env, make_stored

    env = compact_env(1, seed=9951)
    make_stored(env.blocks[0], (2, 2), duration=30)
    env.current_state = (1, 1)
    env.time_steps = 5
    return enumerate_viability_candidates_hold_v2(
        env,
        idle_budget=PrimitiveIdleBudgetStateV2(20, 3),
        remaining_episode_steps=7,
        recovery_witness_forced=False,
        search_config=ViabilitySearchConfig(
            max_nodes=None, search_order="breadth_first"
        ),
        hold_rule=CertifiedHoldRuleV2(10, 20),
    )


class V23PolicyOperatorTests(unittest.TestCase):
    def test_validation_training_false_still_samples_same_induced_policy(self):
        config = _small_config()
        training_agent = ConstrainedV23HierarchyAgent(
            config=config, seed=4, device="cpu", epsilon=0.0
        )
        validation_agent = ConstrainedV23HierarchyAgent(
            config=config, seed=4, device="cpu", epsilon=0.0
        )
        training_agent.set_schedule_state(
            trainer.schedule_for_episode(80, validation=False).to_dict()
        )
        validation_agent.set_schedule_state(
            trainer.schedule_for_episode(80, validation=True).to_dict()
        )
        training_agent.reset_behavior_rng(620_123_456)
        validation_agent.reset_behavior_rng(620_123_456)

        trained = training_agent.select(_snapshot(), training=True, epsilon=0.0)
        evaluated = validation_agent.select(
            _snapshot(), training=False, epsilon=0.0
        )
        self.assertEqual(trained.record.key, evaluated.record.key)
        self.assertEqual(
            trained.selection_source,
            "induced_nested_stochastic_lagrangian_policy",
        )
        self.assertEqual(trained.selection_source, evaluated.selection_source)
        self.assertNotIn("map", evaluated.selection_source.lower())
        self.assertEqual(
            training_agent.behavior_rng.getstate(),
            validation_agent.behavior_rng.getstate(),
        )

    def test_select_and_backup_use_identical_policy_temperatures(self):
        agent = ConstrainedV23HierarchyAgent(
            config=_small_config(), seed=5, device="cpu", epsilon=0.0
        )
        schedule = trainer.schedule_for_episode(40).to_dict()
        agent.set_schedule_state(schedule)
        agent.reset_behavior_rng(trainer.training_policy_rng_seed(40))
        calls = []
        original = v2.shared_lagrangian_policy_values

        def recording(*args, **kwargs):
            calls.append(
                (
                    tuple(float(value) for value in torch.as_tensor(args[5])),
                    float(torch.as_tensor(args[6])),
                )
            )
            return original(*args, **kwargs)

        transition = v2.ConstrainedV2Transition(
            chosen=_record(0),
            operational_reward=1.0,
            physical_rehandle_cost=0.0,
            duration=1,
            next_candidates=(_record(0, "next-a"), _record(1, "next-b")),
            done=False,
            behavior_lambda=0.0,
        )
        with mock.patch.object(
            v2, "shared_lagrangian_policy_values", side_effect=recording
        ):
            agent.select(_snapshot(), training=True, epsilon=0.0)
            agent._td_batch([transition])

        self.assertGreaterEqual(len(calls), 2)
        expected = (
            tuple(schedule["within_group_temperatures"]),
            schedule["group_temperature"],
        )
        for observed in calls:
            for actual, frozen in zip(observed[0], expected[0]):
                self.assertAlmostEqual(actual, frozen, places=7)
            self.assertAlmostEqual(observed[1], expected[1], places=7)

    def test_replay_rng_draws_do_not_perturb_behavior_rng(self):
        agent = ConstrainedV23HierarchyAgent(
            config=_small_config(), seed=6, device="cpu", epsilon=0.0
        )
        agent.set_schedule_state(trainer.schedule_for_episode(80).to_dict())
        agent.reset_behavior_rng(610_000_079)
        behavior_before = agent.behavior_rng.getstate()
        replay_before = agent.replay_rng.getstate()
        for _ in range(25):
            agent.replay_rng.random()
        self.assertEqual(agent.behavior_rng.getstate(), behavior_before)
        self.assertNotEqual(agent.replay_rng.getstate(), replay_before)


class V23CheckpointAndIsolationTests(unittest.TestCase):
    def _payload(self):
        agent = ConstrainedV23HierarchyAgent(
            config=_small_config(), seed=7, device="cpu", epsilon=0.0
        )
        agent.set_schedule_state(
            trainer.schedule_for_episode(80, validation=True).to_dict()
        )
        agent.reset_behavior_rng(620_000_000)
        return agent, agent.checkpoint(include_replay=False)

    def test_checkpoint_rejects_every_policy_semantic_change(self):
        agent, payload = self._payload()
        restored = ConstrainedV23HierarchyAgent.from_checkpoint(
            payload, device="cpu", resumable=False, seed=7
        )
        self.assertEqual(dict(restored.schedule_state), dict(agent.schedule_state))
        self.assertIsNone(restored.behavior_rng_seed)
        with self.assertRaisesRegex(RuntimeError, "behavior RNG"):
            restored.select(_snapshot(), training=False, epsilon=0.0)
        self.assertEqual(payload["policy_diagnostic_version"], V2_3_DIAGNOSTIC_VERSION)

        corruptions = {
            "training_policy": "map",
            "backup_policy": "map",
            "validation_policy": "map",
            "deployment_policy": "map",
            "policy_realization_decoupled_from_training_flag": False,
            "same_policy_for_behavior_backup_validation_deployment": False,
            "map_policy_available": True,
            "replay_rng_seed": trainer.REPLAY_RNG_SEED + 1,
            "policy_diagnostic_distribution": "wrong",
            "exact_safe_mask_authoritative": False,
        }
        for field, value in corruptions.items():
            with self.subTest(field=field):
                corrupted = dict(payload)
                corrupted[field] = value
                with self.assertRaisesRegex(ValueError, "incompatible constrained V2.3"):
                    ConstrainedV23HierarchyAgent.from_checkpoint(
                        corrupted, device="cpu", resumable=False, seed=7
                    )

        with self.assertRaisesRegex(ValueError, "incompatible constrained V2 checkpoint"):
            v2.ConstrainedV2HierarchyAgent.from_checkpoint(
                payload, device="cpu", resumable=False, seed=7
            )
        with self.assertRaisesRegex(ValueError, "incompatible constrained V2.1"):
            v21.ConstrainedV21HierarchyAgent.from_checkpoint(
                payload, device="cpu", resumable=False, seed=7
            )
        with self.assertRaisesRegex(ValueError, "incompatible constrained V2.2"):
            v22.ConstrainedV22HierarchyAgent.from_checkpoint(
                payload, device="cpu", resumable=False, seed=7
            )


class V23DiagnosticLoaderTests(unittest.TestCase):
    def _fixture(self, root: Path):
        args = trainer.build_parser().parse_args([
            "--output-dir",
            str(root / "unused-output"),
            "--device",
            "cpu",
        ])
        contract = trainer.build_training_contract(args)
        config = ConstrainedV23Config.from_dict(contract["agent_config"])
        agent = ConstrainedV23HierarchyAgent(
            config=config,
            seed=10,
            device="cpu",
            epsilon=0.0,
            dual_lambda=0.0,
        )
        schedule = trainer.schedule_for_episode(80, validation=True).to_dict()
        agent.set_schedule_state(schedule)

        class Runtime:
            dual_lambda = 0.0
            core_contract = {"checkpoint_family": trainer.CHECKPOINT_FAMILY}

            @staticmethod
            def checkpoint_state(*, include_replay):
                return agent.checkpoint(include_replay=include_replay)

        dual = trainer.CompleteBlockProjectedDual(trainer.V23DualConfig())
        validation = {
            "checkpoint_episode": 80,
            "candidate_look_gate": True,
            "development_candidate_eligible": False,
            "validation_lambda": 0.0,
            "schedule_state": schedule,
        }
        payload = trainer._checkpoint_payload(
            contract=contract,
            runtime=Runtime(),
            dual=dual,
            completed_episodes=80,
            role="candidate_look_diagnostic",
            schedule_state=schedule,
            pending_dual_runs=(),
            validation_summary=validation,
            include_replay=False,
        )
        checkpoint = root / "candidate-look-checkpoints/episode-0080-model-only.pth"
        checkpoint.parent.mkdir(parents=True)
        torch.save(payload, checkpoint)
        artifact = {
            "checkpoint_episode": 80,
            "relative_path": "candidate-look-checkpoints/episode-0080-model-only.pth",
            "sha256": trainer._file_sha256(checkpoint),
            "checkpoint_role": "candidate_look_diagnostic",
            "model_only": True,
            "diagnostic_only": True,
            "development_candidate_eligible": False,
            "deployment_checkpoint_eligible": False,
            "validation_development_candidate_eligible": False,
        }
        manifest = {
            "schema_version": 1,
            "checkpoint_family": trainer.CHECKPOINT_FAMILY,
            "training_contract_sha256": contract["contract_sha256"],
            "expected_candidate_look_episodes": trainer.CANDIDATE_EPISODES,
            "saved_candidate_look_episodes": (80,),
            "artifact_count": 1,
            "complete": False,
            "model_only": True,
            "diagnostic_only": True,
            "final_86xxx_panel_opened": False,
            "artifacts": (artifact,),
        }
        manifest["manifest_sha256"] = trainer.contract_hash(manifest)
        manifest_path = root / "candidate-look-checkpoint-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return contract, checkpoint, manifest_path, payload, manifest

    @staticmethod
    def _write_manifest(path: Path, manifest):
        canonical = dict(manifest)
        canonical.pop("manifest_sha256", None)
        canonical["manifest_sha256"] = trainer.contract_hash(canonical)
        path.write_text(json.dumps(canonical), encoding="utf-8")
        return canonical

    def test_authenticated_loader_accepts_only_bound_model_artifact(self):
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent)
            contract, checkpoint, manifest_path, _, _ = self._fixture(root)
            loaded = trainer.load_candidate_look_diagnostic(
                checkpoint,
                manifest_path=manifest_path,
                contract=contract,
                device="cpu",
            )
            self.assertIsInstance(loaded["agent"], ConstrainedV23HierarchyAgent)
            self.assertEqual(loaded["checkpoint"]["completed_episodes"], 80)
            self.assertIsNone(loaded["agent"].behavior_rng_seed)

    def test_loader_rejects_manifest_hash_traversal_and_file_sha(self):
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent)
            contract, checkpoint, manifest_path, _, manifest = self._fixture(root)

            corrupted = dict(manifest)
            corrupted["artifact_count"] = 2
            manifest_path.write_text(json.dumps(corrupted), encoding="utf-8")
            with self.assertRaisesRegex(trainer.ConstrainedV23TrainerError, "self-hash"):
                trainer.load_candidate_look_diagnostic(
                    checkpoint, manifest_path=manifest_path, contract=contract
                )

            traversing = json.loads(json.dumps(manifest))
            traversing["artifacts"][0]["relative_path"] = "../episode-0080-model-only.pth"
            self._write_manifest(manifest_path, traversing)
            with self.assertRaisesRegex(trainer.ConstrainedV23TrainerError, "relative path"):
                trainer.load_candidate_look_diagnostic(
                    checkpoint, manifest_path=manifest_path, contract=contract
                )

            bad_sha = json.loads(json.dumps(manifest))
            bad_sha["artifacts"][0]["sha256"] = "0" * 64
            self._write_manifest(manifest_path, bad_sha)
            with self.assertRaisesRegex(trainer.ConstrainedV23TrainerError, "file SHA"):
                trainer.load_candidate_look_diagnostic(
                    checkpoint, manifest_path=manifest_path, contract=contract
                )

            bad_schema = json.loads(json.dumps(manifest))
            bad_schema["schema_version"] = 999
            self._write_manifest(manifest_path, bad_schema)
            with self.assertRaisesRegex(trainer.ConstrainedV23TrainerError, "manifest contract"):
                trainer.load_candidate_look_diagnostic(
                    checkpoint, manifest_path=manifest_path, contract=contract
                )

            eligibility_drift = json.loads(json.dumps(manifest))
            eligibility_drift["artifacts"][0][
                "validation_development_candidate_eligible"
            ] = True
            self._write_manifest(manifest_path, eligibility_drift)
            with self.assertRaisesRegex(trainer.ConstrainedV23TrainerError, "eligibility mismatch"):
                trainer.load_candidate_look_diagnostic(
                    checkpoint, manifest_path=manifest_path, contract=contract
                )

    def test_loader_rejects_outer_flag_lambda_and_v22_family(self):
        mutations = (
            ("eligibility", lambda payload: payload.update(development_candidate_eligible=True), "envelope"),
            ("lambda", lambda payload: payload.update(runtime_lambda=0.5), "lambda"),
            ("v22", lambda payload: payload.update(checkpoint_family=trainer22.CHECKPOINT_FAMILY), "envelope"),
            ("schema", lambda payload: payload.update(checkpoint_schema_version=999), "envelope"),
            ("map", lambda payload: payload.update(map_validation_used=True), "envelope"),
        )
        for name, mutate, pattern in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as parent:
                root = Path(parent)
                contract, checkpoint, manifest_path, payload, manifest = self._fixture(root)
                mutate(payload)
                torch.save(payload, checkpoint)
                updated_manifest = json.loads(json.dumps(manifest))
                updated_manifest["artifacts"][0]["sha256"] = trainer._file_sha256(checkpoint)
                self._write_manifest(manifest_path, updated_manifest)
                with self.assertRaisesRegex(trainer.ConstrainedV23TrainerError, pattern):
                    trainer.load_candidate_look_diagnostic(
                        checkpoint,
                        manifest_path=manifest_path,
                        contract=contract,
                        device="cpu",
                    )

    def test_loader_rejects_injected_training_state_and_false_lifecycle(self):
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent)
            contract, checkpoint, manifest_path, payload, manifest = self._fixture(root)
            payload["agent_state"]["agent_state"]["optimizer"] = {"injected": True}
            torch.save(payload, checkpoint)
            updated_manifest = json.loads(json.dumps(manifest))
            updated_manifest["artifacts"][0]["sha256"] = trainer._file_sha256(checkpoint)
            self._write_manifest(manifest_path, updated_manifest)
            with self.assertRaisesRegex(trainer.ConstrainedV23TrainerError, "state keys"):
                trainer.load_candidate_look_diagnostic(
                    checkpoint,
                    manifest_path=manifest_path,
                    contract=contract,
                    device="cpu",
                )

        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent)
            contract, checkpoint, manifest_path, _, manifest = self._fixture(root)
            false_lifecycle = json.loads(json.dumps(manifest))
            false_lifecycle["saved_candidate_look_episodes"] = [200]
            false_lifecycle["complete"] = True
            self._write_manifest(manifest_path, false_lifecycle)
            with self.assertRaisesRegex(trainer.ConstrainedV23TrainerError, "lifecycle"):
                trainer.load_candidate_look_diagnostic(
                    checkpoint,
                    manifest_path=manifest_path,
                    contract=contract,
                    device="cpu",
                )

    def test_loader_rejects_self_rehashed_contract_without_parent_binding(self):
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent)
            contract, checkpoint, manifest_path, _, _ = self._fixture(root)
            corrupted_contract = json.loads(json.dumps(contract))
            corrupted_contract.pop("parent_v2_2_binding")
            corrupted_contract.pop("contract_sha256")
            corrupted_contract["contract_sha256"] = trainer.contract_hash(
                corrupted_contract
            )
            with self.assertRaisesRegex(trainer.ConstrainedV23TrainerError, "parent binding"):
                trainer.load_candidate_look_diagnostic(
                    checkpoint,
                    manifest_path=manifest_path,
                    contract=corrupted_contract,
                    device="cpu",
                )

    def test_loader_rejects_self_rehashed_contract_model_or_objective_drift(self):
        mutations = (
            ("model_seed", lambda contract: contract.update(model_seed=11)),
            (
                "objective",
                lambda contract: contract["dense_objective_spec"].update(step_penalty=999.0),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as parent:
                root = Path(parent)
                contract, checkpoint, manifest_path, _, _ = self._fixture(root)
                corrupted = json.loads(json.dumps(contract))
                corrupted.pop("contract_sha256")
                mutate(corrupted)
                corrupted["contract_sha256"] = trainer.contract_hash(corrupted)
                with self.assertRaisesRegex(trainer.ConstrainedV23TrainerError, "contract mismatch"):
                    trainer.load_candidate_look_diagnostic(
                        checkpoint,
                        manifest_path=manifest_path,
                        contract=corrupted,
                        device="cpu",
                    )

    def test_loader_rejects_consistently_rebound_frozen_contract_drift(self):
        def agent_learning_rate(contract, payload):
            contract["agent_config"]["learning_rate"] = 0.0001
            payload["agent_state"]["config"]["learning_rate"] = 0.0001

        def agent_network(contract, payload):
            contract["agent_config"]["graph_hidden_dim"] = 32
            payload["agent_state"]["config"]["graph_hidden_dim"] = 32

        def agent_replay(contract, payload):
            contract["agent_config"]["replay_capacity"] = 3_000
            payload["agent_state"]["config"]["replay_capacity"] = 3_000

        mutations = (
            ("agent_learning_rate", agent_learning_rate),
            ("agent_network", agent_network),
            ("agent_replay", agent_replay),
            ("dual", lambda contract, payload: contract["dual"].update(learning_rate=0.02)),
            ("environment", lambda contract, payload: contract["environment"].update(arrival_rate=11.0)),
            ("cadence", lambda contract, payload: contract.update(validation_every=10)),
            ("rng_formula", lambda contract, payload: contract.update(training_policy_rng_formula="forged")),
            ("controller", lambda contract, payload: contract.update(controller="forged-controller")),
        )
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as parent:
                root = Path(parent)
                contract, checkpoint, manifest_path, payload, manifest = self._fixture(root)
                corrupted_contract = json.loads(json.dumps(contract))
                corrupted_payload = payload
                mutate(corrupted_contract, corrupted_payload)
                corrupted_contract.pop("contract_sha256")
                corrupted_contract["contract_sha256"] = trainer.contract_hash(
                    corrupted_contract
                )
                corrupted_payload["training_contract_sha256"] = corrupted_contract[
                    "contract_sha256"
                ]
                torch.save(corrupted_payload, checkpoint)
                rebound_manifest = json.loads(json.dumps(manifest))
                rebound_manifest["training_contract_sha256"] = corrupted_contract[
                    "contract_sha256"
                ]
                rebound_manifest["artifacts"][0]["sha256"] = trainer._file_sha256(
                    checkpoint
                )
                self._write_manifest(manifest_path, rebound_manifest)
                with self.assertRaisesRegex(
                    trainer.ConstrainedV23TrainerError,
                    "contract provenance mismatch",
                ):
                    trainer.load_candidate_look_diagnostic(
                        checkpoint,
                        manifest_path=manifest_path,
                        contract=corrupted_contract,
                        device="cpu",
                    )

    def test_loader_rejects_extra_fields_at_every_model_artifact_layer(self):
        mutations = (
            ("manifest", "manifest", "unknown", "manifest schema"),
            ("artifact", "artifact", "unknown", "artifact schema"),
            ("outer_optimizer", "outer", "optimizer", "checkpoint envelope schema"),
            ("outer_replay", "outer", "replay", "checkpoint envelope schema"),
            ("nested", "nested", "unknown", "nested candidate-look checkpoint envelope schema"),
        )
        for name, layer, field, pattern in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as parent:
                root = Path(parent)
                contract, checkpoint, manifest_path, payload, manifest = self._fixture(root)
                rebound_manifest = json.loads(json.dumps(manifest))
                if layer == "manifest":
                    rebound_manifest[field] = True
                elif layer == "artifact":
                    rebound_manifest["artifacts"][0][field] = True
                elif layer == "outer":
                    payload[field] = {"injected": True}
                    torch.save(payload, checkpoint)
                    rebound_manifest["artifacts"][0]["sha256"] = trainer._file_sha256(
                        checkpoint
                    )
                else:
                    payload["agent_state"][field] = True
                    torch.save(payload, checkpoint)
                    rebound_manifest["artifacts"][0]["sha256"] = trainer._file_sha256(
                        checkpoint
                    )
                self._write_manifest(manifest_path, rebound_manifest)
                with self.assertRaisesRegex(trainer.ConstrainedV23TrainerError, pattern):
                    trainer.load_candidate_look_diagnostic(
                        checkpoint,
                        manifest_path=manifest_path,
                        contract=contract,
                        device="cpu",
                    )

class V23CheckpointAndIsolationAdditionalTests(unittest.TestCase):
    def _payload(self):
        agent = ConstrainedV23HierarchyAgent(
            config=_small_config(), seed=7, device="cpu", epsilon=0.0
        )
        agent.set_schedule_state(
            trainer.schedule_for_episode(80, validation=True).to_dict()
        )
        agent.reset_behavior_rng(620_000_000)
        return agent, agent.checkpoint(include_replay=False)

    def test_agent_rejects_parent_operational_discount(self):
        with self.assertRaisesRegex(ValueError, "gamma_operational=1.0"):
            ConstrainedV23Config(**{
                **_small_config().to_dict(),
                "gamma_operational": 0.99,
            })

    def test_v23_config_round_trip_is_strict_and_keeps_parent_schema(self):
        config = _small_config()
        self.assertIsInstance(config, v2.ConstrainedV2Config)
        self.assertEqual(
            ConstrainedV23Config.from_dict(config.to_dict()),
            config,
        )
        corrupted = config.to_dict()
        corrupted["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "configuration field mismatch"):
            ConstrainedV23Config.from_dict(corrupted)

    def test_v23_loader_rejects_authentic_v22_checkpoint(self):
        parent = v22.ConstrainedV22HierarchyAgent(
            config=v2.ConstrainedV2Config(**{
                **_small_config().to_dict(),
                "gamma_operational": 0.99,
            }),
            seed=4,
            device="cpu",
            epsilon=0.0,
        )
        parent.set_schedule_state(
            trainer22.schedule_for_episode(80, validation=True).to_dict()
        )
        payload = parent.checkpoint(include_replay=False)
        with self.assertRaisesRegex(ValueError, "incompatible constrained V2.3"):
            ConstrainedV23HierarchyAgent.from_checkpoint(
                payload,
                device="cpu",
                resumable=False,
                seed=4,
            )

    def test_model_only_candidate_look_state_is_loadable(self):
        agent, _ = self._payload()

        class Runtime:
            @staticmethod
            def checkpoint_state(*, include_replay):
                return agent.checkpoint(include_replay=include_replay)

        checkpoint = trainer._model_only_runtime_checkpoint(Runtime())
        self.assertEqual(
            set(checkpoint["agent_state"]),
            set(trainer._MODEL_ONLY_AGENT_STATE_KEYS),
        )
        self.assertNotIn("optimizer", checkpoint["agent_state"])
        self.assertNotIn("replay", checkpoint["agent_state"])
        restored = ConstrainedV23HierarchyAgent.from_checkpoint(
            checkpoint,
            device="cpu",
            resumable=False,
            seed=7,
        )
        self.assertEqual(dict(restored.schedule_state), dict(agent.schedule_state))

    def test_training_signature_covers_optimizer_dual_and_all_rng_state(self):
        agent, _ = self._payload()
        runtime = object.__new__(ConstrainedV23DevelopmentRuntime)
        runtime.agent = agent

        baseline = runtime._training_signature()
        parameter = next(agent.Q_local.parameters())
        agent.optimizer.state[parameter]["audit_probe"] = torch.tensor(1.0)
        self.assertNotEqual(runtime._training_signature(), baseline)

        optimizer_state = agent.optimizer.state.pop(parameter)
        restored_baseline = runtime._training_signature()
        agent.set_dual_lambda(0.5)
        self.assertNotEqual(runtime._training_signature(), restored_baseline)
        agent.set_dual_lambda(0.0)

        rng_baseline = runtime._training_signature()
        agent.rng.random()
        self.assertNotEqual(runtime._training_signature(), rng_baseline)
        agent.optimizer.state[parameter] = optimizer_state

    def test_expensive_full_signature_runs_only_at_batch_boundaries(self):
        agent, _ = self._payload()
        runtime = object.__new__(ConstrainedV23DevelopmentRuntime)
        runtime.agent = agent
        runtime._schedule_state = trainer.schedule_for_episode(
            80, validation=True
        ).to_dict()
        runtime._validation_batch_signature_before = None

        with mock.patch.object(
            runtime,
            "_full_training_signature",
            wraps=runtime._full_training_signature,
        ) as full_signature:
            started = runtime.begin_validation_batch()
            for _ in range(48):
                runtime._training_signature()
            ended = runtime.end_validation_batch()
        self.assertTrue(started["validation_batch_started"])
        self.assertTrue(ended["training_agent_unchanged"])
        self.assertEqual(full_signature.call_count, 2)

        runtime.begin_validation_batch()
        agent.set_dual_lambda(0.5)
        changed = runtime.end_validation_batch()
        self.assertFalse(changed["training_agent_unchanged"])

    def test_runtime_refuses_all_protected_instance_panels_before_sampling(self):
        runtime = object.__new__(ConstrainedV23DevelopmentRuntime)
        runtime._schedule_state = trainer.schedule_for_episode(
            80, validation=True
        ).to_dict()
        for seed, pattern in ((83_000, "83"), (86_000, "final")):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, pattern):
                    runtime.run_episode(
                        instance_seed=seed,
                        training=False,
                        max_steps=2_000,
                        policy_rng_index=0,
                        policy_rng_seed=620_000_000,
                    )


if __name__ == "__main__":
    unittest.main()
