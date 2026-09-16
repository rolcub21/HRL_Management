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
import train_vcg_constrained_v2_2 as trainer
import viability_graph_constrained_v2 as v2
import viability_graph_constrained_v2_1 as v21
from viability_graph_constrained_v2_2 import (
    ConstrainedV22DevelopmentRuntime,
    ConstrainedV22HierarchyAgent,
    V2_2_DIAGNOSTIC_VERSION,
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
    }
    values.update(overrides)
    return v2.ConstrainedV2Config(**values)


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


class V22PolicyOperatorTests(unittest.TestCase):
    def test_validation_training_false_still_samples_same_induced_policy(self):
        config = _small_config()
        training_agent = ConstrainedV22HierarchyAgent(
            config=config, seed=4, device="cpu", epsilon=0.0
        )
        validation_agent = ConstrainedV22HierarchyAgent(
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
        agent = ConstrainedV22HierarchyAgent(
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
        agent = ConstrainedV22HierarchyAgent(
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


class V22CheckpointAndIsolationTests(unittest.TestCase):
    def _payload(self):
        agent = ConstrainedV22HierarchyAgent(
            config=_small_config(), seed=7, device="cpu", epsilon=0.0
        )
        agent.set_schedule_state(
            trainer.schedule_for_episode(80, validation=True).to_dict()
        )
        agent.reset_behavior_rng(620_000_000)
        return agent, agent.checkpoint(include_replay=False)

    def test_checkpoint_rejects_every_policy_semantic_change(self):
        agent, payload = self._payload()
        restored = ConstrainedV22HierarchyAgent.from_checkpoint(
            payload, device="cpu", resumable=False, seed=7
        )
        self.assertEqual(dict(restored.schedule_state), dict(agent.schedule_state))
        self.assertIsNone(restored.behavior_rng_seed)
        with self.assertRaisesRegex(RuntimeError, "behavior RNG"):
            restored.select(_snapshot(), training=False, epsilon=0.0)
        self.assertEqual(payload["policy_diagnostic_version"], V2_2_DIAGNOSTIC_VERSION)

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
                with self.assertRaisesRegex(ValueError, "incompatible constrained V2.2"):
                    ConstrainedV22HierarchyAgent.from_checkpoint(
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

    def test_training_signature_covers_optimizer_dual_and_all_rng_state(self):
        agent, _ = self._payload()
        runtime = object.__new__(ConstrainedV22DevelopmentRuntime)
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
        runtime = object.__new__(ConstrainedV22DevelopmentRuntime)
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
        runtime = object.__new__(ConstrainedV22DevelopmentRuntime)
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
