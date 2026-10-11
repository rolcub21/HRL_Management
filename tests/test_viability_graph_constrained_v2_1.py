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
from train_vcg_constrained_v2 import (
    build_parser as build_v2_parser,
    build_training_contract as build_v2_contract,
)
from train_vcg_constrained_v2_1 import (
    BACKUP_VERSION,
    CHECKPOINT_FAMILY,
    CONTROLLER_ARCHITECTURE,
    METHOD_VERSION,
    POLICY_SCHEDULE_PROTOCOL,
    build_parser,
    build_training_contract,
    schedule_for_episode,
)
import viability_graph_constrained_v2 as v2
from viability_graph_constrained_v2_1 import (
    ConstrainedV21HierarchyAgent,
    V2_1_DIAGNOSTIC_VERSION,
    build_v2_1_development_runtime,
)
from tests.test_viability_graph_constrained_v2 import _record


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


class V21ScheduleAndBackupTests(unittest.TestCase):
    def test_schedule_setter_requires_exact_frozen_mapping(self):
        agent = ConstrainedV21HierarchyAgent(
            config=_small_config(), seed=1, device="cpu", epsilon=0.0
        )
        with self.assertRaisesRegex(RuntimeError, "schedule has not been installed"):
            _ = agent.group_temperature
        exact = schedule_for_episode(40).to_dict()
        agent.set_schedule_state(exact)
        self.assertEqual(dict(agent.schedule_state), exact)
        self.assertEqual(
            agent.within_group_temperature_values,
            tuple(exact["within_group_temperatures"]),
        )
        self.assertEqual(agent.group_temperature, exact["group_temperature"])

        corrupted = dict(exact, group_temperature=0.123)
        with self.assertRaisesRegex(ValueError, "frozen clock"):
            agent.set_schedule_state(corrupted)
        extra = dict(exact, unowned_field=True)
        with self.assertRaisesRegex(ValueError, "frozen clock"):
            agent.set_schedule_state(extra)

    def test_select_and_td_backup_read_the_same_active_temperatures(self):
        from tests.test_viability_candidates_hold_v2 import compact_env, make_stored

        agent = ConstrainedV21HierarchyAgent(
            config=_small_config(), seed=2, device="cpu", epsilon=0.0
        )
        schedule = schedule_for_episode(40).to_dict()
        agent.set_schedule_state(schedule)

        env = compact_env(1, seed=9951)
        make_stored(env.blocks[0], (2, 2), duration=30)
        env.current_state = (1, 1)
        env.time_steps = 5
        snapshot = enumerate_viability_candidates_hold_v2(
            env,
            idle_budget=PrimitiveIdleBudgetStateV2(20, 3),
            remaining_episode_steps=7,
            recovery_witness_forced=False,
            search_config=ViabilitySearchConfig(
                max_nodes=None, search_order="breadth_first"
            ),
            hold_rule=CertifiedHoldRuleV2(10, 20),
        )

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

        with mock.patch.object(
            v2, "shared_lagrangian_policy_values", side_effect=recording
        ):
            agent.select(snapshot, training=True, epsilon=0.0)
            transition = v2.ConstrainedV2Transition(
                chosen=_record(0),
                operational_reward=1.0,
                physical_rehandle_cost=0.0,
                duration=1,
                next_candidates=(_record(0, "next-a"), _record(1, "next-b")),
                done=False,
                behavior_lambda=0.0,
            )
            agent._td_batch([transition])

        self.assertGreaterEqual(len(calls), 2)
        expected = (
            tuple(schedule["within_group_temperatures"]),
            schedule["group_temperature"],
        )
        self.assertTrue(all(call == calls[0] for call in calls))
        for actual, canonical in zip(calls[0][0], expected[0]):
            self.assertAlmostEqual(actual, canonical, places=7)
        self.assertAlmostEqual(calls[0][1], expected[1], places=7)
        diagnostic = agent.policy_diagnostic_state()
        self.assertEqual(diagnostic["protocol"], V2_1_DIAGNOSTIC_VERSION)
        self.assertEqual(diagnostic["count"], 1)
        self.assertGreaterEqual(diagnostic["mean_outer_entropy"], 0.0)
        self.assertGreater(diagnostic["mean_outer_map_probability"], 0.0)


class V21CheckpointIsolationTests(unittest.TestCase):
    def test_v2_1_round_trip_and_mutual_v2_rejection(self):
        config = _small_config()
        agent = ConstrainedV21HierarchyAgent(
            config=config, seed=3, device="cpu", epsilon=0.0
        )
        agent.set_schedule_state(schedule_for_episode(80, validation=True).to_dict())
        payload = agent.checkpoint(include_replay=False)
        self.assertEqual(payload["checkpoint_family"], CHECKPOINT_FAMILY)
        self.assertEqual(payload["controller_architecture"], CONTROLLER_ARCHITECTURE)
        self.assertEqual(payload["backup_version"], BACKUP_VERSION)
        restored = ConstrainedV21HierarchyAgent.from_checkpoint(
            payload, device="cpu", resumable=False, seed=3
        )
        self.assertEqual(dict(restored.schedule_state), dict(agent.schedule_state))
        self.assertEqual(restored.dual_lambda, agent.dual_lambda)
        for field, invalid in {
            "single_shared_lagrangian_policy_for_both_target_heads": False,
            "entropy_or_kl_added_to_raw_component_targets": True,
            "exact_safe_mask_authoritative": False,
            "policy_diagnostic_version": "wrong_diagnostic",
            "policy_diagnostic_distribution": "executed_randomness",
            "training_policy": "wrong_training_policy",
            "deployment_policy": "stochastic_deployment",
        }.items():
            with self.subTest(field=field):
                corrupted = dict(payload)
                corrupted[field] = invalid
                with self.assertRaisesRegex(
                    ValueError, "incompatible constrained V2.1"
                ):
                    ConstrainedV21HierarchyAgent.from_checkpoint(
                        corrupted, device="cpu", resumable=False, seed=3
                    )

        with self.assertRaisesRegex(ValueError, "incompatible constrained V2 checkpoint"):
            v2.ConstrainedV2HierarchyAgent.from_checkpoint(
                payload, device="cpu", resumable=False, seed=3
            )
        base = v2.ConstrainedV2HierarchyAgent(
            config=config, seed=3, device="cpu", epsilon=0.0
        ).checkpoint(include_replay=False)
        with self.assertRaisesRegex(ValueError, "incompatible constrained V2.1"):
            ConstrainedV21HierarchyAgent.from_checkpoint(
                base, device="cpu", resumable=False, seed=3
            )


class V21RuntimeIntegrationTests(unittest.TestCase):
    def _args_contract(self, output):
        args = build_parser().parse_args(
            [
                "--output-dir",
                str(output),
                "--validation-seeds",
                "84000",
                "--number-blocks",
                "1",
                "--max-steps",
                "1",
                "--device",
                "cpu",
            ]
        )
        return args, build_training_contract(args)

    def test_factory_contract_and_real_map_clone_isolation(self):
        with tempfile.TemporaryDirectory() as parent:
            args, contract = self._args_contract(Path(parent) / "runtime")
            runtime = build_v2_1_development_runtime(args=args, contract=contract)
            self.assertEqual(runtime.core_contract["checkpoint_family"], CHECKPOINT_FAMILY)
            self.assertEqual(runtime.core_contract["controller"], CONTROLLER_ARCHITECTURE)
            self.assertEqual(
                runtime.core_contract["policy_schedule_protocol"],
                POLICY_SCHEDULE_PROTOCOL,
            )
            self.assertEqual(
                runtime.core_contract["agent_config"], contract["agent_config"]
            )

            training_schedule = schedule_for_episode(1).to_dict()
            runtime.set_schedule_state(training_schedule)
            training = runtime.run_episode(
                instance_seed=60_000_000, training=True, max_steps=1
            )
            self.assertEqual(training["method_version"], METHOD_VERSION)
            self.assertEqual(training["policy_mode"], "regularized_sample")
            cumulative_training_diagnostics = (
                runtime.agent.policy_diagnostic_state()["count"]
            )
            self.assertGreater(cumulative_training_diagnostics, 0)

            map_schedule = schedule_for_episode(10, validation=True).to_dict()
            runtime.set_schedule_state(map_schedule)
            evaluation = runtime.run_episode(
                instance_seed=84_000, training=False, max_steps=1
            )
            self.assertEqual(evaluation["policy_mode"], "map")
            self.assertFalse(evaluation["evaluation_learning"])
            self.assertTrue(evaluation["training_agent_unchanged"])
            self.assertEqual(
                evaluation["policy_diagnostic_decisions"],
                evaluation["macro_decisions"],
            )
            # The evaluation delta is clone-local, not cumulative training
            # history copied into the clone checkpoint.
            self.assertLess(
                evaluation["policy_diagnostic_decisions"],
                cumulative_training_diagnostics
                + evaluation["policy_diagnostic_decisions"],
            )
            self.assertEqual(
                runtime.agent.policy_diagnostic_state()["count"],
                cumulative_training_diagnostics,
            )

    def test_factory_rejects_v2_contract_and_policy_mode_mismatch(self):
        with tempfile.TemporaryDirectory() as parent:
            args, contract = self._args_contract(Path(parent) / "runtime")
            runtime = build_v2_1_development_runtime(args=args, contract=contract)
            runtime.set_schedule_state(schedule_for_episode(1).to_dict())
            with self.assertRaisesRegex(RuntimeError, "training flag disagrees"):
                runtime.run_episode(
                    instance_seed=84_000, training=False, max_steps=1
                )

            v2_args = build_v2_parser().parse_args(
                ["--output-dir", str(Path(parent) / "v2-contract")]
            )
            v2_contract = build_v2_contract(v2_args)
            with self.assertRaisesRegex(ValueError, "V2.1 runtime contract mismatch"):
                build_v2_1_development_runtime(
                    args=v2_args, contract=v2_contract
                )


if __name__ == "__main__":
    unittest.main()
