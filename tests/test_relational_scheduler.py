from dataclasses import replace
import unittest

import numpy as np
import torch

from example.Options.RetrieveDeliverOption import (
    RetrieveDeliverOption,
    StrictRetrieveDeliverOption,
)
from example.Options.selector_v5 import StorageSelectOptionV5
from example.controller_observation import OnlineManifestTimingObservationEncoder
from example.controller_options import build_controller_options
from example.small_rooms_env import SmallRoomsEnv
from gated_agent import ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE
from options_agent import controller_ids, option_identifier
from PSLAP.reg_selector_v5 import REGV5AssignmentSource, REGV5Config
from relational_scheduler import (
    RELATIONAL_ACTION_INTERFACE,
    RELATIONAL_CONTROLLER_ARCHITECTURE,
    RELATIONAL_POLICY_REALIZATIONS,
    RelationalResidualSchedulerAgent,
    RelationalSchedulerConfig,
    RelationalTransition,
    validate_relational_checkpoint_metadata,
)
from train_relational_track_b import run_episode


def make_system(action_interface=RELATIONAL_ACTION_INTERFACE):
    env = SmallRoomsEnv(
        choose_storage=False, arrival_rate=0.5, proc_mean=50
    )
    base = env.sample_episode_instance(901)
    instance = replace(
        base,
        arrival_steps=(0,) + (10_000,) * (len(env.blocks) - 1),
    )
    selector_config = REGV5Config(
        block_embedding_dim=8,
        candidate_embedding_dim=8,
        context_dim=16,
        replay_size=8,
        batch_size=1,
        min_replay_size=1,
        updates_per_episode=1,
    )
    source = REGV5AssignmentSource(
        env,
        selector_config,
        seed=901,
        learning_enabled=False,
        device="cpu",
    )
    selector = StorageSelectOptionV5(env, source)
    build_controller_options(
        env,
        selector,
        controller_action_interface=action_interface,
        max_defer_steps=10,
    )
    state = env.reset(instance=instance)
    return env, state, selector


def make_agent(env):
    encoder = OnlineManifestTimingObservationEncoder(env)
    config = RelationalSchedulerConfig(
        block_embedding_dim=8,
        global_embedding_dim=8,
        candidate_embedding_dim=8,
        context_dim=16,
        history_length=4,
    )
    agent = RelationalResidualSchedulerAgent(
        env,
        encoder,
        config=config,
        seed=901,
        device="cpu",
        batch_size=1,
        buffer_size=8,
        update_every=1,
    )
    return encoder, agent


class RelationalInterfaceTests(unittest.TestCase):
    def test_new_interface_is_strict_and_old_v5_remains_v1(self):
        relational_env, _, _ = make_system()
        relational_managers = [
            option
            for option in relational_env.options
            if not option.is_primitive
        ]
        self.assertEqual(len(relational_managers), 42)
        self.assertEqual(
            sum(
                isinstance(option, StrictRetrieveDeliverOption)
                for option in relational_managers
            ),
            40,
        )

        old_env, _, _ = make_system(ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE)
        old_managers = [
            option for option in old_env.options if not option.is_primitive
        ]
        self.assertEqual(len(old_managers), 42)
        self.assertEqual(
            sum(
                type(option) is RetrieveDeliverOption
                for option in old_managers
            ),
            40,
        )
        self.assertFalse(
            any(
                isinstance(option, StrictRetrieveDeliverOption)
                for option in old_managers
            )
        )

    def test_checkpoint_contract_is_relational_only(self):
        env, _, _ = make_system()
        encoder, agent = make_agent(env)
        payload = agent.checkpoint_metadata()
        validate_relational_checkpoint_metadata(payload, env, encoder)
        self.assertEqual(
            payload["controller_architecture"],
            RELATIONAL_CONTROLLER_ARCHITECTURE,
        )
        broken = dict(payload)
        broken["controller_action_interface"] = (
            ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE
        )
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            validate_relational_checkpoint_metadata(broken, env, encoder)


class RelationalScoringTests(unittest.TestCase):
    @staticmethod
    def add_due_retrieval(env):
        env.time_steps = 20
        block = env.blocks[1]
        block.arrival_step = 0
        block.position = (4, 4)
        block.storage_location = block.position
        block.stored = True
        block.delivered = False
        block.carrying = False
        block.storage_steps_needed = 1
        block.stored_time_step = 0
        return block

    def test_untrained_residual_starts_from_due_first_baseline(self):
        env, _, _ = make_system()
        due = self.add_due_retrieval(env)
        encoder, agent = make_agent(env)
        state = env.get_current_state()
        choice = agent.select_action(state, eps=0.0)

        self.assertIsInstance(choice, StrictRetrieveDeliverOption)
        self.assertEqual(choice.target_label, due.label)
        audit = agent.decision_audit[-1]
        self.assertEqual(audit["reason"], "baseline_agreement")
        self.assertEqual(audit["selected"], audit["baseline"])
        np.testing.assert_allclose(
            audit["q_values"], audit["baseline_scores"], atol=1e-6
        )
        np.testing.assert_allclose(audit["residuals"], 0.0, atol=1e-7)
        self.assertIsInstance(encoder, OnlineManifestTimingObservationEncoder)

    def test_kind_safe_policy_rejects_cross_kind_override(self):
        env, _, _ = make_system()
        _, agent = make_agent(env)
        agent.set_policy_realization("kind_safe_residual_map")
        candidates = (
            type("Candidate", (), {"kind": "accept"})(),
            type("Candidate", (), {"kind": "retrieve"})(),
        )

        selected, reason = agent._deployment_choice(
            np.asarray((10.0, 20.0)), candidates, baseline_index=0
        )

        self.assertEqual(selected, 0)
        self.assertEqual(reason, "cross_kind_override_rejected")

    def test_kind_safe_policy_keeps_retrieval_reordering(self):
        env, _, _ = make_system()
        _, agent = make_agent(env)
        agent.set_policy_realization("kind_safe_residual_map")
        candidates = (
            type("Candidate", (), {"kind": "retrieve"})(),
            type("Candidate", (), {"kind": "retrieve"})(),
            type("Candidate", (), {"kind": "defer"})(),
        )

        selected, reason = agent._deployment_choice(
            np.asarray((10.0, 12.0, 0.0)), candidates, baseline_index=0
        )

        self.assertEqual(selected, 1)
        self.assertEqual(reason, "learned_override")

    def test_policy_realizations_are_explicit(self):
        env, _, _ = make_system()
        _, agent = make_agent(env)
        self.assertIn("kind_safe_residual_map", RELATIONAL_POLICY_REALIZATIONS)
        with self.assertRaisesRegex(ValueError, "policy must be one of"):
            agent.set_policy_realization("unknown")

    def test_candidate_and_block_permutations_only_permute_candidate_scores(self):
        env, state, _ = make_system()
        self.add_due_retrieval(env)
        _, agent = make_agent(env)
        relational_state = agent.capture_state(state)
        candidates = agent.builder.build(state, relational_state)
        features, baselines = agent._candidate_arrays(candidates)
        network = agent.Q_local

        global_tensor = torch.from_numpy(
            relational_state.global_features
        ).unsqueeze(0)
        blocks = torch.from_numpy(relational_state.block_features).unsqueeze(0)
        mask = torch.from_numpy(relational_state.block_mask).unsqueeze(0)
        candidate_tensor = torch.from_numpy(features).unsqueeze(0)
        baseline_tensor = torch.from_numpy(baselines).unsqueeze(0)
        q, risk, residual = network(
            global_tensor,
            blocks,
            mask,
            candidate_tensor,
            baseline_tensor,
        )

        candidate_order = np.arange(len(candidates))[::-1].copy()
        block_order = np.arange(len(env.blocks))[::-1].copy()
        permuted_q, permuted_risk, permuted_residual = network(
            global_tensor,
            blocks[:, block_order],
            mask[:, block_order],
            candidate_tensor[:, candidate_order],
            baseline_tensor[:, candidate_order],
        )
        torch.testing.assert_close(
            permuted_q, q[:, candidate_order], atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            permuted_risk, risk[:, candidate_order], atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            permuted_residual,
            residual[:, candidate_order],
            atol=1e-6,
            rtol=0,
        )

    def test_action_identity_is_input_data_not_output_position(self):
        env, state, _ = make_system()
        self.add_due_retrieval(env)
        _, agent = make_agent(env)
        relational_state = agent.capture_state(state)
        candidates = agent.builder.build(state, relational_state)
        ids = [option_identifier(item.option) for item in candidates]

        self.assertEqual(len(agent.Q_local.residual_head.weight), 1)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            controller_ids(agent.manager_options),
            sorted(controller_ids(agent.manager_options)),
        )
        self.assertGreater(len(candidates), 1)

    def test_variable_candidate_replay_updates_shared_residual(self):
        env, state, _ = make_system()
        _, agent = make_agent(env)
        relational_state = agent.capture_state(state)
        candidates = agent.builder.build(state, relational_state)
        selected = candidates[0]
        next_features, next_baselines = agent._candidate_arrays(candidates)
        agent.replay.add(
            RelationalTransition(
                state=relational_state.copy(),
                candidate_features=selected.features.copy(),
                candidate_baseline=selected.baseline_score,
                reward=selected.baseline_score + 5.0,
                next_state=relational_state.copy(),
                next_candidate_features=next_features,
                next_candidate_baselines=next_baselines,
                done=True,
                duration=7,
            )
        )
        before = agent.Q_local.residual_head.bias.detach().clone()
        agent.step_count = 1
        loss = agent.learn()

        self.assertIsNotNone(loss)
        self.assertIn("mean_abs_td_error", loss)
        self.assertFalse(
            torch.equal(before, agent.Q_local.residual_head.bias.detach())
        )


class RelationalEpisodeFlowTests(unittest.TestCase):
    def test_run_episode_exposes_full_storage_flow_schema(self):
        env, _, selector = make_system()
        _, agent = make_agent(env)
        instance = env.current_episode_instance

        result = run_episode(
            agent,
            selector,
            env,
            instance,
            max_steps=1,
            target_window=20.0,
            training=False,
        )

        expected = {
            "storage_flow_metric_contract",
            "storage_flow_observation_end_step",
            "storage_flow_manifest_count",
            "storage_flow_arrived_count",
            "storage_flow_completed_count",
            "storage_flow_right_censored_count",
            "storage_flow_not_yet_arrived_count",
            "storage_flow_unfinished_count",
            "storage_flow_completion_rate",
            "storage_flow_arrived_completion_rate",
            "storage_flow_fully_observed",
            "mean_storage_flow_time",
            "median_storage_flow_time",
            "p90_storage_flow_time",
            "p95_storage_flow_time",
            "max_storage_flow_time",
            "completed_storage_flow_times",
            "right_censored_storage_flow_ages",
            "storage_flow_records",
        }
        self.assertTrue(expected.issubset(result))
        self.assertEqual(result["storage_flow_observation_end_step"], 1)
        self.assertEqual(result["storage_flow_manifest_count"], 40)
        self.assertEqual(len(result["storage_flow_records"]), 40)
        self.assertFalse(result["storage_flow_fully_observed"])


if __name__ == "__main__":
    unittest.main()
