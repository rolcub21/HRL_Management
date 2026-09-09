from dataclasses import replace
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from example.Options.AcceptStoreOption import AcceptStoreOption
from example.Options.RetrieveDeliverOption import RetrieveDeliverOption
from example.Options.StrategicDeferOption import StrategicDeferOption
from example.Options.selector_v5 import StorageSelectOptionV5
from example.controller_observation import OnlineManifestTimingObservationEncoder
from example.controller_options import build_controller_options
from example.small_rooms_env import SmallRoomsEnv
from gated_agent import (
    ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE,
    ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE,
    GatedAtomicInboundSchedulerAgent,
)
from options_agent import (
    CHECKPOINT_SCHEMA_VERSION,
    controller_ids,
    option_identifier,
    validate_checkpoint_metadata,
)
from PSLAP.reg_selector_v5 import REGV5AssignmentSource, REGV5Config
from PSLAP.checkpoint_identity import selector_deployment_digest
from track_b_evaluate import evaluate_one as evaluate_track_b_one


ACCEPT_STORE_ID = "option:AcceptStore:v1"
DEFER_ID = "option:DeferUntilEvent:v1"


def make_inbound_system(*, with_agent=False, training_policy="map"):
    """Build one observable inbound job and no subsequent arrival pressure."""

    random.seed(811)
    np.random.seed(811)
    torch.manual_seed(811)
    env = SmallRoomsEnv(
        choose_storage=False,
        arrival_rate=0.5,
        proc_mean=50,
    )
    base = env.sample_episode_instance(811)
    instance = replace(
        base,
        arrival_steps=(0,) + (10_000,) * (len(env.blocks) - 1),
    )
    config = REGV5Config(
        block_embedding_dim=8,
        candidate_embedding_dim=8,
        context_dim=16,
        replay_size=4,
        batch_size=1,
        min_replay_size=1,
        updates_per_episode=1,
    )
    source = REGV5AssignmentSource(
        env,
        config,
        seed=811,
        learning_enabled=False,
        device="cpu",
    )
    selector = StorageSelectOptionV5(env, source)
    build_controller_options(
        env,
        selector,
        controller_action_interface=(
            ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE
        ),
        max_defer_steps=10,
    )
    state = env.reset(instance=instance)
    if not with_agent:
        return env, state, selector, None, None

    encoder = OnlineManifestTimingObservationEncoder(env)
    agent = GatedAtomicInboundSchedulerAgent(
        env=env,
        state_size=encoder.feature_size,
        action_size=len(
            [option for option in env.options if option.is_primitive]
        ),
        n_episodes=1,
        n_steps=200,
        batch_size=1,
        buffer_size=100,
        reward_clip=1_000_000.0,
        gamma=0.99,
        training_policy=training_policy,
        state_encoder=encoder,
        controller_observation_metadata=encoder.metadata(),
        disable_tensorboard=True,
        device="cpu",
        verbose=False,
        max_defer_steps=10,
    )
    return env, state, selector, encoder, agent


class AtomicInboundInterfaceTests(unittest.TestCase):
    def test_factory_builds_exact_stable_42_output_interface(self):
        env, _, _, _, _ = make_inbound_system()
        manager = sorted(
            (option for option in env.options if not option.is_primitive),
            key=option_identifier,
        )
        ids = controller_ids(manager)
        expected = {
            ACCEPT_STORE_ID,
            DEFER_ID,
            *{
                f"option:RetrieveDeliver:block-{index:04d}"
                for index in range(len(env.blocks))
            },
        }

        self.assertEqual(len(manager), 42)
        self.assertEqual(set(ids), expected)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(
            sum(isinstance(option, AcceptStoreOption) for option in manager),
            1,
        )
        self.assertEqual(
            sum(isinstance(option, StrategicDeferOption) for option in manager),
            1,
        )
        self.assertEqual(
            sum(isinstance(option, RetrieveDeliverOption) for option in manager),
            40,
        )
        for split_id in (
            "option:PickupOption",
            "option:StoreOption",
            "option:StorageSelectOption",
            "option:PickupRipeOption",
            "option:DeliverOption",
        ):
            self.assertNotIn(split_id, ids)

    def test_inbound_state_mask_and_selection_force_accept_store_purely(self):
        env, state, selector, encoder, agent = make_inbound_system(
            with_agent=True,
            training_policy="map",
        )
        try:
            option_mask, primitive_mask = agent._availability_masks(
                state, done=False
            )
            available_ids = {
                option_identifier(option)
                for option, available in zip(
                    agent.manager_options, option_mask
                )
                if available
            }
            self.assertEqual(available_ids, {ACCEPT_STORE_ID})
            self.assertFalse(primitive_mask.any())

            manager_values = torch.full(
                (len(agent.manager_options),), -100.0
            )
            manager_values[
                next(
                    index
                    for index, option in enumerate(agent.manager_options)
                    if option_identifier(option) == DEFER_ID
                )
            ] = 100.0
            primitive_values = torch.full(
                (len(agent.primitive_options),), 200.0
            )
            features = encoder(state).copy()
            with patch.object(
                agent.Q_manager_local,
                "forward",
                return_value=manager_values,
            ), patch.object(
                agent.Q_worker_local,
                "forward",
                return_value=primitive_values,
            ):
                greedy = agent.select_action(
                    state, eps=0.0, state_features=features
                )

            # Force the inherited epsilon branch toward the primitive mode.
            # A correct v5 selector ignores that branch because AcceptStore is
            # the sole structurally admissible controller at this epoch.
            with patch("gated_agent.random.random", side_effect=(0.0, 0.99)):
                exploratory = agent.select_action(
                    state, eps=1.0, state_features=features
                )

            self.assertIsInstance(greedy, AcceptStoreOption)
            self.assertIsInstance(exploratory, AcceptStoreOption)
            self.assertEqual(selector.decision_count, 0)
            self.assertIsNone(env.blocks[0].storage_location)
            self.assertFalse(env.blocks[0].carrying)
            self.assertEqual(agent.decision_counts["primitive"], 0)
            self.assertEqual(
                set(agent.control_decisions), {ACCEPT_STORE_ID}
            )
        finally:
            agent.writer.close()


class AtomicInboundLifecycleTests(unittest.TestCase):
    def test_real_reg_assignment_is_once_and_replay_is_one_atomic_record(self):
        env, state, selector, encoder, agent = make_inbound_system(
            with_agent=True,
            training_policy="map",
        )
        try:
            option = next(
                option
                for option in agent.manager_options
                if isinstance(option, AcceptStoreOption)
            )
            self.assertTrue(option.initiation(state))
            start_features = encoder(state).copy()
            agent.current_option = option
            agent.option_start_state = state
            agent.option_start_features = start_features.copy()
            agent.option_reward_traj = []
            rewards = []
            infos = []
            target = env.blocks[0]
            expected_pickup_actions = (
                len(
                    env.plan_path_heuristic(
                        env.current_state,
                        target.position,
                        ignore_block=target,
                    )
                )
                + 1
            )
            assignment_action = None
            expected_store_actions = None

            with patch.object(
                selector.source,
                "propose",
                wraps=selector.source.propose,
            ) as propose:
                for _ in range(200):
                    state_features = encoder(state).copy()
                    proposal_count = propose.call_count
                    action = option.policy(state)
                    if propose.call_count != proposal_count:
                        assignment_action = action
                        expected_store_actions = (
                            len(
                                env.plan_path_heuristic(
                                    env.current_state,
                                    target.storage_location,
                                    ignore_block=target,
                                )
                            )
                            + 1
                        )
                    next_state, reward, done, info = env.step(action)
                    next_features = encoder(next_state).copy()
                    selector.on_step(reward, info)
                    terminated = option.termination(next_state)
                    agent.process_step(
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                        terminated,
                        state_features=state_features,
                        next_state_features=next_features,
                    )
                    rewards.append(float(reward))
                    infos.append(info)
                    state = next_state
                    if terminated:
                        break
                else:
                    self.fail("AcceptStoreOption did not terminate")

            self.assertEqual(propose.call_count, 1)
            self.assertIsNotNone(assignment_action)
            self.assertNotEqual(
                assignment_action,
                env.ACTION_IDS["WAIT"],
                "REG assignment is zero-duration and must immediately emit "
                "the first storage action",
            )
            self.assertEqual(
                len(rewards),
                expected_pickup_actions + expected_store_actions,
            )
            self.assertEqual(selector.decision_count, 1)
            self.assertEqual(len(selector.decisions), 1)
            self.assertTrue(selector.decisions[0]["valid"])
            self.assertEqual(selector.decisions[0]["block_label"], target.label)
            self.assertEqual(
                [info.get("picked_block") for info in infos if info.get("picked_block")],
                [target.label],
            )
            self.assertEqual(
                [info.get("stored_block") for info in infos if info.get("stored_block")],
                [target.label],
            )
            self.assertFalse(any(info.get("illegal_drop") for info in infos))
            self.assertTrue(target.stored)
            self.assertFalse(target.carrying)
            self.assertFalse(target.delivered)
            self.assertEqual(option.episode_success_count, 1)
            self.assertEqual(option.episode_failure_count, 0)
            self.assertEqual(option.last_outcome["actual_steps"], len(rewards))

            self.assertEqual(len(agent.ManagerBuffer), 1)
            self.assertEqual(len(agent.WorkerBuffer), 0)
            experience = agent.ManagerBuffer.memory[0]
            self.assertEqual(
                experience.option_idx,
                agent.manager_options.index(option),
            )
            self.assertEqual(experience.k, len(rewards))
            self.assertAlmostEqual(
                float(experience.reward),
                sum(
                    (agent.gamma**index) * reward
                    for index, reward in enumerate(rewards)
                ),
            )
            self.assertEqual(float(experience.done), 0.0)
            np.testing.assert_array_equal(experience.state, start_features)

            continuation_ids = {
                option_identifier(candidate)
                for candidate, available in zip(
                    agent.manager_options, experience.next_option_mask
                )
                if available
            }
            self.assertEqual(
                continuation_ids,
                {DEFER_ID, "option:RetrieveDeliver:block-0000"},
            )
            self.assertFalse(experience.next_primitive_mask.any())
        finally:
            agent.writer.close()


class AtomicInboundCheckpointTests(unittest.TestCase):
    def test_checkpoint_metadata_and_controller_indices_define_new_abi(self):
        _, state, _, encoder, agent = make_inbound_system(with_agent=True)
        try:
            metadata = agent.checkpoint_metadata()
            self.assertEqual(
                metadata["controller_architecture"],
                ATOMIC_INBOUND_SCHEDULER_ARCHITECTURE,
            )
            self.assertEqual(
                metadata["controller_action_interface"],
                ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE,
            )
            self.assertEqual(
                metadata["decision_epoch_contract"],
                "mandatory_inbound_then_schedule_v1",
            )
            self.assertEqual(
                metadata["scheduler_decision_state_contract"],
                "forced_inbound_else_retrieve_or_defer_v1",
            )
            self.assertEqual(
                metadata["controller_backup"],
                "masked_macro_smdp_optimality_v1",
            )
            self.assertEqual(
                tuple(metadata["active_decision_modes"]),
                ("macro_option",),
            )
            self.assertEqual(
                metadata["primitive_decision_mode"],
                "masked_at_scheduler_epochs_v1",
            )
            self.assertFalse(metadata["primitive_decisions_enabled"])
            self.assertEqual(
                metadata["exploration_reference"],
                "half_dispatch_group_half_defer_then_uniform_block_v1",
            )
            self.assertEqual(
                metadata["forced_decision_epsilon_clock"],
                "does_not_advance_v1",
            )
            self.assertEqual(
                metadata["accept_store_option_version"],
                "accept_store_option_v1",
            )
            self.assertEqual(
                metadata["selector_assignment_duration"],
                "zero_environment_steps_v1",
            )
            self.assertEqual(len(agent.manager_options), 42)
            self.assertEqual(
                tuple(agent.Q_manager_local(torch.from_numpy(encoder(state))).shape),
                (42,),
            )

            payload = {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                **metadata,
                "manager_option_ids": controller_ids(agent.manager_options),
                "primitive_action_ids": controller_ids(
                    agent.primitive_options
                ),
            }
            validate_checkpoint_metadata(
                payload,
                agent.manager_options,
                agent.primitive_options,
                selector=None,
            )

            mismatched = dict(payload)
            mismatched["manager_option_ids"] = list(
                reversed(payload["manager_option_ids"])
            )
            with self.assertRaisesRegex(
                ValueError, "manager_option_ids mismatch"
            ):
                validate_checkpoint_metadata(
                    mismatched,
                    agent.manager_options,
                    agent.primitive_options,
                    selector=None,
                )
        finally:
            agent.writer.close()


class AtomicInboundEvaluationFlowTests(unittest.TestCase):
    def test_learned_evaluator_exposes_full_storage_flow_schema(self):
        env, _, selector, _, agent = make_inbound_system(with_agent=True)
        try:
            instance = env.current_episode_instance
            selector_payload = selector.checkpoint(
                training_lambda=0.5,
                training_mu=50.0,
                completed_training_episodes=0,
            )
            controller_payload = {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                **agent.checkpoint_metadata(),
                "manager_option_ids": controller_ids(agent.manager_options),
                "primitive_action_ids": controller_ids(
                    agent.primitive_options
                ),
                "manager_state_dict": agent.Q_manager_local.state_dict(),
                "worker_state_dict": agent.Q_worker_local.state_dict(),
                "gamma": agent.gamma,
                "retrieval_duration_definition": (
                    "atomic_named_retrieval_v1"
                ),
                "selector_deployment_digest": selector_deployment_digest(
                    selector_payload
                ),
            }
            args = SimpleNamespace(
                lam=0.5,
                mu=50.0,
                instance=None,
                device="cpu",
                policy="map",
                tau_option=float(agent.tau_option),
                tau_primitive=float(agent.tau_primitive),
                tau_mode=float(agent.tau_mode),
                allow_legacy_controller=False,
                max_steps=1,
                target_window=20.0,
                save_instances_dir=None,
            )

            result = evaluate_track_b_one(
                args,
                811,
                controller_payload,
                selector_payload,
                episode_instance=instance,
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
        finally:
            agent.writer.close()


if __name__ == "__main__":
    unittest.main()
