import pickle
import random
import tempfile
import unittest
import warnings

import numpy as np
import torch

from example.small_rooms_env import SmallRoomsEnv
from example.Options.DeliverOption import DeliverOption
from example.Options.GAStorageSelectOption import GAStorageSelectOption
from example.Options.PickupRipeOption import PickupRipeOption
from example.Options.pickupOption import PickupOption
from example.Options.selector import StorageSelectOption
from example.Options.storeOption import StoreOption
from options_agent import (
    CHECKPOINT_SCHEMA_VERSION,
    accumulate_selector_pending,
    canonical_manager_options,
    canonical_primitive_options,
    controller_ids,
    validate_checkpoint_metadata,
)
from primitive_option import PrimitiveOption


def execute_option(env, option, state, limit=100):
    if not option.initiation(state):
        return state, False
    for _ in range(limit):
        state, _, done, _ = env.step(option.policy(state))
        if option.termination(state):
            return state, True
        if done:
            return state, False
    return state, False


class ControllerMappingTests(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)

    def make_options(self, env):
        options = [PrimitiveOption(action, env) for action in env.get_action_space()]
        options.extend(
            [
                PickupOption(env),
                PickupRipeOption(env),
                DeliverOption(env),
                StoreOption(env),
                StorageSelectOption(env),
            ]
        )
        random.shuffle(options)
        return set(options)

    def test_controller_indices_are_canonical(self):
        env = SmallRoomsEnv(choose_storage=False)
        options = self.make_options(env)
        manager = canonical_manager_options(options)
        primitive = canonical_primitive_options(options)

        self.assertEqual(controller_ids(manager), sorted(controller_ids(manager)))
        self.assertEqual(
            [option.action for option in primitive],
            list(env.get_action_space()),
        )

    def test_checkpoint_metadata_rejects_mismatch_and_unverified_legacy(self):
        env = SmallRoomsEnv(choose_storage=False)
        options = self.make_options(env)
        manager = canonical_manager_options(options)
        primitive = canonical_primitive_options(options)
        selector = next(o for o in options if isinstance(o, StorageSelectOption))
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "manager_option_ids": controller_ids(manager),
            "primitive_action_ids": controller_ids(primitive),
            "selector_feature_version": selector.FEATURE_VERSION,
            "selector_return_definition": selector.return_definition,
            "selector_gamma": selector.gamma,
        }

        validate_checkpoint_metadata(payload, manager, primitive, selector)
        bad = dict(payload)
        bad["primitive_action_ids"] = list(reversed(payload["primitive_action_ids"]))
        with self.assertRaisesRegex(ValueError, "primitive_action_ids mismatch"):
            validate_checkpoint_metadata(bad, manager, primitive, selector)
        bad_gamma = dict(payload)
        bad_gamma["selector_gamma"] = 0.5
        with self.assertRaisesRegex(ValueError, "selector_gamma mismatch"):
            validate_checkpoint_metadata(bad_gamma, manager, primitive, selector)
        with self.assertRaisesRegex(ValueError, "Legacy checkpoint"):
            validate_checkpoint_metadata({}, manager, primitive, selector)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_checkpoint_metadata(
                {}, manager, primitive, selector, allow_legacy=True
            )
        self.assertEqual(len(caught), 1)

        older = dict(payload)
        older["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION - 1
        older.pop("selector_return_definition")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_checkpoint_metadata(
                older,
                manager,
                primitive,
                selector,
                allow_legacy=True,
            )
        self.assertEqual(len(caught), 2)


class OptionLifecycleTests(unittest.TestCase):
    def setUp(self):
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)

    def test_store_puts_down_immediately_when_already_at_target(self):
        env = SmallRoomsEnv(choose_storage=False, arrival_rate=0.2, proc_mean=20)
        env.reset()
        block = env.blocks[0]
        target = env.storage_positions[0]
        env.current_state = target
        block.position = target
        block.carrying = True
        block.storage_location = target
        option = StoreOption(env)

        self.assertTrue(option.initiation(env.get_current_state()))
        action = option.policy(env.get_current_state())
        self.assertEqual(action, env.ACTION_IDS["PUTDOWN"])
        next_state, _, _, _ = env.step(action)
        self.assertTrue(option.termination(next_state))
        self.assertTrue(block.stored)

    def test_store_refuses_an_occupied_assignment(self):
        env = SmallRoomsEnv(choose_storage=False)
        env.reset()
        carried, occupant = env.blocks[:2]
        target = env.storage_positions[0]
        carried.carrying = True
        carried.position = env.current_state
        carried.storage_location = target
        occupant.position = target

        self.assertFalse(StoreOption(env).initiation(env.get_current_state()))

    def test_deliver_uses_a_reachable_alternate_exit(self):
        env = SmallRoomsEnv(choose_storage=False, arrival_rate=0.2, proc_mean=20)
        env.reset()
        for block in env.blocks:
            block.position = None
            block.arrival_step = 999
        carried, occupant = env.blocks[:2]
        blocked_exit = env.exit_cells[0]
        env.current_state = (env.grid_rows - 2, blocked_exit[1])
        carried.position = env.current_state
        carried.carrying = True
        carried.stored = True
        carried.stored_time_step = 0
        carried.storage_location = env.storage_positions[0]
        occupant.position = blocked_exit
        occupant.stored = True
        occupant.stored_time_step = 0
        occupant.storage_location = env.storage_positions[1]
        option = DeliverOption(env)

        self.assertTrue(option.initiation(env.get_current_state()))
        self.assertNotEqual(option.path[-1], blocked_exit)
        state, terminated = execute_option(
            env, option, env.get_current_state(), limit=10
        )
        self.assertTrue(terminated)
        self.assertTrue(carried.delivered)

    def test_pickup_only_targets_the_pickup_cell(self):
        env = SmallRoomsEnv(choose_storage=False)
        env.reset()
        for block in env.blocks:
            block.position = None
            block.arrival_step = 999
        env.blocks[0].position = env.storage_positions[0]

        self.assertFalse(PickupOption(env).initiation(env.get_current_state()))

    def test_ripe_pickup_ignores_a_block_with_no_exit_route(self):
        env = SmallRoomsEnv(choose_storage=False, proc_mean=1)
        env.reset()
        for block in env.blocks:
            block.position = None
            block.arrival_step = 999
        target = env.blocks[0]
        target.position = env.storage_positions[0]
        target.storage_location = target.position
        target.stored = True
        target.stored_time_step = 0
        target.storage_steps_needed = 1
        for blocker, exit_cell in zip(env.blocks[1:], env.exit_cells):
            blocker.position = exit_cell

        self.assertFalse(
            PickupRipeOption(env).initiation(env.get_current_state())
        )

    def test_nominal_option_chain_completes(self):
        env = SmallRoomsEnv(
            choose_storage=False, arrival_rate=0.2, proc_mean=20
        )
        state = env.reset()
        state, terminated = execute_option(env, PickupOption(env), state)
        self.assertTrue(terminated)
        block = next(block for block in env.blocks if block.carrying)

        selector = StorageSelectOption(env)
        selector.eps = 1.0
        state, terminated = execute_option(env, selector, state)
        self.assertTrue(terminated)
        state, terminated = execute_option(env, StoreOption(env), state)
        self.assertTrue(terminated)
        self.assertTrue(block.stored)

        ripe = PickupRipeOption(env)
        for _ in range(100):
            if ripe.initiation(state):
                break
            state, _, _, _ = env.step(env.ACTION_IDS["WAIT"])
        else:
            self.fail("PickupRipeOption never became initiable")
        state, terminated = execute_option(env, ripe, state)
        self.assertTrue(terminated)
        state, terminated = execute_option(env, DeliverOption(env), state)
        self.assertTrue(terminated)
        self.assertTrue(block.delivered)


class SelectorContractTests(unittest.TestCase):
    def setUp(self):
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)

    def test_carried_block_is_represented_and_features_are_bounded(self):
        env = SmallRoomsEnv(choose_storage=False, proc_mean=80)
        env.reset()
        block = env.blocks[0]
        block.carrying = True
        block.position = env.current_state
        urgent, less_urgent = env.blocks[1:3]
        urgent.stored = True
        urgent.position = env.storage_positions[0]
        urgent.storage_location = urgent.position
        urgent.storage_steps_needed = 100
        urgent.stored_time_step = 0
        less_urgent.stored = True
        less_urgent.position = env.storage_positions[1]
        less_urgent.storage_location = less_urgent.position
        less_urgent.storage_steps_needed = 20
        less_urgent.stored_time_step = 99
        env.time_steps = 99
        selector = StorageSelectOption(env)

        block_features = selector._block_feats()
        self.assertAlmostEqual(block_features[0], 1.0 / 3.0)
        self.assertAlmostEqual(block_features[9], 1.0 / selector.MAX_T)
        self.assertTrue(np.isfinite(block_features).all())
        self.assertTrue(((block_features >= 0.0) & (block_features <= 1.0)).all())
        mask = np.ones(selector.n_cells, dtype=np.float32)
        zeros = np.zeros(selector.n_cells, dtype=np.float32)
        phi = selector._φ(mask, zeros, zeros, env.current_state)
        self.assertEqual(len(phi), selector.q.net[0].in_features)
        self.assertTrue(np.isfinite(phi).all())
        self.assertTrue(((phi[-2:] >= 0.0) & (phi[-2:] <= 1.0)).all())

    def test_delivery_credit_counts_the_bonus_once(self):
        env = SmallRoomsEnv(choose_storage=False, proc_mean=20)
        env.reset()
        selector = StorageSelectOption(env)
        block = env.blocks[0]
        block.storage_location = env.storage_positions[0]
        block.position = block.storage_location
        block.stored = True
        state_size = selector.q.net[0].in_features
        selector.pending[block.label] = {
            "phi_s": np.zeros(state_size, dtype=np.float32),
            "a": 0,
            "imm": 0.0,
            "acc": 0.0,
            "disc": 1.0,
        }
        delivery_reward = env._delivery_reward(block.storage_location, 0.0)
        environment_reward = -0.09 + delivery_reward

        accumulate_selector_pending(
            selector,
            environment_reward,
            {
                "delivered_block": block.label,
                "delivery_reward": delivery_reward,
            },
        )
        self.assertAlmostEqual(
            selector.pending[block.label]["acc"], environment_reward
        )
        self.assertAlmostEqual(selector.pending[block.label]["disc"], selector.gamma)
        block.delivered = True
        selector.on_delivery(block.label, 0.0, delivery_reward)
        _, _, recorded_return, _, done = selector.buffer[-1]
        self.assertAlmostEqual(recorded_return, environment_reward)
        self.assertEqual(done, 1.0)

    def test_explicit_terminal_return_matches_full_environment_return(self):
        env = SmallRoomsEnv(choose_storage=False, proc_mean=20)
        env.reset()
        selector = StorageSelectOption(
            env,
            return_mode=StorageSelectOption.RETURN_EXPLICIT_TERMINAL,
        )
        block = env.blocks[0]
        block.storage_location = env.storage_positions[0]
        block.position = block.storage_location
        block.stored = True
        state_size = selector.q.net[0].in_features
        selector.pending[block.label] = {
            "phi_s": np.zeros(state_size, dtype=np.float32),
            "a": 0,
            "imm": 0.0,
            "acc": 0.0,
            "disc": 1.0,
            "steps": 0,
        }
        delivery_reward = env._delivery_reward(block.storage_location, 0.0)
        environment_reward = -0.09 + delivery_reward

        accumulate_selector_pending(selector, 2.0, {})
        accumulate_selector_pending(selector, 3.0, {})
        accumulate_selector_pending(
            selector,
            environment_reward,
            {
                "delivered_block": block.label,
                "delivery_reward": delivery_reward,
            },
        )
        expected = (
            2.0
            + selector.gamma * 3.0
            + selector.gamma**2 * environment_reward
        )
        expected_without_terminal = (
            2.0 + selector.gamma * 3.0 + selector.gamma**2 * -0.09
        )
        self.assertAlmostEqual(
            selector.pending[block.label]["acc"], expected_without_terminal
        )
        self.assertAlmostEqual(
            selector.pending[block.label]["terminal_discount"],
            selector.gamma**2,
        )
        block.delivered = True
        selector.on_delivery(block.label, 0.0, delivery_reward)
        _, _, recorded_return, _, done = selector.buffer[-1]
        self.assertAlmostEqual(recorded_return, expected)
        self.assertEqual(done, 1.0)

    def test_selector_event_return_uses_one_discount_per_environment_step(self):
        env = SmallRoomsEnv(choose_storage=False, proc_mean=20)
        env.reset()
        selector = StorageSelectOption(env, gamma=0.5)
        block = env.blocks[0]
        block.storage_location = env.storage_positions[0]
        block.position = block.storage_location
        block.stored = True
        state_size = selector.q.net[0].in_features
        selector.pending[block.label] = {
            "phi_s": np.zeros(state_size, dtype=np.float32),
            "a": 0,
            "imm": 0.0,
            "acc": 0.0,
            "disc": 1.0,
            "steps": 0,
        }
        delivery_reward = env._delivery_reward(block.storage_location, 0.0)
        rewards = [2.0, 3.0, -0.09 + delivery_reward]
        accumulate_selector_pending(selector, rewards[0], {})
        accumulate_selector_pending(selector, rewards[1], {})
        accumulate_selector_pending(
            selector,
            rewards[2],
            {
                "delivered_block": block.label,
                "delivery_reward": delivery_reward,
            },
        )
        block.delivered = True
        selector.on_delivery(block.label, 0.0, delivery_reward)
        _, _, recorded_return, _, done = selector.buffer[-1]
        expected = rewards[0] + 0.5 * rewards[1] + 0.25 * rewards[2]
        self.assertAlmostEqual(recorded_return, expected)
        self.assertEqual(done, 1.0)

    def test_selector_learning_rejects_nonterminal_bootstrap_records(self):
        env = SmallRoomsEnv(choose_storage=False)
        env.reset()
        selector = StorageSelectOption(env, batch_size=1)
        state_size = selector.q.net[0].in_features
        state = np.zeros(state_size, dtype=np.float32)
        selector.buffer.append((state, 0, 1.0, state.copy(), 0.0))

        with self.assertRaisesRegex(RuntimeError, "without bootstrap"):
            selector._learn()

    def test_evaluation_mode_freezes_selector_exploration_and_replay(self):
        env = SmallRoomsEnv(choose_storage=False)
        env.reset()
        block = env.blocks[0]
        block.carrying = True
        block.position = env.current_state
        selector = StorageSelectOption(env)
        selector.eps = 1.0
        selector.set_learning_enabled(False)
        call_count = selector.call_cnt
        buffer_size = len(selector.buffer)

        self.assertTrue(selector.initiation(env.get_current_state()))
        selector.policy(env.get_current_state())
        self.assertEqual(selector.call_cnt, call_count)
        self.assertEqual(len(selector.buffer), buffer_size)
        self.assertEqual(selector.pending, {})
        self.assertEqual(selector.eps, 0.0)

    def test_reset_clears_episode_local_counts_and_selector_state(self):
        env = SmallRoomsEnv(choose_storage=False)
        selector = StorageSelectOption(env)
        env.options.add(selector)
        env.reset()
        cell = env.storage_positions[0]
        env.storage_counts[cell] = 7
        selector.pending["B1"] = {"sentinel": True}
        env.current_episode = 9

        env.reset()
        self.assertEqual(env.storage_counts[cell], 0)
        self.assertEqual(env.current_episode, 9)
        self.assertEqual(selector.pending, {})

    def test_ga_assignment_rejects_duplicate_cells(self):
        env = SmallRoomsEnv(choose_storage=False)
        assignment = list(env.storage_positions[: len(env.blocks)])
        assignment[1] = assignment[0]
        with tempfile.NamedTemporaryFile(suffix=".pkl") as handle:
            pickle.dump(assignment, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "duplicate"):
                GAStorageSelectOption(env, handle.name)

if __name__ == "__main__":
    unittest.main()
