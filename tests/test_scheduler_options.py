from dataclasses import replace
import unittest

import numpy as np
import torch

from example.Options.RetrieveDeliverOption import (
    FirstLegRetrieveDeliverOption,
    RetrieveDeliverOption,
    StrictRetrieveDeliverOption,
)
from example.Options.StrategicDeferOption import StrategicDeferOption
from example.controller_observation import (
    OnlineManifestTimingObservationEncoder,
    OnlineSignedTimingObservationEncoder,
)
from example.controller_options import build_controller_options
from example.small_rooms_env import SmallRoomsEnv
from gated_agent import (
    GatedSchedulingAgent,
    SCHEDULER_CONTROLLER_ACTION_INTERFACE,
)
from option import BaseOption
from options_agent import canonical_manager_options, controller_ids
from PSLAP.dynamic_yard import YardSnapshot
from PSLAP.neutral_protocol import neutral_relocation_selector
from PSLAP.retrieval_dispatch import plan_retrieval
from PSLAP.retrieval_executor import (
    CompleteLiveRetrievalExecutor,
    RetrievalExecutor,
    StrictStartRetrievalExecutor,
)


class DummySelector(BaseOption):
    controller_identifier = "option:StorageSelectOption"

    def __init__(self, env):
        super().__init__(is_primitive=False)
        self.env = env

    def initiation(self, state):
        return False

    def policy(self, state, test=False):
        return self.env.ACTION_IDS["WAIT"]

    def termination(self, state):
        return True

    def __hash__(self):
        return hash(type(self))


def isolated_retrieval_env(target_index=0):
    env = SmallRoomsEnv(
        choose_storage=False,
        arrival_rate=0.0,
        proc_mean=50,
    )
    env.reset(instance=env.sample_episode_instance(801))
    for block in env.blocks:
        block.position = None
        block.storage_location = None
        block.carrying = False
        block.stored = False
        block.delivered = True
        block.stored_time_step = None
        block.storage_steps_elapsed = 0
        block.arrival_step = 10_000

    env.current_state = (1, 1)
    env.time_steps = 0
    target = env.blocks[target_index]
    target.position = env.storage_positions[len(env.storage_positions) // 2]
    target.storage_location = target.position
    target.stored = True
    target.delivered = False
    target.stored_time_step = 0
    target.storage_steps_elapsed = 0
    target.storage_steps_needed = 100
    return env, target


def execute_option(env, option, limit=400):
    state = env.get_current_state()
    if not option.initiation(state):
        raise AssertionError("option is not initiable")
    actions = []
    rewards = []
    infos = []
    target_pickup_termination = None
    for _ in range(limit):
        action = option.policy(state)
        next_state, reward, done, info = env.step(action)
        terminated = option.termination(next_state)
        actions.append(action)
        rewards.append(float(reward))
        infos.append(info)
        if info.get("picked_block") == option.target_label:
            target_pickup_termination = terminated
        state = next_state
        if terminated:
            return {
                "actions": actions,
                "rewards": rewards,
                "infos": infos,
                "done": done,
                "target_pickup_termination": target_pickup_termination,
            }
    raise AssertionError("option did not terminate")


class AtomicRetrieveDeliverTests(unittest.TestCase):
    def test_named_retrieval_completes_one_target_atomically(self):
        env, target = isolated_retrieval_env()
        plan = plan_retrieval(
            YardSnapshot.from_env(env),
            env.current_state,
            target.label,
            relocation_selector=neutral_relocation_selector,
        )
        option = RetrieveDeliverOption(
            env, 0, relocation_selector=neutral_relocation_selector
        )

        run = execute_option(env, option)

        self.assertTrue(target.delivered)
        self.assertFalse(run["target_pickup_termination"])
        self.assertEqual(len(run["actions"]), plan.estimated_steps)
        self.assertEqual(option.last_outcome["actual_steps"], plan.estimated_steps)
        self.assertEqual(
            [
                info["delivered_block"]
                for info in run["infos"]
                if info.get("delivered_block")
            ],
            [target.label],
        )

    def test_admissibility_is_physical_not_a_deadline_threshold(self):
        for remaining in (50, 0, -10):
            with self.subTest(remaining=remaining):
                env, target = isolated_retrieval_env()
                target.storage_steps_needed = 100
                target.stored_time_step = remaining - 100
                option = RetrieveDeliverOption(
                    env, 0, relocation_selector=neutral_relocation_selector
                )
                self.assertTrue(option.initiation(env.get_current_state()))

    def test_bound_identity_does_not_switch_to_more_urgent_block(self):
        env, target = isolated_retrieval_env(target_index=1)
        other = env.blocks[0]
        other.position = env.storage_positions[0]
        other.storage_location = other.position
        other.stored = True
        other.delivered = False
        other.stored_time_step = 0
        other.storage_steps_needed = 100
        option = RetrieveDeliverOption(
            env, 1, relocation_selector=neutral_relocation_selector
        )
        self.assertTrue(option.initiation(env.get_current_state()))
        other.stored_time_step = -1_000

        run = execute_option(env, option)

        self.assertTrue(target.delivered)
        self.assertTrue(other.stored)
        self.assertFalse(other.delivered)
        self.assertEqual(option.target_label, target.label)
        self.assertNotIn(
            other.label,
            [
                info.get("delivered_block")
                for info in run["infos"]
                if info.get("delivered_block")
            ],
        )

    def test_start_race_fails_cleanly_instead_of_waiting_forever(self):
        env, target = isolated_retrieval_env()
        option = RetrieveDeliverOption(
            env, 0, relocation_selector=neutral_relocation_selector
        )
        self.assertTrue(option.initiation(env.get_current_state()))
        target.stored = False

        action = option.policy(env.get_current_state())
        next_state, _, _, _ = env.step(action)

        self.assertEqual(action, env.ACTION_IDS["WAIT"])
        self.assertTrue(option.termination(next_state))
        self.assertFalse(option.last_outcome["success"])

    def test_stale_path_replans_before_putdown(self):
        env, target = isolated_retrieval_env()
        option = RetrieveDeliverOption(
            env, 0, relocation_selector=neutral_relocation_selector
        )
        state = env.get_current_state()
        self.assertTrue(option.initiation(state))
        infos = []

        for _ in range(200):
            action = option.policy(state)
            state, _, _, info = env.step(action)
            infos.append(info)
            self.assertFalse(option.termination(state))
            if info.get("picked_block") == target.label:
                break
        else:
            self.fail("target was not picked up")

        cached_action = option.executor._actions[0]
        blocked_cell = env._get_intended_cell(env.current_state, cached_action)
        self.assertNotEqual(blocked_cell, env.current_state)
        env.rooms[blocked_cell] = "#"

        for _ in range(200):
            action = option.policy(state)
            state, _, _, info = env.step(action)
            infos.append(info)
            if option.termination(state):
                break
        else:
            self.fail("replanned retrieval did not terminate")

        self.assertTrue(target.delivered)
        self.assertGreaterEqual(option.last_outcome["replans"], 1)
        self.assertFalse(any(info.get("illegal_drop") for info in infos))

    def test_forced_obstruction_relocates_and_preserves_clock(self):
        env, target, blockers = self._forced_obstruction_env()
        initial_clocks = {
            block.label: block.stored_time_step for block in blockers
        }
        plan = plan_retrieval(
            YardSnapshot.from_env(env),
            env.current_state,
            target.label,
            relocation_selector=neutral_relocation_selector,
        )
        option = RetrieveDeliverOption(
            env, 0, relocation_selector=neutral_relocation_selector
        )

        run = execute_option(env, option)

        relocated = [
            info["relocated_block"]
            for info in run["infos"]
            if info.get("relocated_block")
        ]
        self.assertEqual(
            relocated,
            [relocation.block_label for relocation in plan.relocations],
        )
        self.assertEqual(len(run["actions"]), plan.estimated_steps)
        self.assertTrue(target.delivered)
        self.assertFalse(any(info.get("illegal_drop") for info in run["infos"]))
        for label in relocated:
            block = next(item for item in blockers if item.label == label)
            self.assertEqual(block.stored_time_step, initial_clocks[label])

    def test_strict_start_rejects_planner_live_obstacle_mismatch(self):
        env, target, _ = self._forced_obstruction_env()
        env.current_state = env.waiting_cell

        # The detached yard excludes both queue blocks.  The pickup block is
        # nevertheless a live obstacle on the waiting cell's only route into
        # the yard, so the canonical first relocation cannot be executed.
        inbound = env.blocks[5]
        inbound.position = env.pickup_cell
        inbound.stored = False
        inbound.delivered = False
        waiting = env.blocks[6]
        waiting.position = env.waiting_cell
        waiting.stored = False
        waiting.delivered = False

        legacy = RetrievalExecutor(
            env,
            target.label,
            relocation_selector=neutral_relocation_selector,
        )
        canonical_plan = legacy._plan()
        self.assertIsNotNone(canonical_plan)
        self.assertTrue(canonical_plan.relocations)

        # v1 remains reproducible as an ablation: canonical-plan admission is
        # followed by the historical immediate relocation_route_failed result.
        self.assertTrue(legacy.can_start())
        self.assertTrue(legacy.start())
        self.assertEqual(legacy.next_action(), env.ACTION_IDS["WAIT"])
        self.assertEqual(legacy.failure_reason, "relocation_route_failed")

        strict = StrictStartRetrievalExecutor(
            env,
            target.label,
            relocation_selector=neutral_relocation_selector,
        )
        self.assertFalse(strict.can_start())
        self.assertFalse(strict.start())
        self.assertEqual(strict.failure_reason, "first_live_leg_unavailable")

        legacy_option = RetrieveDeliverOption(
            env, 0, relocation_selector=neutral_relocation_selector
        )
        strict_option = StrictRetrieveDeliverOption(
            env, 0, relocation_selector=neutral_relocation_selector
        )
        historical_option = FirstLegRetrieveDeliverOption(
            env, 0, relocation_selector=neutral_relocation_selector
        )
        state = env.get_current_state()
        self.assertEqual(strict.VERSION, "named_atomic_retrieval_executor_v2")
        self.assertEqual(
            historical_option.VERSION, "retrieve_deliver_option_v2"
        )
        self.assertEqual(strict_option.VERSION, "retrieve_deliver_option_v3")
        self.assertNotEqual(legacy_option, strict_option)
        self.assertTrue(legacy_option.initiation(state))
        self.assertFalse(strict_option.initiation(state))

    def test_v3_rejects_later_live_leg_while_v2_accepts_first_leg(self):
        env, target, _ = self._forced_obstruction_env()
        canonical = RetrievalExecutor(
            env,
            target.label,
            relocation_selector=neutral_relocation_selector,
        )._plan()
        self.assertIsNotNone(canonical)
        self.assertTrue(canonical.relocations)

        # Find a currently traversable, non-storage cell that leaves the first
        # canonical relocation executable but blocks a later canonical leg.
        # This searches only the tiny deterministic fixture and then freezes
        # the discovered cell as a visible inbound obstacle.
        inbound = env.blocks[5]
        selected = None
        for position in sorted(
            set(
                (row, col)
                for row in range(env.grid_rows)
                for col in range(env.grid_cols)
                if env.rooms[row, col] != "#"
            )
            - set(env.storage_positions)
        ):
            inbound.position = position
            inbound.delivered = False
            v2 = StrictStartRetrievalExecutor(
                env,
                target.label,
                relocation_selector=neutral_relocation_selector,
            )
            v3 = CompleteLiveRetrievalExecutor(
                env,
                target.label,
                relocation_selector=neutral_relocation_selector,
            )
            if v2.can_start() and not v3.can_start():
                selected = position
                break
        self.assertIsNotNone(selected)

        v2 = StrictStartRetrievalExecutor(
            env,
            target.label,
            relocation_selector=neutral_relocation_selector,
        )
        v3 = CompleteLiveRetrievalExecutor(
            env,
            target.label,
            relocation_selector=neutral_relocation_selector,
        )
        self.assertTrue(v2.can_start())
        self.assertFalse(v3.can_start())
        self.assertFalse(v3.start())
        self.assertEqual(v3.failure_reason, "complete_live_plan_unavailable")
        self.assertEqual(
            v3.VERSION, "named_atomic_retrieval_executor_v3"
        )

    def test_strict_start_preserves_executable_relocation(self):
        env, target, _ = self._forced_obstruction_env()
        strict = StrictStartRetrievalExecutor(
            env,
            target.label,
            relocation_selector=neutral_relocation_selector,
        )
        option = StrictRetrieveDeliverOption(
            env, 0, relocation_selector=neutral_relocation_selector
        )

        self.assertTrue(strict.can_start())
        run = execute_option(env, option)

        self.assertTrue(target.delivered)
        self.assertTrue(option.last_outcome["success"])
        self.assertEqual(option.last_outcome["replans"], 0)
        self.assertFalse(any(info.get("illegal_drop") for info in run["infos"]))

    @staticmethod
    def _forced_obstruction_env():
        env = SmallRoomsEnv(
            choose_storage=False, arrival_rate=0.0, proc_mean=50
        )
        env.reset(instance=env.sample_episode_instance(803))
        env.time_steps = 20
        env.current_state = (1, 1)
        for block in env.blocks:
            block.position = None
            block.storage_location = None
            block.carrying = False
            block.stored = False
            block.delivered = True
            block.storage_steps_elapsed = 0
            block.storage_steps_needed = 100
            block.stored_time_step = None
            block.arrival_step = 10_000
        target = env.blocks[0]
        target.position = (4, 4)
        target.storage_location = target.position
        target.stored = True
        target.delivered = False
        target.storage_steps_needed = 1
        target.storage_steps_elapsed = 1
        target.stored_time_step = 19
        blocker_positions = ((3, 4), (5, 4), (4, 3), (4, 5))
        for block, position in zip(env.blocks[1:5], blocker_positions):
            block.position = position
            block.storage_location = position
            block.stored = True
            block.delivered = False
            block.storage_steps_needed = 100
            block.storage_steps_elapsed = 10
            block.stored_time_step = 10
        return env, target, env.blocks[1:5]


class StrategicDeferTests(unittest.TestCase):
    def test_stops_at_first_observed_arrival(self):
        env = SmallRoomsEnv(choose_storage=False, arrival_rate=0.5, proc_mean=50)
        base = env.sample_episode_instance(804)
        arrivals = (0, 2) + tuple(range(100, 138))
        env.reset(instance=replace(base, arrival_steps=arrivals))
        option = StrategicDeferOption(env, max_defer_steps=10)

        terminations = []
        for _ in range(2):
            action = option.policy(env.get_current_state())
            state, _, _, _ = env.step(action)
            terminations.append(option.termination(state))

        self.assertEqual(terminations, [False, True])
        self.assertEqual(
            option.last_outcome, {"reason": "observed_event", "steps": 2}
        )

    def test_stops_when_complete_retrieval_slack_reaches_zero(self):
        env, target = isolated_retrieval_env()
        plan = plan_retrieval(
            YardSnapshot.from_env(env),
            env.current_state,
            target.label,
            relocation_selector=neutral_relocation_selector,
        )
        target.storage_steps_needed = plan.estimated_steps + 2
        option = StrategicDeferOption(
            env,
            max_defer_steps=10,
            relocation_selector=neutral_relocation_selector,
        )

        terminations = []
        for _ in range(2):
            state, _, _, _ = env.step(option.policy(env.get_current_state()))
            terminations.append(option.termination(state))

        self.assertEqual(terminations, [False, True])
        self.assertEqual(option.last_outcome["reason"], "retrieval_became_due")

    def test_due_job_forces_single_step_and_cap_is_bounded(self):
        env, target = isolated_retrieval_env()
        plan = plan_retrieval(
            YardSnapshot.from_env(env), env.current_state, target.label
        )
        target.storage_steps_needed = plan.estimated_steps
        due = StrategicDeferOption(env, max_defer_steps=10)
        state, _, _, _ = env.step(due.policy(env.get_current_state()))
        self.assertTrue(due.termination(state))
        self.assertEqual(due.last_outcome["reason"], "already_due_single_step")

        capped = SmallRoomsEnv(
            choose_storage=False, arrival_rate=0.5, proc_mean=50
        )
        base = capped.sample_episode_instance(805)
        capped.reset(
            instance=replace(
                base,
                arrival_steps=(0,) + tuple(range(100, 139)),
            )
        )
        cap = StrategicDeferOption(capped, max_defer_steps=3)
        terminations = []
        for _ in range(3):
            state, _, _, _ = capped.step(cap.policy(capped.get_current_state()))
            terminations.append(cap.termination(state))
        self.assertEqual(terminations, [False, False, True])
        self.assertEqual(cap.last_outcome, {"reason": "defer_cap", "steps": 3})

    def test_terminal_reason_is_explicit(self):
        env = SmallRoomsEnv(choose_storage=False, arrival_rate=0.5, proc_mean=50)
        env.reset(instance=env.sample_episode_instance(806))
        option = StrategicDeferOption(env, max_defer_steps=10)
        option.policy(env.get_current_state())
        for block in env.blocks:
            block.delivered = True
        self.assertTrue(option.termination(env.get_current_state()))
        self.assertEqual(option.last_outcome["reason"], "environment_terminal")


class SchedulerInterfaceTests(unittest.TestCase):
    def test_factory_builds_exact_stable_44_option_interface(self):
        env = SmallRoomsEnv(choose_storage=False, arrival_rate=0.5, proc_mean=50)
        selector = DummySelector(env)
        build_controller_options(
            env,
            selector,
            controller_action_interface=SCHEDULER_CONTROLLER_ACTION_INTERFACE,
            max_defer_steps=7,
        )
        manager = canonical_manager_options(env.options)
        ids = controller_ids(manager)
        retrieve_ids = [
            item for item in ids if item.startswith("option:RetrieveDeliver:")
        ]
        self.assertEqual(len(manager), 44)
        self.assertEqual(len(retrieve_ids), 40)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("option:DeferUntilEvent:v1", ids)
        self.assertNotIn("option:PickupRipeOption", ids)
        self.assertNotIn("option:DeliverOption", ids)

    def test_manifest_observation_rows_match_retrieval_slots(self):
        env = SmallRoomsEnv(choose_storage=False, arrival_rate=0.5, proc_mean=50)
        env.reset(instance=env.sample_episode_instance(807))
        for block in env.blocks:
            block.position = None
            block.stored = False
            block.delivered = False
        env.blocks[1].position = env.pickup_cell

        compact = OnlineSignedTimingObservationEncoder(env).capture()
        manifest = OnlineManifestTimingObservationEncoder(env).capture()

        self.assertTrue(compact.block_mask[0])
        self.assertFalse(compact.block_mask[1])
        self.assertFalse(manifest.block_mask[0])
        self.assertTrue(manifest.block_mask[1])
        self.assertEqual(
            OnlineManifestTimingObservationEncoder(env).metadata()[
                "controller_observation_block_row_contract"
            ],
            "episode_manifest_slot_v1",
        )

    def test_atomic_retrieval_creates_one_manager_record_and_no_worker_record(self):
        torch.manual_seed(0)
        env, target = isolated_retrieval_env()
        selector = DummySelector(env)
        build_controller_options(
            env,
            selector,
            controller_action_interface=SCHEDULER_CONTROLLER_ACTION_INTERFACE,
        )
        encoder = OnlineManifestTimingObservationEncoder(env)
        agent = GatedSchedulingAgent(
            env=env,
            state_size=encoder.feature_size,
            action_size=len(
                [option for option in env.options if option.is_primitive]
            ),
            n_episodes=1,
            n_steps=400,
            batch_size=2,
            buffer_size=100,
            reward_clip=1_000_000.0,
            gamma=0.99,
            training_policy="mode_regularized",
            state_encoder=encoder,
            controller_observation_metadata=encoder.metadata(),
            disable_tensorboard=True,
            device="cpu",
            verbose=False,
            max_defer_steps=10,
        )
        option = next(
            item
            for item in agent.manager_options
            if isinstance(item, RetrieveDeliverOption)
            and item.target_label == target.label
        )
        state = env.get_current_state()
        agent.current_option = option
        agent.option_start_state = state
        agent.option_start_features = encoder(state).copy()
        agent.option_reward_traj = []
        rewards = []
        delivery_events = 0

        for _ in range(400):
            state_features = encoder(state).copy()
            action = option.policy(state)
            next_state, reward, done, info = env.step(action)
            next_features = encoder(next_state).copy()
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
            delivery_events += int(bool(info.get("delivered_block")))
            state = next_state
            if terminated:
                break
        else:
            self.fail("atomic option did not terminate")

        self.assertEqual(len(agent.WorkerBuffer), 0)
        self.assertEqual(len(agent.ManagerBuffer), 1)
        experience = agent.ManagerBuffer.memory[0]
        expected = sum(
            (agent.gamma**index) * reward
            for index, reward in enumerate(rewards)
        )
        self.assertEqual(experience.k, len(rewards))
        self.assertAlmostEqual(float(experience.reward), expected)
        self.assertEqual(float(experience.done), 1.0)
        self.assertEqual(delivery_events, 1)
        agent.writer.close()


if __name__ == "__main__":
    unittest.main()
