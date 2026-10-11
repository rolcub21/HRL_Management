import unittest
from unittest.mock import patch

from example.Options.ReconfigureOption import ReconfigureOption
from example.small_rooms_env import SmallRoomsEnv
from PSLAP.dynamic_yard import YardSnapshot
from PSLAP.viability import (
    RecoveryActionKind,
    RecoveryState,
    legal_recovery_actions,
)
from PSLAP.viability_filter import online_fixed_obstacles

from tests.test_direct_deliver_option import queue_arrival_corridor_env


def isolated_reconfiguration_env(
    *,
    grid_rows=8,
    grid_cols=8,
    source=(3, 3),
    destination=(5, 5),
    agent=(1, 1),
):
    env = SmallRoomsEnv(
        choose_storage=False,
        arrival_rate=0.0,
        proc_mean=50,
        number_blocks=4,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        start_state=agent,
        door_cell=(0, grid_cols - 2),
        exit_cells=[(grid_rows - 1, grid_cols - 2)],
    )
    env.reset(instance=env.sample_episode_instance(9101))
    for block in env.blocks:
        block.position = None
        block.storage_location = None
        block.carrying = False
        block.stored = False
        block.delivered = True
        block.stored_time_step = None
        block.storage_steps_elapsed = 0
        block.arrival_step = 10_000

    env.current_state = agent
    env.time_steps = 20
    target = env.blocks[0]
    target.position = source
    target.storage_location = source
    target.carrying = False
    target.stored = True
    target.delivered = False
    target.stored_time_step = 4
    target.storage_steps_elapsed = 16
    target.storage_steps_needed = 100
    if source not in env.storage_positions:
        raise AssertionError("test source must be a storage cell")
    if destination not in env.storage_positions:
        raise AssertionError("test destination must be a storage cell")
    return env, target


def execute_option(env, option, *, limit=200):
    state = env.get_current_state()
    actions = []
    infos = []
    for _ in range(limit):
        action = option.policy(state)
        state, _, _, info = env.step(action)
        actions.append(action)
        infos.append(info)
        if option.termination(state):
            return actions, infos
    raise AssertionError("ReconfigureOption did not terminate")


class ReconfigureOptionTests(unittest.TestCase):
    def test_certified_path_survives_queue_arrival_that_blocks_shortcut(self):
        env, target, arrival = queue_arrival_corridor_env()
        fixed = online_fixed_obstacles(env, reserve_queue_cells=True)
        state = RecoveryState.from_yard_snapshot(
            YardSnapshot.from_env(env),
            env.current_state,
            fixed_obstacles=fixed - {env.current_state},
            pickup_cells=(env.pickup_cell,),
            wait_cells=(env.waiting_cell,),
        )
        witness = next(
            action
            for action in legal_recovery_actions(state)
            if action.kind is RecoveryActionKind.RELOCATION
            and action.block_label == target.label
            and action.destination == (2, 3)
        )
        unconstrained = env.plan_path_heuristic(
            env.current_state, target.position, ignore_block=target
        )
        self.assertEqual(unconstrained[0], env.ACTION_IDS["UP"])
        self.assertEqual(witness.approach_path[1], (3, 2))

        option = ReconfigureOption.from_recovery_action(
            env, witness, fixed_obstacles=state.fixed_obstacles
        )
        emitted, _ = execute_option(env, option)

        self.assertEqual(emitted[0], env.ACTION_IDS["RIGHT"])
        self.assertEqual(arrival.position, env.pickup_cell)
        self.assertEqual(target.position, (2, 3))
        self.assertEqual(target.storage_location, (2, 3))
        self.assertEqual(option.last_outcome["replans"], 0)
        self.assertTrue(option.last_outcome["certified_path_bound"])

    def test_explicit_bound_move_preserves_clock_and_commits_at_putdown(self):
        destination = (5, 5)
        env, target = isolated_reconfiguration_env(destination=destination)
        source = target.storage_location
        stored_time_step = target.stored_time_step
        elapsed = target.storage_steps_elapsed
        option = ReconfigureOption(env, target.label, destination)

        self.assertTrue(option.initiation(env.get_current_state()))
        state = env.get_current_state()
        actions = []
        infos = []
        with patch(
            "PSLAP.dynamic_yard.select_storage_location",
            side_effect=AssertionError("a baseline selector was invoked"),
        ):
            for _ in range(200):
                action = option.policy(state)
                if action == env.ACTION_IDS["PUTDOWN"]:
                    self.assertTrue(target.carrying)
                    self.assertEqual(target.storage_location, destination)
                else:
                    self.assertEqual(target.storage_location, source)
                state, _, _, info = env.step(action)
                actions.append(action)
                infos.append(info)
                if option.termination(state):
                    break
            else:
                self.fail("bound reconfiguration did not terminate")

        self.assertTrue(target.stored)
        self.assertFalse(target.carrying)
        self.assertEqual(target.position, destination)
        self.assertEqual(target.storage_location, destination)
        self.assertEqual(target.stored_time_step, stored_time_step)
        self.assertEqual(target.storage_steps_elapsed, elapsed + len(actions))
        self.assertEqual(
            [info["relocated_block"] for info in infos if info["relocated_block"]],
            [target.label],
        )
        self.assertEqual(option.success_count, 1)
        self.assertEqual(option.failure_count, 0)
        self.assertEqual(option.last_outcome["replans"], 0)
        self.assertEqual(
            option.last_outcome["storage_steps_elapsed_delta"], len(actions)
        )

    def test_other_blocks_obstruct_both_canonical_legs(self):
        env, target = isolated_reconfiguration_env(
            grid_rows=6,
            grid_cols=6,
            source=(2, 3),
            destination=(2, 4),
            agent=(2, 1),
        )
        env.rooms[:] = "#"
        for cell in ((2, 1), (2, 2), (2, 3), (2, 4)):
            env.rooms[cell] = "."
        blocker = env.blocks[1]
        blocker.position = (2, 2)
        blocker.storage_location = (2, 2)
        blocker.stored = True
        blocker.delivered = False
        blocker.stored_time_step = 0

        blocked = ReconfigureOption(env, target.label, (2, 4))
        self.assertFalse(blocked.initiation(env.get_current_state()))

        blocker.position = None
        blocker.storage_location = None
        blocker.stored = False
        blocker.delivered = True
        clear = ReconfigureOption(env, target.label, (2, 4))
        planner = env.plan_path_heuristic
        with patch.object(
            env, "plan_path_heuristic", wraps=planner
        ) as plan_path:
            self.assertTrue(clear.initiation(env.get_current_state()))
        self.assertEqual(plan_path.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["ignore_block"] is target
                for call in plan_path.call_args_list
            )
        )

    def test_stale_cached_path_replans_without_changing_destination(self):
        destination = (5, 5)
        env, target = isolated_reconfiguration_env(destination=destination)
        option = ReconfigureOption(env, target.label, destination)
        state = env.get_current_state()
        self.assertTrue(option.initiation(state))

        first = option.policy(state)
        state, _, _, _ = env.step(first)
        self.assertFalse(option.termination(state))
        stale_action = option._actions[0]
        stale_cell = env._get_intended_cell(env.current_state, stale_action)
        self.assertNotEqual(stale_cell, env.current_state)
        env.rooms[stale_cell] = "#"

        actions, _ = execute_option(env, option)

        self.assertTrue(target.stored)
        self.assertEqual(target.position, destination)
        self.assertGreaterEqual(option.last_outcome["replans"], 1)
        self.assertGreater(len(actions), 0)

    def test_start_race_terminates_with_an_audited_failure(self):
        destination = (5, 5)
        env, target = isolated_reconfiguration_env(destination=destination)
        option = ReconfigureOption(env, target.label, destination)
        self.assertTrue(option.initiation(env.get_current_state()))
        occupant = env.blocks[1]
        occupant.position = destination
        occupant.storage_location = destination
        occupant.stored = True
        occupant.delivered = False
        occupant.stored_time_step = 0

        action = option.policy(env.get_current_state())
        state, _, _, _ = env.step(action)

        self.assertEqual(action, env.ACTION_IDS["WAIT"])
        self.assertTrue(option.termination(state))
        self.assertFalse(option.last_outcome["success"])
        self.assertEqual(option.last_outcome["reason"], "destination_occupied")
        self.assertEqual(option.failure_count, 1)
        self.assertEqual(option.episode_failure_count, 1)

    def test_binding_expires_after_decision_epoch(self):
        env, target = isolated_reconfiguration_env()
        option = ReconfigureOption(env, target.label, (5, 5))
        env.step(env.ACTION_IDS["WAIT"])

        self.assertFalse(option.initiation(env.get_current_state()))
        action = option.policy(env.get_current_state())
        state, _, _, _ = env.step(action)
        self.assertTrue(option.termination(state))
        self.assertEqual(
            option.last_outcome["reason"],
            "bound_action_not_started_at_decision_epoch",
        )


if __name__ == "__main__":
    unittest.main()
