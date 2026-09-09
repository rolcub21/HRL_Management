import unittest
from unittest.mock import patch

from example.Options.ExplicitAcceptOption import ExplicitAcceptOption
from example.small_rooms_env import SmallRoomsEnv


def isolated_inbound_env(
    *,
    grid_rows=8,
    grid_cols=8,
    pickup=(1, 6),
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
        pickup_cells=pickup,
    )
    env.reset(instance=env.sample_episode_instance(9201))
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
    target.position = pickup
    target.storage_location = None
    target.carrying = False
    target.stored = False
    target.delivered = False
    target.stored_time_step = None
    target.storage_steps_elapsed = 0
    target.storage_steps_needed = 100
    if tuple(pickup) != tuple(env.pickup_cell):
        raise AssertionError("test target must be in the inbound pickup cell")
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
    raise AssertionError("ExplicitAcceptOption did not terminate")


class ExplicitAcceptOptionTests(unittest.TestCase):
    def test_bound_placement_assigns_only_at_putdown_and_starts_clock_there(self):
        destination = (5, 5)
        env, target = isolated_inbound_env(destination=destination)
        option = ExplicitAcceptOption(env, target.label, destination)

        self.assertTrue(option.initiation(env.get_current_state()))
        state = env.get_current_state()
        actions = []
        infos = []
        expected_stored_time_step = None
        with (
            patch(
                "PSLAP.dynamic_yard.select_storage_location",
                side_effect=AssertionError("a baseline selector was invoked"),
            ),
            patch(
                "PSLAP.baselines.make_policy",
                side_effect=AssertionError("a baseline registry was invoked"),
            ),
        ):
            for _ in range(200):
                action = option.policy(state)
                if action == env.ACTION_IDS["PUTDOWN"]:
                    self.assertTrue(target.carrying)
                    self.assertEqual(target.storage_location, destination)
                    self.assertIsNone(target.stored_time_step)
                    expected_stored_time_step = env.time_steps + 1
                else:
                    self.assertIsNone(target.storage_location)
                    self.assertIsNone(target.stored_time_step)
                state, _, _, info = env.step(action)
                actions.append(action)
                infos.append(info)
                if action == env.ACTION_IDS["PUTDOWN"]:
                    self.assertEqual(
                        target.stored_time_step, expected_stored_time_step
                    )
                    self.assertEqual(target.stored_time_step, env.time_steps)
                    self.assertEqual(target.storage_steps_elapsed, 1)
                    self.assertEqual(info["stored_block"], target.label)
                if option.termination(state):
                    break
            else:
                self.fail("bound inbound placement did not terminate")

        self.assertTrue(target.stored)
        self.assertFalse(target.carrying)
        self.assertEqual(target.position, destination)
        self.assertEqual(target.storage_location, destination)
        self.assertEqual(target.stored_time_step, expected_stored_time_step)
        self.assertEqual(
            [info["stored_block"] for info in infos if info["stored_block"]],
            [target.label],
        )
        self.assertEqual(option.success_count, 1)
        self.assertEqual(option.failure_count, 0)
        self.assertEqual(option.last_outcome["source_cell"], env.pickup_cell)
        self.assertEqual(option.last_outcome["destination"], destination)
        self.assertIsNone(
            option.last_outcome["stored_time_step_before"]
        )
        self.assertEqual(
            option.last_outcome["stored_time_step_after"],
            expected_stored_time_step,
        )
        self.assertEqual(
            option.last_outcome["expected_stored_time_step"],
            expected_stored_time_step,
        )
        self.assertEqual(option.last_outcome["actual_steps"], len(actions))
        self.assertEqual(option.last_outcome["replans"], 0)

    def test_other_blocks_obstruct_both_canonical_legs(self):
        destination = (2, 5)
        env, target = isolated_inbound_env(
            grid_rows=6,
            grid_cols=7,
            pickup=(2, 3),
            destination=destination,
            agent=(2, 1),
        )
        env.rooms[:] = "#"
        for cell in ((2, 1), (2, 2), (2, 3), (2, 4), (2, 5)):
            env.rooms[cell] = "."
        blocker = env.blocks[1]
        blocker.position = (2, 2)
        blocker.storage_location = (2, 2)
        blocker.stored = True
        blocker.delivered = False
        blocker.stored_time_step = 0

        approach_blocked = ExplicitAcceptOption(
            env, target.label, destination
        )
        self.assertFalse(approach_blocked.initiation(env.get_current_state()))

        blocker.position = (2, 4)
        blocker.storage_location = (2, 4)
        transport_blocked = ExplicitAcceptOption(
            env, target.label, destination
        )
        self.assertFalse(transport_blocked.initiation(env.get_current_state()))

        blocker.position = None
        blocker.storage_location = None
        blocker.stored = False
        blocker.delivered = True
        clear = ExplicitAcceptOption(env, target.label, destination)
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

    def test_duplicate_inbound_is_rejected_and_audited(self):
        env, target = isolated_inbound_env()
        duplicate = env.blocks[1]
        duplicate.position = env.pickup_cell
        duplicate.storage_location = None
        duplicate.carrying = False
        duplicate.stored = False
        duplicate.delivered = False
        duplicate.stored_time_step = None
        option = ExplicitAcceptOption(env, target.label, (5, 5))

        self.assertFalse(option.initiation(env.get_current_state()))
        action = option.policy(env.get_current_state())
        state, _, _, _ = env.step(action)

        self.assertEqual(action, env.ACTION_IDS["WAIT"])
        self.assertTrue(option.termination(state))
        self.assertFalse(option.last_outcome["success"])
        self.assertEqual(
            option.last_outcome["reason"], "inbound_block_not_unique"
        )
        self.assertEqual(option.failure_count, 1)
        self.assertEqual(option.episode_failure_count, 1)

    def test_stale_cached_path_replans_without_changing_destination(self):
        destination = (5, 5)
        env, target = isolated_inbound_env(destination=destination)
        option = ExplicitAcceptOption(env, target.label, destination)
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
        self.assertEqual(target.storage_location, destination)
        self.assertEqual(option.destination, destination)
        self.assertGreaterEqual(option.last_outcome["replans"], 1)
        self.assertGreater(len(actions), 0)

    def test_start_race_terminates_with_an_audited_failure(self):
        destination = (5, 5)
        env, target = isolated_inbound_env(destination=destination)
        option = ExplicitAcceptOption(env, target.label, destination)
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
        self.assertEqual(option.last_outcome["destination"], destination)
        self.assertIsNone(target.storage_location)
        self.assertIsNone(target.stored_time_step)

    def test_binding_expires_after_decision_epoch(self):
        env, target = isolated_inbound_env()
        option = ExplicitAcceptOption(env, target.label, (5, 5))
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
