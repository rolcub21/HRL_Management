import unittest
from unittest.mock import patch

from example.Options.DirectDeliverOption import DirectDeliverOption
from example.small_rooms_env import SmallRoomsEnv
from PSLAP.dynamic_yard import YardSnapshot
from PSLAP.viability import (
    RecoveryActionKind,
    RecoveryState,
    legal_recovery_actions,
)


def isolated_delivery_env(
    *,
    grid_rows=8,
    grid_cols=8,
    source=(3, 3),
    exits=((7, 2), (7, 6)),
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
        exit_cells=list(exits),
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
    target.position = source
    target.storage_location = source
    target.carrying = False
    target.stored = True
    target.delivered = False
    target.stored_time_step = 4
    target.storage_steps_elapsed = 16
    target.storage_steps_needed = 30
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
    raise AssertionError("DirectDeliverOption did not terminate")


class DirectDeliverOptionTests(unittest.TestCase):
    def test_delivery_witness_executes_exact_block_and_exit(self):
        env, target = isolated_delivery_env()
        bound_exit = (7, 6)
        recovery_state = RecoveryState.from_yard_snapshot(
            YardSnapshot.from_env(env), env.current_state
        )
        actions = legal_recovery_actions(recovery_state)
        witness = next(
            action
            for action in actions
            if action.kind is RecoveryActionKind.DELIVERY
            and action.block_label == target.label
            and action.destination == bound_exit
        )
        relocation = next(
            action
            for action in actions
            if action.kind is RecoveryActionKind.RELOCATION
        )
        with self.assertRaisesRegex(ValueError, "DELIVERY witness"):
            DirectDeliverOption.from_recovery_action(env, relocation)
        option = DirectDeliverOption.from_recovery_action(env, witness)
        stored_time_step = target.stored_time_step

        self.assertTrue(option.initiation(env.get_current_state()))
        with patch(
            "PSLAP.retrieval_dispatch.plan_retrieval",
            side_effect=AssertionError("a retrieval planner was invoked"),
        ), patch(
            "PSLAP.dynamic_yard.select_storage_location",
            side_effect=AssertionError("a baseline selector was invoked"),
        ):
            emitted, infos = execute_option(env, option)

        self.assertTrue(target.delivered)
        self.assertEqual(target.position, bound_exit)
        self.assertEqual(env.current_state, bound_exit)
        self.assertEqual(target.stored_time_step, stored_time_step)
        self.assertEqual(len(emitted), witness.steps)
        self.assertEqual(option.last_outcome["actual_steps"], witness.steps)
        self.assertEqual(
            option.last_outcome["initial_estimated_steps"], witness.steps
        )
        self.assertEqual(
            option.last_outcome["recovery_witness_steps"], witness.steps
        )
        self.assertEqual(
            [info["delivered_block"] for info in infos if info["delivered_block"]],
            [target.label],
        )
        self.assertEqual(option.success_count, 1)
        self.assertEqual(option.failure_count, 0)

    def test_other_blocks_obstruct_and_only_target_is_ignored(self):
        env, target = isolated_delivery_env(
            grid_rows=6,
            grid_cols=6,
            source=(2, 3),
            exits=((2, 4),),
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

        blocked = DirectDeliverOption(env, target.label, (2, 4))
        self.assertFalse(blocked.initiation(env.get_current_state()))

        blocker.position = None
        blocker.storage_location = None
        blocker.stored = False
        blocker.delivered = True
        clear = DirectDeliverOption(env, target.label, (2, 4))
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

    def test_stale_path_replans_to_same_bound_exit(self):
        env, target = isolated_delivery_env()
        bound_exit = (7, 6)
        option = DirectDeliverOption(env, target.label, bound_exit)
        state = env.get_current_state()
        self.assertTrue(option.initiation(state))

        for _ in range(100):
            action = option.policy(state)
            state, _, _, info = env.step(action)
            self.assertFalse(option.termination(state))
            if info.get("picked_block") == target.label:
                break
        else:
            self.fail("bound block was not picked up")
        self.assertTrue(target.carrying)
        stale_action = option._actions[0]
        stale_cell = env._get_intended_cell(env.current_state, stale_action)
        self.assertNotEqual(stale_cell, env.current_state)
        env.rooms[stale_cell] = "#"

        execute_option(env, option)

        self.assertTrue(target.delivered)
        self.assertEqual(target.position, bound_exit)
        self.assertGreaterEqual(option.last_outcome["replans"], 1)

    def test_start_race_has_audited_failure(self):
        env, target = isolated_delivery_env()
        option = DirectDeliverOption(env, target.label, (7, 6))
        self.assertTrue(option.initiation(env.get_current_state()))
        target.stored = False

        action = option.policy(env.get_current_state())
        state, _, _, _ = env.step(action)

        self.assertEqual(action, env.ACTION_IDS["WAIT"])
        self.assertTrue(option.termination(state))
        self.assertFalse(option.last_outcome["success"])
        self.assertEqual(option.last_outcome["reason"], "bound_block_not_stored")
        self.assertEqual(option.failure_count, 1)

    def test_binding_expires_after_decision_epoch(self):
        env, target = isolated_delivery_env()
        option = DirectDeliverOption(env, target.label, (7, 6))
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
