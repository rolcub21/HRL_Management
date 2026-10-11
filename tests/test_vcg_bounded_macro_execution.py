from dataclasses import replace
import unittest
from unittest.mock import patch

from example.small_rooms_env import SmallRoomsEnv
from PSLAP.viability_candidates import (
    ViabilityActionType,
    enumerate_viability_candidates,
)
from PSLAP.viability_filter import ViabilitySearchConfig
from vcg_bounded_macro_execution import (
    BoundedMacroRealization,
    declared_clear_adjacent_stops,
    execute_bounded_macro_realization,
)


def _live_reconfigure_fixture():
    env = SmallRoomsEnv(
        grid_rows=5,
        grid_cols=5,
        number_blocks=2,
        choose_storage=False,
        arrival_rate=0.05,
        proc_mean=30,
    )
    env.reset(instance=env.sample_episode_instance(93_101))
    target, pending = env.blocks
    target.position = (2, 2)
    target.storage_location = (2, 2)
    target.carrying = False
    target.stored = True
    target.delivered = False
    target.stored_time_step = 1
    target.storage_steps_elapsed = 9
    pending.position = None
    pending.storage_location = None
    pending.carrying = False
    pending.stored = False
    pending.delivered = False
    pending.arrival_step = 10_000
    env.current_state = (1, 1)
    env.time_steps = 10
    snapshot = enumerate_viability_candidates(
        env,
        consecutive_defer_decisions=0,
        search_config=ViabilitySearchConfig(max_nodes=20_000),
    )
    candidates = [
        candidate
        for candidate in snapshot.candidates
        if candidate.action_type is ViabilityActionType.RECONFIGURE
        and declared_clear_adjacent_stops(candidate)
    ]
    if not candidates:
        raise AssertionError(
            "fixture has no SAFE reconfiguration with an adjacent stop"
        )
    candidate = candidates[0]
    return env, candidate, declared_clear_adjacent_stops(candidate)[0]


def _live_final_delivery_fixture():
    env = SmallRoomsEnv(
        grid_rows=5,
        grid_cols=5,
        number_blocks=1,
        choose_storage=False,
        arrival_rate=0.0,
        proc_mean=30,
    )
    env.reset(instance=env.sample_episode_instance(93_102))
    block = env.blocks[0]
    block.position = (2, 2)
    block.storage_location = (2, 2)
    block.carrying = False
    block.stored = True
    block.delivered = False
    block.stored_time_step = 1
    block.storage_steps_elapsed = 9
    env.current_state = (1, 1)
    env.time_steps = 10
    snapshot = enumerate_viability_candidates(
        env,
        consecutive_defer_decisions=0,
        search_config=ViabilitySearchConfig(max_nodes=20_000),
    )
    candidate = next(
        item
        for item in snapshot.candidates
        if item.action_type is ViabilityActionType.DELIVER
    )
    return env, candidate


def _live_accept_candidate():
    env = SmallRoomsEnv(
        grid_rows=5,
        grid_cols=5,
        number_blocks=2,
        choose_storage=False,
        arrival_rate=0.05,
        proc_mean=30,
    )
    env.reset(instance=env.sample_episode_instance(93_103))
    env.blocks[1].arrival_step = 10_000
    snapshot = enumerate_viability_candidates(
        env,
        consecutive_defer_decisions=0,
        search_config=ViabilitySearchConfig(max_nodes=20_000),
    )
    return env, next(
        item
        for item in snapshot.candidates
        if item.action_type is ViabilityActionType.ACCEPT
    )


class LiveBoundedMacroExecutionTests(unittest.TestCase):
    def test_real_wait_and_cardinal_stop_advance_live_time_and_position(self):
        env, candidate, stop = _live_reconfigure_fixture()
        started = env.time_steps

        result = execute_bounded_macro_realization(
            env,
            candidate,
            realization=BoundedMacroRealization(1, stop),
            gamma=0.99,
            remaining_steps=100,
            evaluation=True,
        )

        self.assertTrue(result.realization_complete)
        self.assertIsNone(result.failure_reason)
        self.assertEqual(result.base_replan_count, 0)
        self.assertEqual(result.injected_primitive_steps, 2)
        self.assertEqual(
            result.total_primitive_steps,
            result.base_primitive_steps + result.injected_primitive_steps,
        )
        self.assertEqual(env.time_steps, started + result.total_primitive_steps)
        self.assertEqual(result.base_endpoint, candidate.successor_state.agent_position)
        self.assertEqual(result.realized_endpoint, stop)
        self.assertEqual(env.current_state, stop)
        self.assertEqual(
            [item.phase for item in result.injected_transitions],
            ["delay", "adjacent_stop"],
        )
        wait, move = result.injected_transitions
        self.assertEqual(wait.action_name, "WAIT")
        self.assertEqual(wait.before_agent_position, wait.after_agent_position)
        self.assertEqual(move.before_agent_position, result.base_endpoint)
        self.assertEqual(move.after_agent_position, stop)
        self.assertEqual(
            result.realized_end_time_step - result.base_end_time_step,
            2,
        )

    def test_real_wait_advances_stored_block_clock(self):
        nominal_env, nominal_candidate, _ = _live_reconfigure_fixture()
        delayed_env, delayed_candidate, _ = _live_reconfigure_fixture()

        nominal = execute_bounded_macro_realization(
            nominal_env,
            nominal_candidate,
            realization=BoundedMacroRealization(0),
            gamma=0.99,
            remaining_steps=100,
            evaluation=True,
        )
        delayed = execute_bounded_macro_realization(
            delayed_env,
            delayed_candidate,
            realization=BoundedMacroRealization(1),
            gamma=0.99,
            remaining_steps=100,
            evaluation=True,
        )

        self.assertTrue(nominal.realization_complete)
        self.assertTrue(delayed.realization_complete)
        self.assertEqual(delayed.base_primitive_steps, nominal.base_primitive_steps)
        self.assertEqual(
            delayed.total_primitive_steps,
            nominal.total_primitive_steps + 1,
        )
        self.assertEqual(delayed.realized_endpoint, nominal.realized_endpoint)
        self.assertEqual(
            delayed_env.blocks[0].storage_steps_elapsed,
            nominal_env.blocks[0].storage_steps_elapsed + 1,
        )

    def test_real_wait_can_expose_a_live_arrival(self):
        reference_env, reference_candidate, _ = _live_reconfigure_fixture()
        reference = execute_bounded_macro_realization(
            reference_env,
            reference_candidate,
            realization=BoundedMacroRealization(0),
            gamma=0.99,
            remaining_steps=100,
            evaluation=True,
        )
        env, candidate, _ = _live_reconfigure_fixture()
        pending = env.blocks[1]
        pending.arrival_step = env.time_steps + reference.base_primitive_steps + 1

        delayed = execute_bounded_macro_realization(
            env,
            candidate,
            realization=BoundedMacroRealization(1),
            gamma=0.99,
            remaining_steps=100,
            evaluation=True,
        )

        self.assertTrue(delayed.realization_complete)
        self.assertEqual(delayed.base_primitive_steps, reference.base_primitive_steps)
        self.assertEqual(delayed.base_end_time_step, pending.arrival_step - 1)
        self.assertEqual(delayed.realized_end_time_step, pending.arrival_step)
        self.assertEqual(pending.position, env.pickup_cell)

    def test_reserved_successor_cell_is_not_a_declared_stop(self):
        _, candidate, _ = _live_reconfigure_fixture()
        reservable = next(
            cell
            for cell in declared_clear_adjacent_stops(candidate)
            if cell in candidate.successor_state.storage_cells
        )
        state = replace(
            candidate.successor_state,
            reserved_cells=(
                candidate.successor_state.reserved_cells | {reservable}
            ),
        )
        with_reservation = replace(candidate, successor_state=state)

        self.assertNotIn(
            reservable,
            declared_clear_adjacent_stops(with_reservation),
        )

    def test_requested_disturbance_after_final_delivery_is_suppressed(self):
        env, candidate = _live_final_delivery_fixture()

        result = execute_bounded_macro_realization(
            env,
            candidate,
            realization=BoundedMacroRealization(1),
            gamma=0.99,
            remaining_steps=100,
            evaluation=True,
        )

        self.assertTrue(result.base_execution.option_success)
        self.assertTrue(result.base_execution.env_terminal)
        self.assertFalse(result.realization_complete)
        self.assertEqual(
            result.failure_reason,
            "environment_terminal_before_disturbance_injection",
        )
        self.assertEqual(result.injected_primitive_steps, 0)
        self.assertEqual(result.total_primitive_steps, result.base_primitive_steps)
        self.assertEqual(result.realized_end_time_step, result.base_end_time_step)

    def test_nominal_final_delivery_completes_without_post_terminal_step(self):
        env, candidate = _live_final_delivery_fixture()

        result = execute_bounded_macro_realization(
            env,
            candidate,
            realization=BoundedMacroRealization(0),
            gamma=0.99,
            remaining_steps=100,
            evaluation=True,
        )

        self.assertTrue(result.base_execution.option_success)
        self.assertTrue(result.base_execution.env_terminal)
        self.assertTrue(result.realization_complete)
        self.assertIsNone(result.failure_reason)
        self.assertEqual(result.injected_primitive_steps, 0)
        self.assertEqual(result.total_primitive_steps, result.base_primitive_steps)
        self.assertEqual(result.realized_end_time_step, result.base_end_time_step)

    def test_non_recovery_candidate_is_rejected_before_environment_mutation(self):
        env, candidate = _live_accept_candidate()
        start_position = env.current_state
        start_time = env.time_steps

        with self.assertRaisesRegex(ValueError, "strict recovery"):
            execute_bounded_macro_realization(
                env,
                candidate,
                realization=BoundedMacroRealization(0),
                gamma=0.99,
                remaining_steps=100,
                evaluation=True,
            )

        self.assertEqual(env.current_state, start_position)
        self.assertEqual(env.time_steps, start_time)

    def test_successful_live_replan_is_rejected_before_injection(self):
        env, candidate, _ = _live_reconfigure_fixture()
        original = candidate.option._action_is_valid
        calls = 0

        def force_one_replan(action):
            nonlocal calls
            calls += 1
            if calls == 2:
                return False
            return original(action)

        with patch.object(
            candidate.option,
            "_action_is_valid",
            side_effect=force_one_replan,
        ):
            result = execute_bounded_macro_realization(
                env,
                candidate,
                realization=BoundedMacroRealization(1),
                gamma=0.99,
                remaining_steps=100,
                evaluation=True,
            )

        self.assertTrue(result.base_execution.option_success)
        self.assertEqual(result.base_replan_count, 1)
        self.assertFalse(result.realization_complete)
        self.assertEqual(
            result.failure_reason,
            "base_macro_replanned_outside_fixed_path_contract",
        )
        self.assertEqual(result.injected_primitive_steps, 0)

    def test_undeclared_stop_is_rejected_before_environment_mutation(self):
        env, candidate, _ = _live_reconfigure_fixture()
        start_position = env.current_state
        start_time = env.time_steps
        undeclared = (start_position[0] + 2, start_position[1])

        with self.assertRaisesRegex(ValueError, "declared clear stop set"):
            execute_bounded_macro_realization(
                env,
                candidate,
                realization=BoundedMacroRealization(0, undeclared),
                gamma=0.99,
                remaining_steps=100,
                evaluation=True,
            )

        self.assertEqual(env.current_state, start_position)
        self.assertEqual(env.time_steps, start_time)


if __name__ == "__main__":
    unittest.main()
