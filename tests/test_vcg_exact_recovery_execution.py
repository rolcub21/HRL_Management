from dataclasses import replace
import unittest
from unittest.mock import patch

from example.small_rooms_env import SmallRoomsEnv
from PSLAP.viability_candidates import (
    ViabilityActionType,
    enumerate_viability_candidates,
)
from PSLAP.viability_filter import ViabilitySearchConfig
from vcg_exact_recovery_execution import (
    execute_exact_bounded_recovery_realization,
    execute_exact_recovery_macro,
)
from vcg_bounded_macro_execution import (
    BoundedMacroRealization,
    declared_clear_adjacent_stops,
)


def _stored_env(*, number_blocks, seed):
    env = SmallRoomsEnv(
        grid_rows=5,
        grid_cols=5,
        number_blocks=number_blocks,
        choose_storage=False,
        arrival_rate=0.05,
        proc_mean=30,
    )
    env.reset(instance=env.sample_episode_instance(seed))
    target = env.blocks[0]
    target.position = (2, 2)
    target.storage_location = (2, 2)
    target.carrying = False
    target.stored = True
    target.delivered = False
    target.stored_time_step = 1
    target.storage_steps_elapsed = 9
    for pending in env.blocks[1:]:
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
    return env, target, snapshot


def _path_from_actions(env, start, actions):
    cells = [tuple(start)]
    current = tuple(start)
    for action in actions:
        current = tuple(env._get_intended_cell(current, action))
        cells.append(current)
    return tuple(cells)


class ExactRecoveryExecutionIntegrationTests(unittest.TestCase):
    def test_exact_base_and_fixed_live_realization_share_one_step_account(self):
        env, _, snapshot = _stored_env(number_blocks=2, seed=94_200)
        candidate = next(
            item
            for item in snapshot.candidates
            if item.action_type is ViabilityActionType.RECONFIGURE
            and declared_clear_adjacent_stops(item)
        )
        stop = declared_clear_adjacent_stops(candidate)[0]
        realization = BoundedMacroRealization(1, stop)

        result = execute_exact_bounded_recovery_realization(
            env,
            candidate,
            realization=realization,
            gamma=0.99,
            remaining_steps=100,
            evaluation=True,
        )

        self.assertTrue(result.realization_complete)
        self.assertTrue(result.exact_base.trace_matches_certified_action)
        self.assertEqual(result.base_primitive_steps, candidate.recovery_action.steps)
        self.assertEqual(result.injected_primitive_steps, 2)
        self.assertEqual(
            result.total_primitive_steps,
            candidate.recovery_action.steps + 2,
        )
        self.assertEqual(result.realized_endpoint, stop)

    def test_canonical_path_avoids_pickup_while_live_astar_would_cross_it(self):
        """Regression for equal-duration, different-interior live execution."""

        env, target, _ = _stored_env(number_blocks=2, seed=94_204)
        target.position = (1, 1)
        target.storage_location = (1, 1)
        env.current_state = (1, 2)
        snapshot = enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            search_config=ViabilitySearchConfig(max_nodes=20_000),
        )
        candidate = next(
            item
            for item in snapshot.candidates
            if item.action_type is ViabilityActionType.DELIVER
            and item.recovery_action.destination == (4, 3)
        )
        action = candidate.recovery_action
        planned_actions = env.plan_path_heuristic(
            action.source,
            action.destination,
            ignore_block=target,
        )
        planned_transport = _path_from_actions(
            env,
            action.source,
            planned_actions,
        )

        self.assertEqual(len(planned_transport), len(action.transport_path))
        self.assertNotEqual(planned_transport, action.transport_path)
        self.assertIn(tuple(env.pickup_cell), planned_transport)
        self.assertNotIn(tuple(env.pickup_cell), action.transport_path)

        with patch.object(
            env,
            "plan_path_heuristic",
            side_effect=AssertionError("exact executor must not call A*"),
        ):
            result = execute_exact_recovery_macro(
                env,
                candidate,
                gamma=0.99,
                remaining_steps=100,
                evaluation=True,
            )

        # The trace contains the approach endpoint and then duplicates that
        # source once for stationary PICKUP; the latter begins transport.
        transport_start = len(action.approach_path)
        emitted_transport = result.observed_cell_trace[
            transport_start : transport_start + len(action.transport_path)
        ]
        self.assertTrue(result.macro_execution.option_success)
        self.assertTrue(result.trace_matches_certified_action)
        self.assertEqual(emitted_transport, action.transport_path)
        self.assertNotEqual(emitted_transport, planned_transport)

    def test_equal_length_alternate_interior_path_is_emitted_literally(self):
        env, target, snapshot = _stored_env(number_blocks=2, seed=94_201)
        candidate = next(
            item
            for item in snapshot.candidates
            if item.action_type is ViabilityActionType.RECONFIGURE
        )
        action = candidate.recovery_action
        alternatives = (
            ((1, 1), (1, 2), (2, 2)),
            ((1, 1), (2, 1), (2, 2)),
        )
        planned_actions = env.plan_path_heuristic(
            env.current_state,
            action.source,
            ignore_block=target,
        )
        planned_path = _path_from_actions(
            env, env.current_state, planned_actions
        )
        alternate = next(path for path in alternatives if path != planned_path)
        self.assertEqual(len(alternate), len(planned_path))
        self.assertNotEqual(alternate[1], planned_path[1])
        alternate_action = replace(action, approach_path=alternate)
        alternate_candidate = replace(
            candidate,
            recovery_action=alternate_action,
        )

        with patch.object(
            env,
            "plan_path_heuristic",
            side_effect=AssertionError("exact executor must not call a planner"),
        ):
            result = execute_exact_recovery_macro(
                env,
                alternate_candidate,
                gamma=0.99,
                remaining_steps=100,
                evaluation=True,
            )

        self.assertTrue(result.macro_execution.option_success)
        self.assertTrue(result.trace_matches_certified_action)
        self.assertEqual(result.planner_calls, 0)
        self.assertEqual(result.replan_count, 0)
        self.assertEqual(
            result.observed_cell_trace[: len(alternate)],
            alternate,
        )
        self.assertNotEqual(
            result.observed_cell_trace[1],
            planned_path[1],
        )
        self.assertEqual(
            result.macro_execution.duration,
            alternate_action.steps,
        )
        self.assertEqual(result.macro_execution.relocations, 1)
        self.assertEqual(result.macro_execution.delivery_deviations, ())
        self.assertEqual(result.macro_execution.illegal_drops, 0)
        relocation_infos = [
            transition.info.get("relocated_block")
            for transition in result.emitted_transitions
            if transition.info.get("relocated_block")
        ]
        self.assertEqual(relocation_infos, [target.label])

    def test_exact_delivery_preserves_deviation_formula_and_counts(self):
        env, target, snapshot = _stored_env(number_blocks=1, seed=94_202)
        candidate = next(
            item
            for item in snapshot.candidates
            if item.action_type is ViabilityActionType.DELIVER
        )
        stored_step = target.stored_time_step
        needed = target.storage_steps_needed

        result = execute_exact_recovery_macro(
            env,
            candidate,
            gamma=0.99,
            remaining_steps=100,
            evaluation=True,
        )

        expected_deviation = target.delivered_time_step - (stored_step + needed)
        self.assertTrue(result.macro_execution.option_success)
        self.assertTrue(result.trace_matches_certified_action)
        self.assertTrue(target.delivered)
        self.assertEqual(
            result.macro_execution.delivery_deviations,
            (float(expected_deviation),),
        )
        self.assertEqual(result.macro_execution.relocations, 0)
        self.assertEqual(result.macro_execution.illegal_drops, 0)
        delivered_infos = [
            transition.info.get("delivered_block")
            for transition in result.emitted_transitions
            if transition.info.get("delivered_block")
        ]
        self.assertEqual(delivered_infos, [target.label])
        self.assertEqual(
            result.expected_action_names,
            result.emitted_action_names,
        )
        self.assertEqual(
            result.expected_cell_trace,
            result.observed_cell_trace,
        )

    def test_new_live_occupant_is_rechecked_before_every_movement(self):
        env, _, snapshot = _stored_env(number_blocks=2, seed=94_203)
        pending = env.blocks[1]
        candidate = next(
            item
            for item in snapshot.candidates
            if item.action_type is ViabilityActionType.RECONFIGURE
            and len(item.recovery_action.transport_path) > 1
        )
        blocked_cell = candidate.recovery_action.transport_path[1]
        original_step = env.step

        def inject_occupant_after_pickup(action_id):
            result = original_step(action_id)
            if result[3].get("picked_block") == candidate.target_label:
                pending.position = blocked_cell
            return result

        with patch.object(env, "step", side_effect=inject_occupant_after_pickup):
            result = execute_exact_recovery_macro(
                env,
                candidate,
                gamma=0.99,
                remaining_steps=100,
                evaluation=True,
            )

        self.assertFalse(result.macro_execution.option_success)
        self.assertEqual(
            result.macro_execution.failure_reason,
            "certified_next_cell_became_occupied_before_movement",
        )
        self.assertEqual(tuple(env.current_state), candidate.recovery_action.source)
        self.assertEqual(tuple(pending.position), blocked_cell)


if __name__ == "__main__":
    unittest.main()
