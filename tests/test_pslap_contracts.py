import random
import unittest

import numpy as np

from example.small_rooms_env import SmallRoomsEnv
from PSLAP.baselines import (
    DYNAMIC_PSLAP,
    LEGACY_ADAPTED_PSLAP,
    PSLAP_GA_2009,
    PSLAP_GA_2009_OFFLINE,
    PSLAP_GA_2009_ROLLING,
    make_policy,
    normalize_baseline,
)
from PSLAP.dynamic_policy import DynamicPSLAPPolicy
from PSLAP.dynamic_yard import BlockView, YardSnapshot, select_storage_location
from PSLAP.ga_optimizer import (
    GAConfig,
    evaluate_duration_aware_rolling_assignment,
    evaluate_operational_rolling_assignment,
    evaluate_rolling_assignment,
    optimize_schedule_assignment,
)
from PSLAP.ga_policy import (
    OfflinePSLAPGAPolicy,
    PSLAPGAPolicy,
    RollingGAStorageAssigner,
    RollingPSLAPGAPolicy,
)
from PSLAP.legacy import LegacyAdaptedPSLAPPolicy
from PSLAP.retrieval_dispatch import (
    plan_retrieval,
    select_dispatchable_retrieval,
)
from PSLAP.run_pslap import run_pslap_episode


class DynamicPSLAPContracts(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        np.random.seed(7)

    def make_env(self, **kwargs):
        return SmallRoomsEnv(
            choose_storage=False,
            arrival_rate=kwargs.get("arrival_rate", 0.0),
            proc_mean=kwargs.get("proc_mean", 50),
        )

    def configure_forced_obstruction(self):
        env = self.make_env()
        env.reset()
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

        due = env.blocks[0]
        due.position = (4, 4)
        due.storage_location = due.position
        due.stored = True
        due.delivered = False
        due.storage_steps_needed = 1
        due.storage_steps_elapsed = 1
        due.stored_time_step = 19

        blocker_positions = ((3, 4), (5, 4), (4, 3), (4, 5))
        for block, position in zip(env.blocks[1:5], blocker_positions):
            block.position = position
            block.storage_location = position
            block.stored = True
            block.delivered = False
            block.storage_steps_needed = 100
            block.storage_steps_elapsed = 10
            block.stored_time_step = 10
        return env, due, env.blocks[1:5]

    def test_snapshot_uses_exact_environment_geometry(self):
        env = self.make_env()
        env.reset()
        yard = YardSnapshot.from_env(env)

        self.assertEqual(yard.exits, tuple(env.exit_cells))
        for row in range(env.grid_rows):
            for col in range(env.grid_cols):
                if env.rooms[row, col] == "#":
                    self.assertNotIn((row, col), yard.traversable)

    def test_empty_yard_assignment_is_adjacent_to_a_real_exit(self):
        env = self.make_env()
        env.reset()
        yard = YardSnapshot.from_env(env)
        incoming = env.blocks[0]
        view = BlockView(
            incoming.label,
            incoming.position,
            float(incoming.get_remaining_storage_time()),
        )
        target = select_storage_location(
            yard, view, source=incoming.position
        )
        distance = min(
            env.manhattan_distance(target, exit_cell)
            for exit_cell in env.exit_cells
        )
        self.assertEqual(distance, 1)

    def test_relocation_putdown_preserves_storage_clock(self):
        env = self.make_env()
        env.reset()
        block = env.blocks[0]
        block.stored = True
        block.carrying = True
        block.position = (4, 5)
        block.storage_location = (4, 5)
        block.stored_time_step = 3
        block.storage_steps_needed = 100
        block.storage_steps_elapsed = 11
        env.current_state = (4, 5)

        _, _, _, info = env.step(env.ACTION_IDS["PUTDOWN"])

        self.assertEqual(info["relocated_block"], block.label)
        self.assertNotIn("illegal_drop", info)
        self.assertTrue(block.stored)
        self.assertEqual(block.stored_time_step, 3)
        self.assertEqual(block.storage_steps_elapsed, 12)

    def test_dispatch_starts_at_the_predicted_retrieval_horizon(self):
        env = self.make_env()
        env.reset()
        for block in env.blocks:
            block.position = None
            block.storage_location = None
            block.carrying = False
            block.stored = False
            block.delivered = True

        target = env.blocks[0]
        target.position = env.storage_positions[len(env.storage_positions) // 2]
        target.storage_location = target.position
        target.stored = True
        target.delivered = False
        target.storage_steps_needed = 100
        target.storage_steps_elapsed = 0
        target.stored_time_step = 0

        initial = plan_retrieval(
            YardSnapshot.from_env(env), env.current_state, target.label
        )
        self.assertIsNotNone(initial)
        horizon = initial.estimated_steps

        target.storage_steps_elapsed = 100 - horizon - 1
        env.time_steps = target.storage_steps_elapsed
        self.assertIsNone(
            select_dispatchable_retrieval(
                YardSnapshot.from_env(env), env.current_state
            )
        )

        target.storage_steps_elapsed += 1
        env.time_steps += 1
        selected = select_dispatchable_retrieval(
            YardSnapshot.from_env(env), env.current_state
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.block_label, target.label)
        self.assertEqual(selected.slack, 0)

    def test_single_block_dispatch_delivers_on_its_deadline(self):
        env = self.make_env()
        env.reset()
        for block in env.blocks:
            block.position = None
            block.storage_location = None
            block.carrying = False
            block.stored = False
            block.delivered = True

        target = env.blocks[0]
        target.position = env.storage_positions[len(env.storage_positions) // 2]
        target.storage_location = target.position
        target.stored = True
        target.delivered = False
        target.storage_steps_needed = 100
        target.storage_steps_elapsed = 0
        target.stored_time_step = 0
        plan = plan_retrieval(
            YardSnapshot.from_env(env), env.current_state, target.label
        )
        self.assertIsNotNone(plan)
        target.storage_steps_elapsed = 100 - plan.estimated_steps
        env.time_steps = target.storage_steps_elapsed

        errors = []
        for action in DynamicPSLAPPolicy(env).step():
            _, _, _, info = env.step(action)
            if "delivery_error_time" in info:
                errors.append(info["delivery_error_time"])

        self.assertTrue(target.delivered)
        self.assertEqual(errors, [0])

    def test_forced_obstruction_relocates_without_planning_corruption(self):
        env, due, blockers = self.configure_forced_obstruction()
        before = {
            block.label: (
                block.position,
                block.stored,
                block.stored_time_step,
                block.storage_steps_elapsed,
            )
            for block in blockers
        }
        generator = DynamicPSLAPPolicy(env).step()
        first_action = next(generator)

        for block in blockers:
            self.assertEqual(block.position, before[block.label][0])
            self.assertEqual(block.stored, before[block.label][1])
            self.assertEqual(block.stored_time_step, before[block.label][2])
            self.assertEqual(block.storage_steps_elapsed, before[block.label][3])

        relocated = []
        illegal = []
        action = first_action
        for _ in range(200):
            _, _, _, info = env.step(action)
            if info.get("relocated_block"):
                relocated.append(info["relocated_block"])
            if info.get("illegal_drop"):
                illegal.append(info["illegal_drop"])
            try:
                action = next(generator)
            except StopIteration:
                break

        self.assertEqual(len(relocated), 1)
        self.assertEqual(illegal, [])
        self.assertTrue(due.delivered)
        moved = next(block for block in blockers if block.label == relocated[0])
        self.assertTrue(moved.stored)
        self.assertEqual(moved.stored_time_step, before[moved.label][2])
        self.assertGreater(
            moved.storage_steps_elapsed, before[moved.label][3]
        )

    def test_dispatch_plans_a_blocker_on_the_agent_approach(self):
        target = BlockView("due", (4, 0), 0.0)
        blocker = BlockView("approach-blocker", (2, 0), 100.0)
        traversable = frozenset(
            {
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (2, 1),
            }
        )
        yard = YardSnapshot(
            rows=6,
            cols=2,
            traversable=traversable,
            storage_cells=frozenset({(2, 0), (2, 1), (4, 0)}),
            exits=((5, 0),),
            blocks=(target, blocker),
        )

        plan = plan_retrieval(yard, (0, 0), target.label)

        self.assertIsNotNone(plan)
        self.assertEqual(
            [move.block_label for move in plan.relocations],
            [blocker.label],
        )
        self.assertEqual(plan.relocations[0].target, (2, 1))

    def test_dynamic_smoke_completes_without_illegal_drops(self):
        random.seed(0)
        np.random.seed(0)
        env = self.make_env(arrival_rate=0.5)
        result = run_pslap_episode(
            env, max_steps=2000, baseline=DYNAMIC_PSLAP
        )
        self.assertEqual(result["success"], 1.0)
        self.assertEqual(len(result["errors"]), env.number_blocks)
        self.assertEqual(result["illegal_drops"], 0)

    def test_duration_aware_ga_breaks_inbound_egress_distance_tie(self):
        cells = frozenset((0, col) for col in range(5))
        yard = YardSnapshot(
            rows=1,
            cols=5,
            traversable=cells,
            storage_cells=frozenset({(0, 1), (0, 3)}),
            exits=((0, 4),),
            blocks=(),
        )
        pending = (BlockView("incoming", (0, 0), 10.0),)
        near_pickup = ((0, 1),)
        near_exit = ((0, 3),)

        legacy_pickup = evaluate_rolling_assignment(
            yard, pending, near_pickup, source=(0, 0)
        )
        legacy_exit = evaluate_rolling_assignment(
            yard, pending, near_exit, source=(0, 0)
        )
        self.assertEqual(legacy_pickup.route_steps, legacy_exit.route_steps)

        improved_pickup = evaluate_duration_aware_rolling_assignment(
            yard, pending, near_pickup, source=(0, 0), egress_weight=4
        )
        improved_exit = evaluate_duration_aware_rolling_assignment(
            yard, pending, near_exit, source=(0, 0), egress_weight=4
        )
        self.assertLess(improved_exit.scalar, improved_pickup.scalar)

        operational_pickup = evaluate_operational_rolling_assignment(
            yard, pending, near_pickup, source=(0, 0), egress_weight=4
        )
        operational_exit = evaluate_operational_rolling_assignment(
            yard, pending, near_exit, source=(0, 0), egress_weight=4
        )
        self.assertLess(operational_exit.scalar, operational_pickup.scalar)


class PSLAPBaselineIdentityTests(unittest.TestCase):
    def test_pslap_alias_selects_repaired_dynamic_baseline(self):
        self.assertEqual(normalize_baseline("pslap"), DYNAMIC_PSLAP)

    def test_legacy_baseline_is_explicit_and_frozen_separately(self):
        env = SmallRoomsEnv(choose_storage=False)
        policy = make_policy(env, LEGACY_ADAPTED_PSLAP)
        self.assertIsInstance(policy, LegacyAdaptedPSLAPPolicy)

    def test_paper_inspired_ga_is_an_explicit_independent_baseline(self):
        env = SmallRoomsEnv(choose_storage=False, arrival_rate=0.5)
        env.reset()
        policy = PSLAPGAPolicy(
            env,
            GAConfig(population_size=6, generations=2, seed=11),
        )
        self.assertEqual(
            normalize_baseline(PSLAP_GA_2009), PSLAP_GA_2009_OFFLINE
        )
        self.assertEqual(len(policy.ga_result.best_assignment), env.number_blocks)
        self.assertEqual(
            len(set(policy.ga_result.best_assignment)), env.number_blocks
        )
        self.assertEqual(policy.ga_result.best_cost.infeasible_events, 0)

    def test_policy_factory_distinguishes_offline_and_rolling_ga(self):
        env = SmallRoomsEnv(choose_storage=False, arrival_rate=0.5)
        env.reset()

        offline = make_policy(env, PSLAP_GA_2009_OFFLINE)
        rolling = make_policy(env, PSLAP_GA_2009_ROLLING)

        self.assertIsInstance(offline, OfflinePSLAPGAPolicy)
        self.assertIsInstance(rolling, RollingPSLAPGAPolicy)

    def test_rolling_ga_does_not_read_unarrived_block_information(self):
        source = SmallRoomsEnv(choose_storage=False, arrival_rate=0.5)
        instance = source.sample_episode_instance(29)
        env_a = SmallRoomsEnv(choose_storage=False, arrival_rate=0.5)
        env_b = SmallRoomsEnv(choose_storage=False, arrival_rate=0.5)
        env_a.reset(instance=instance)
        env_b.reset(instance=instance)
        for index, (block_a, block_b) in enumerate(
            zip(env_a.blocks[1:], env_b.blocks[1:]), start=1
        ):
            block_a.arrival_step = 1000
            block_b.arrival_step = index
            block_a.storage_steps_needed = 1
            block_b.storage_steps_needed = 999

        config = GAConfig(population_size=6, generations=2, seed=41)
        assigner_a = RollingGAStorageAssigner(env_a, config)
        assigner_b = RollingGAStorageAssigner(env_b, config)
        current_a = env_a.blocks[0]
        current_b = env_b.blocks[0]
        view_a = BlockView(
            current_a.label,
            current_a.position,
            float(current_a.get_remaining_storage_time()),
        )
        view_b = BlockView(
            current_b.label,
            current_b.position,
            float(current_b.get_remaining_storage_time()),
        )

        target_a = assigner_a(
            YardSnapshot.from_env(env_a), view_a, current_a.position
        )
        target_b = assigner_b(
            YardSnapshot.from_env(env_b), view_b, current_b.position
        )

        self.assertEqual(assigner_a.last_known_labels, (current_a.label,))
        self.assertEqual(assigner_b.last_known_labels, (current_b.label,))
        self.assertEqual(target_a, target_b)

    def test_ga_search_is_deterministic_for_a_fixed_schedule_and_seed(self):
        random.seed(17)
        np.random.seed(17)
        env = SmallRoomsEnv(choose_storage=False, arrival_rate=0.5)
        env.reset()
        config = GAConfig(population_size=6, generations=3, seed=23)

        first = optimize_schedule_assignment(env, config)
        second = optimize_schedule_assignment(env, config)

        self.assertEqual(first.best_assignment, second.best_assignment)
        self.assertEqual(first.best_cost, second.best_cost)


if __name__ == "__main__":
    unittest.main()
