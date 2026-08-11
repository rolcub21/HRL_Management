import unittest
from unittest.mock import patch

import numpy as np
import torch

from example.small_rooms_env import SmallRoomsEnv
from example.Options.selector import StorageSelectOption
from PSLAP.dynamic_yard import BlockView, YardSnapshot, shortest_clear_path
from PSLAP.reg_selector_v4 import (
    REGV4AssignmentSource,
    REGV4Config,
    build_observation,
)
from PSLAP.retrieval_dispatch import select_dispatchable_retrieval
from PSLAP.track_a import (
    NearestFreeAssignmentSource,
    StrictAssignmentAdapter,
    TrackAOnlinePolicy,
    REGSelectorAssignmentSource,
    TRACK_A_NEAREST_FREE,
    TRACK_A_REG_SELECTOR_V4,
    run_track_a_episode,
    shared_candidate_mask,
)


class FixedSource:
    def __init__(self, proposal):
        self.proposal = proposal
        self.received_masks = []

    def propose(self, yard, block, source, valid_candidates):
        self.received_masks.append(valid_candidates)
        return self.proposal


class FailingIfCalledSource:
    def propose(self, yard, block, source, valid_candidates):
        raise AssertionError("source must not be called without candidates")


class TrackAAssignmentContracts(unittest.TestCase):
    def make_env(self, **kwargs):
        return SmallRoomsEnv(
            choose_storage=False,
            arrival_rate=kwargs.get("arrival_rate", 0.5),
            proc_mean=kwargs.get("proc_mean", 50),
        )

    def inbound_context(self, env):
        inbound = next(
            block for block in env.blocks
            if block.position == env.pickup_cell and not block.stored
        )
        view = BlockView(
            inbound.label,
            inbound.position,
            float(inbound.get_remaining_storage_time()),
        )
        return YardSnapshot.from_env(env), view, inbound.position

    def test_valid_proposal_uses_the_exact_shared_mask(self):
        env = self.make_env()
        env.reset()
        yard, block, source = self.inbound_context(env)
        candidates = shared_candidate_mask(yard, block, source)
        assignment_source = FixedSource(candidates[-1])
        adapter = StrictAssignmentAdapter(
            env, "fixed_valid", assignment_source
        )

        executed = adapter(yard, block, source)

        self.assertEqual(assignment_source.received_masks, [candidates])
        self.assertEqual(executed, candidates[-1])
        self.assertEqual(adapter.invalid_assignment_count, 0)
        self.assertEqual(adapter.fallback_count, 0)
        self.assertFalse(adapter.contaminated)

    def test_invalid_proposal_is_recorded_before_neutral_fallback(self):
        env = self.make_env()
        env.reset()
        yard, block, source = self.inbound_context(env)
        adapter = StrictAssignmentAdapter(
            env, "invalid", FixedSource((999, 999))
        )

        executed = adapter(yard, block, source)
        decision = adapter.decisions[0]

        self.assertEqual(decision.proposed_cell, (999, 999))
        self.assertEqual(decision.invalid_reason, "non_storage_cell")
        self.assertTrue(decision.fallback_used)
        self.assertNotEqual(decision.executed_cell, decision.proposed_cell)
        self.assertEqual(executed, decision.executed_cell)
        self.assertIn(executed, decision.valid_candidates)
        self.assertEqual(adapter.invalid_assignment_count, 1)
        self.assertEqual(adapter.fallback_count, 1)
        self.assertTrue(adapter.contaminated)

    def test_no_candidates_is_system_infeasibility_not_method_failure(self):
        env = self.make_env()
        env.reset()
        yard, block, source = self.inbound_context(env)
        occupied = tuple(
            BlockView(f"X{index}", cell, 100.0)
            for index, cell in enumerate(yard.storage_cells)
        )
        full_yard = YardSnapshot(
            rows=yard.rows,
            cols=yard.cols,
            traversable=yard.traversable,
            storage_cells=yard.storage_cells,
            exits=yard.exits,
            blocks=occupied,
        )
        adapter = StrictAssignmentAdapter(
            env, "not_called", FailingIfCalledSource()
        )

        executed = adapter(full_yard, block, source)

        self.assertIsNone(executed)
        self.assertEqual(adapter.infeasible_epoch_count, 1)
        self.assertEqual(adapter.invalid_assignment_count, 0)
        self.assertFalse(adapter.contaminated)

    def test_unreachable_pickup_defers_before_calling_assignment_source(self):
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(72))
        source = FailingIfCalledSource()
        policy = TrackAOnlinePolicy(env, "not_called", source)

        with patch.object(env, "plan_path_heuristic", return_value=[]):
            generator = policy.step()
            self.assertEqual(next(generator), env.ACTION_IDS["WAIT"])
            with self.assertRaises(StopIteration):
                next(generator)

        self.assertEqual(policy.assignment_audit.decisions, [])
        self.assertEqual(policy.assignment_audit.invalid_assignment_count, 0)
        self.assertEqual(policy.inbound_approach_defer_count, 1)

    def test_retrieval_planning_respects_live_inbound_obstacles(self):
        env = SmallRoomsEnv(
            grid_rows=6,
            grid_cols=8,
            exit_cells=[(5, 6)],
            choose_storage=False,
            number_blocks=12,
            arrival_rate=1.5,
            proc_mean=100,
        )
        env.reset(instance=env.sample_episode_instance(34002))
        stored_positions = (
            (4, 6),
            (4, 5),
            (3, 6),
            (4, 4),
            (3, 5),
            (4, 3),
            (2, 6),
            (3, 4),
            (2, 5),
            (4, 2),
        )
        for block, position in zip(env.blocks[:10], stored_positions):
            block.position = position
            block.storage_location = position
            block.stored = True
            block.delivered = False
            block.carrying = False
            block.stored_time_step = 0
            block.storage_steps_needed = 1
        inbound = env.blocks[10]
        inbound.position = env.pickup_cell
        inbound.storage_location = None
        inbound.stored = False
        inbound.delivered = False
        inbound.carrying = False
        waiting = env.blocks[11]
        waiting.position = env.waiting_cell
        waiting.storage_location = None
        waiting.stored = False
        waiting.delivered = False
        waiting.carrying = False
        env.current_state = (4, 2)
        env.time_steps = 3_000

        policy = TrackAOnlinePolicy(
            env,
            TRACK_A_NEAREST_FREE,
            NearestFreeAssignmentSource(),
        )
        raw_yard = YardSnapshot.from_env(env)
        raw_plan = select_dispatchable_retrieval(
            raw_yard,
            env.current_state,
            relocation_selector=policy.relocation_selector,
        )
        live_plan = select_dispatchable_retrieval(
            policy._retrieval_snapshot(),
            env.current_state,
            relocation_selector=policy.relocation_selector,
        )

        self.assertIsNotNone(raw_plan)
        first_relocation = raw_plan.relocations[0]
        relocated_block = raw_yard.block(first_relocation.block_label)
        raw_approach = shortest_clear_path(
            raw_yard,
            env.current_state,
            relocated_block.position,
            ignore_labels=(relocated_block.label,),
        )
        self.assertIn(env.pickup_cell, raw_approach)
        self.assertIsNone(live_plan)
        first_action = next(policy.step())
        self.assertNotEqual(first_action, env.ACTION_IDS["WAIT"])
        self.assertEqual(policy.retrieval_live_plan_failure_count, 0)

    def test_exact_recovery_fallback_breaks_greedy_retrieval_stall(self):
        env = SmallRoomsEnv(
            grid_rows=6,
            grid_cols=8,
            exit_cells=[(5, 6)],
            choose_storage=False,
            number_blocks=12,
            arrival_rate=1.5,
            proc_mean=100,
        )
        env.reset(instance=env.sample_episode_instance(34002))
        stored_positions = (
            (4, 6),
            (4, 5),
            (3, 6),
            (4, 4),
            (3, 5),
            (4, 3),
            (2, 6),
            (3, 4),
            (2, 5),
            (4, 2),
            (2, 4),
            (1, 4),
        )
        for block, position in zip(env.blocks, stored_positions):
            block.position = position
            block.storage_location = position
            block.stored = True
            block.delivered = False
            block.carrying = False
            block.stored_time_step = 0
            block.storage_steps_needed = 1
        env.current_state = (1, 4)
        env.time_steps = 3_000

        policy = TrackAOnlinePolicy(
            env,
            TRACK_A_NEAREST_FREE,
            NearestFreeAssignmentSource(),
        )
        self.assertIsNone(
            select_dispatchable_retrieval(
                policy._retrieval_snapshot(),
                env.current_state,
                relocation_selector=policy.relocation_selector,
            )
        )

        first_info = None
        for action in policy.step():
            self.assertNotEqual(action, env.ACTION_IDS["WAIT"])
            _, _, _, first_info = env.step(action)

        self.assertEqual(policy.exact_recovery_search_count, 1)
        self.assertEqual(policy.exact_recovery_fallback_count, 1)
        self.assertEqual(policy.exact_recovery_failure_count, 0)
        self.assertTrue(first_info["relocated_block"])

        steps = 0
        while not env.is_state_terminal(env.current_state) and steps < 1_000:
            for action in policy.step():
                _, _, _, _ = env.step(action)
                steps += 1
                if steps >= 1_000:
                    break

        self.assertTrue(env.is_state_terminal(env.current_state))
        self.assertEqual(policy.exact_recovery_failure_count, 0)
        self.assertGreaterEqual(policy.exact_recovery_fallback_count, 1)

    def test_exact_recovery_clears_reachable_inbound_without_candidates(self):
        env = SmallRoomsEnv(
            grid_rows=6,
            grid_cols=8,
            exit_cells=[(5, 6)],
            choose_storage=False,
            number_blocks=13,
            arrival_rate=1.5,
            proc_mean=100,
        )
        env.reset(instance=env.sample_episode_instance(34004))
        stored_positions = (
            (4, 6),
            (4, 5),
            (3, 6),
            (4, 4),
            (3, 5),
            (4, 3),
            (2, 6),
            (3, 4),
            (2, 5),
            (4, 2),
            (2, 4),
            (1, 5),
        )
        for block, position in zip(env.blocks[:12], stored_positions):
            block.position = position
            block.storage_location = position
            block.stored = True
            block.delivered = False
            block.carrying = False
            block.stored_time_step = 0
            block.storage_steps_needed = 1
        inbound = env.blocks[12]
        inbound.position = env.pickup_cell
        inbound.storage_location = None
        inbound.stored = False
        inbound.delivered = False
        inbound.carrying = False
        env.current_state = env.pickup_cell
        env.time_steps = 3_000

        policy = TrackAOnlinePolicy(
            env,
            TRACK_A_NEAREST_FREE,
            NearestFreeAssignmentSource(),
        )
        inbound_view = BlockView(
            inbound.label,
            inbound.position,
            float(inbound.get_remaining_storage_time()),
        )
        self.assertEqual(
            shared_candidate_mask(
                YardSnapshot.from_env(env),
                inbound_view,
                inbound.position,
            ),
            (),
        )
        first_info = None
        for action in policy.step():
            self.assertNotEqual(action, env.ACTION_IDS["WAIT"])
            _, _, _, first_info = env.step(action)

        self.assertEqual(policy.assignment_audit.infeasible_epoch_count, 1)
        self.assertEqual(policy.assignment_audit.invalid_assignment_count, 0)
        self.assertEqual(policy.exact_recovery_search_count, 1)
        self.assertEqual(policy.exact_recovery_fallback_count, 1)
        self.assertEqual(policy.exact_recovery_failure_count, 0)
        self.assertTrue(first_info["relocated_block"])
        self.assertTrue(
            shared_candidate_mask(
                YardSnapshot.from_env(env),
                inbound_view,
                inbound.position,
            )
        )

    def test_operational_fallback_is_excluded_from_strict_result(self):
        env = self.make_env(arrival_rate=0.5)
        instance = env.sample_episode_instance(73)

        result = run_track_a_episode(
            env,
            TRACK_A_NEAREST_FREE,
            max_steps=2000,
            episode_instance=instance,
            assignment_source=FixedSource((999, 999)),
        )

        self.assertEqual(result["success"], 1.0)
        self.assertEqual(result["completion_fraction"], 1.0)
        self.assertEqual(result["strict_method_success"], 0.0)
        self.assertEqual(result["strict_completion_fraction"], 0.0)
        self.assertGreater(result["invalid_assignment_count"], 0)
        self.assertEqual(
            result["fallback_count"], result["invalid_assignment_count"]
        )
        self.assertEqual(result["fallback_contaminated"], 1)
        self.assertGreater(result["contaminated_steps"], 0)
        self.assertEqual(result["illegal_drops"], 0)

        expected_flow_times = [
            block.stored_time_step - block.arrival_step
            for block in env.blocks
        ]
        self.assertEqual(
            result["storage_flow_observation_end_step"], env.time_steps
        )
        self.assertEqual(
            result["storage_flow_manifest_count"], len(env.blocks)
        )
        self.assertEqual(
            result["storage_flow_arrived_count"], len(env.blocks)
        )
        self.assertEqual(
            result["storage_flow_completed_count"], len(env.blocks)
        )
        self.assertEqual(result["storage_flow_right_censored_count"], 0)
        self.assertEqual(result["storage_flow_not_yet_arrived_count"], 0)
        self.assertEqual(result["storage_flow_unfinished_count"], 0)
        self.assertEqual(result["storage_flow_completion_rate"], 1.0)
        self.assertEqual(result["storage_flow_arrived_completion_rate"], 1.0)
        self.assertTrue(result["storage_flow_fully_observed"])
        self.assertEqual(
            result["completed_storage_flow_times"], expected_flow_times
        )
        self.assertEqual(result["right_censored_storage_flow_ages"], [])
        self.assertEqual(
            len(result["storage_flow_records"]), len(env.blocks)
        )
        self.assertAlmostEqual(
            result["mean_storage_flow_time"],
            float(np.mean(expected_flow_times)),
        )

    def test_truncated_episode_reports_right_censored_storage_flow(self):
        env = self.make_env(arrival_rate=0.5)
        instance = env.sample_episode_instance(74)

        result = run_track_a_episode(
            env,
            TRACK_A_NEAREST_FREE,
            max_steps=1,
            episode_instance=instance,
        )

        self.assertEqual(result["success"], 0.0)
        self.assertEqual(result["steps"], 1)
        self.assertEqual(
            result["storage_flow_observation_end_step"], env.time_steps
        )
        self.assertEqual(
            result["storage_flow_manifest_count"], len(env.blocks)
        )
        self.assertEqual(result["storage_flow_completed_count"], 0)
        self.assertGreater(result["storage_flow_right_censored_count"], 0)
        self.assertGreater(result["storage_flow_not_yet_arrived_count"], 0)
        self.assertEqual(
            result["storage_flow_unfinished_count"], len(env.blocks)
        )
        self.assertEqual(result["storage_flow_completion_rate"], 0.0)
        self.assertFalse(result["storage_flow_fully_observed"])
        self.assertEqual(result["completed_storage_flow_times"], [])
        self.assertEqual(
            len(result["right_censored_storage_flow_ages"]),
            result["storage_flow_right_censored_count"],
        )
        self.assertEqual(
            len(result["storage_flow_records"]), len(env.blocks)
        )
        self.assertTrue(np.isnan(result["mean_storage_flow_time"]))

    def test_reg_selector_does_not_use_unarrived_storage_durations(self):
        source_env = self.make_env()
        instance = source_env.sample_episode_instance(83)
        env_a = self.make_env()
        env_b = self.make_env()
        env_a.reset(instance=instance)
        env_b.reset(instance=instance)
        for block_a, block_b in zip(env_a.blocks, env_b.blocks):
            if block_a.position is None and block_b.position is None:
                block_a.storage_steps_needed = 1
                block_b.storage_steps_needed = 999

        selector_a = StorageSelectOption(env_a)
        selector_b = StorageSelectOption(env_b)
        selector_b.q.load_state_dict(selector_a.q.state_dict())
        source_a = REGSelectorAssignmentSource(env_a, selector_a)
        source_b = REGSelectorAssignmentSource(env_b, selector_b)
        yard_a, block_a, pickup_a = self.inbound_context(env_a)
        yard_b, block_b, pickup_b = self.inbound_context(env_b)
        candidates_a = shared_candidate_mask(yard_a, block_a, pickup_a)
        candidates_b = shared_candidate_mask(yard_b, block_b, pickup_b)

        target_a = source_a.propose(
            yard_a, block_a, pickup_a, candidates_a
        )
        target_b = source_b.propose(
            yard_b, block_b, pickup_b, candidates_b
        )

        self.assertEqual(StorageSelectOption.FEATURE_VERSION, 3)
        self.assertEqual(candidates_a, candidates_b)
        self.assertEqual(target_a, target_b)

    def test_reg_v4_is_block_permutation_and_candidate_cardinality_invariant(self):
        torch.manual_seed(5)
        env = self.make_env()
        env.reset(instance=env.sample_episode_instance(91))
        yard, block, source = self.inbound_context(env)
        candidates = shared_candidate_mask(yard, block, source)
        observation = build_observation(
            env, yard, block, source, candidates
        )
        selector = REGV4AssignmentSource(env, learning_enabled=False)

        def scores(blocks, mask, candidate_features):
            with torch.no_grad():
                return selector.network(
                    torch.from_numpy(blocks).unsqueeze(0),
                    torch.from_numpy(mask).unsqueeze(0),
                    torch.from_numpy(observation.current_block_features).unsqueeze(0),
                    torch.from_numpy(observation.global_features).unsqueeze(0),
                    torch.from_numpy(candidate_features).unsqueeze(0),
                )[0]

        original = scores(
            observation.block_features,
            observation.block_mask,
            observation.candidate_features,
        )
        permutation = np.arange(len(observation.block_features))[::-1].copy()
        permuted = scores(
            observation.block_features[permutation],
            observation.block_mask[permutation],
            observation.candidate_features,
        )
        singleton = scores(
            observation.block_features,
            observation.block_mask,
            observation.candidate_features[:1],
        )

        torch.testing.assert_close(original, permuted)
        torch.testing.assert_close(original[:1], singleton)

    def test_reg_v4_does_not_use_unarrived_storage_durations(self):
        torch.manual_seed(7)
        base = self.make_env()
        instance = base.sample_episode_instance(92)
        env_a = self.make_env()
        env_b = self.make_env()
        env_a.reset(instance=instance)
        env_b.reset(instance=instance)
        for item_a, item_b in zip(env_a.blocks, env_b.blocks):
            if item_a.position is None:
                item_a.storage_steps_needed = 1
                item_b.storage_steps_needed = 999
        source_a = REGV4AssignmentSource(env_a, learning_enabled=False)
        source_b = REGV4AssignmentSource(env_b, learning_enabled=False)
        source_b.network.load_state_dict(source_a.network.state_dict())
        yard_a, block_a, pickup_a = self.inbound_context(env_a)
        yard_b, block_b, pickup_b = self.inbound_context(env_b)
        candidates_a = shared_candidate_mask(yard_a, block_a, pickup_a)
        candidates_b = shared_candidate_mask(yard_b, block_b, pickup_b)

        target_a = source_a.propose(
            yard_a, block_a, pickup_a, candidates_a
        )
        target_b = source_b.propose(
            yard_b, block_b, pickup_b, candidates_b
        )

        self.assertEqual(candidates_a, candidates_b)
        self.assertEqual(target_a, target_b)

    def test_reg_v4_learns_complete_track_a_events_and_round_trips(self):
        torch.manual_seed(11)
        env = self.make_env()
        instance = env.sample_episode_instance(93)
        config = REGV4Config(
            batch_size=8,
            min_replay_size=8,
            updates_per_episode=1,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_warmup_assignments=0,
            epsilon_decay_assignments=1,
        )
        source = REGV4AssignmentSource(
            env, config, seed=11, learning_enabled=True
        )

        result = run_track_a_episode(
            env,
            TRACK_A_REG_SELECTOR_V4,
            max_steps=2000,
            episode_instance=instance,
            assignment_source=source,
        )

        self.assertEqual(result["success"], 1.0)
        self.assertEqual(result["invalid_assignment_count"], 0)
        self.assertEqual(len(source.replay), len(env.blocks))
        self.assertEqual(source.completed_events, len(env.blocks))
        self.assertEqual(source.failed_events, 0)
        self.assertEqual(source.gradient_steps, 1)
        self.assertTrue(np.isfinite(source.loss_history[-1]))

        restored = REGV4AssignmentSource.from_checkpoint(
            env,
            source.checkpoint(),
            learning_enabled=False,
        )
        for key, value in source.network.state_dict().items():
            torch.testing.assert_close(value.cpu(), restored.network.state_dict()[key])


if __name__ == "__main__":
    unittest.main()
