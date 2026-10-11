import unittest

from example.small_rooms_env import SmallRoomsEnv
from PSLAP.dynamic_yard import BlockView, YardSnapshot
from PSLAP.track_a import shared_candidate_mask
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_filter import (
    ROBUST_QUEUE_OBSTACLE_CONTRACT,
    ViabilitySearchConfig,
    certify_post_accept_candidates,
    post_accept_recovery_state,
)


def inbound_context(env):
    inbound = next(
        block
        for block in env.blocks
        if block.position == env.pickup_cell
        and not block.stored
        and not block.delivered
    )
    yard = YardSnapshot.from_env(env)
    view = BlockView(
        inbound.label,
        tuple(inbound.position),
        float(inbound.get_remaining_storage_time()),
    )
    return inbound, yard, view


class PostAcceptViabilityTests(unittest.TestCase):
    def test_post_state_uses_selected_cell_and_robust_queue_envelope(self):
        env = SmallRoomsEnv(
            grid_rows=5,
            grid_cols=5,
            number_blocks=1,
            choose_storage=False,
        )
        env.reset(instance=env.sample_episode_instance(17))
        inbound, yard, view = inbound_context(env)
        cell = shared_candidate_mask(yard, view, inbound.position)[0]

        state = post_accept_recovery_state(env, yard, view, cell)

        self.assertEqual(state.agent_position, cell)
        self.assertEqual(state.block(inbound.label).position, cell)
        self.assertIn(env.pickup_cell, state.fixed_obstacles)
        self.assertIn(env.waiting_cell, state.fixed_obstacles)

    def test_certification_is_order_preserving_and_cacheable(self):
        env = SmallRoomsEnv(
            grid_rows=5,
            grid_cols=5,
            number_blocks=1,
            choose_storage=False,
        )
        env.reset(instance=env.sample_episode_instance(19))
        inbound, yard, view = inbound_context(env)
        cells = tuple(reversed(shared_candidate_mask(yard, view, inbound.position)[:3]))
        cache = {}

        first = certify_post_accept_candidates(
            env, yard, view, cells, cache=cache
        )
        second = certify_post_accept_candidates(
            env, yard, view, cells, cache=cache
        )

        self.assertEqual(tuple(item.cell for item in first.candidates), cells)
        self.assertEqual(first.safe_cells, cells)
        self.assertEqual(first.unsafe_cells, ())
        self.assertEqual(first.unknown_cells, ())
        self.assertTrue(all(not item.cache_hit for item in first.candidates))
        self.assertTrue(all(item.cache_hit for item in second.candidates))
        self.assertEqual(
            first.fixed_obstacle_contract,
            ROBUST_QUEUE_OBSTACLE_CONTRACT,
        )

    def test_computational_cutoff_is_unknown_and_never_safe(self):
        env = SmallRoomsEnv(
            grid_rows=5,
            grid_cols=5,
            number_blocks=2,
            choose_storage=False,
        )
        env.reset(instance=env.sample_episode_instance(23))
        inbound, yard, view = inbound_context(env)
        stored = env.blocks[1]
        stored.position = next(
            cell for cell in env.storage_positions if cell != env.current_state
        )
        stored.storage_location = stored.position
        stored.stored = True
        stored.stored_time_step = 0
        yard = YardSnapshot.from_env(env)
        cells = shared_candidate_mask(yard, view, inbound.position)[:2]

        report = certify_post_accept_candidates(
            env,
            yard,
            view,
            cells,
            config=ViabilitySearchConfig(max_nodes=1),
        )

        self.assertEqual(report.safe_cells, ())
        self.assertEqual(report.unsafe_cells, ())
        self.assertEqual(report.unknown_cells, tuple(cells))
        self.assertTrue(
            all(
                item.status is ViabilityStatus.UNKNOWN
                for item in report.candidates
            )
        )
        self.assertEqual(
            report.audit_dict()["rejected_accept_count"], len(cells)
        )


if __name__ == "__main__":
    unittest.main()
