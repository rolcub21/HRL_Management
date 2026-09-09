import unittest

from PSLAP.dynamic_yard import BlockView, YardSnapshot
from PSLAP.viability import (
    CLOSED_ADMISSION_CONTRACT,
    STRICT_MACRO_ACTION_MODEL,
    RecoveryActionKind,
    RecoveryState,
    ViabilityStatus,
    analyze_recoverability,
    apply_recovery_action,
    legal_recovery_actions,
)


def make_yard(
    traversable,
    storage_cells,
    exits,
    blocks=(),
):
    cells = frozenset(traversable)
    rows = max((row for row, _ in cells), default=0) + 1
    cols = max((col for _, col in cells), default=0) + 1
    return YardSnapshot(
        rows=rows,
        cols=cols,
        traversable=cells,
        storage_cells=frozenset(storage_cells),
        exits=tuple(exits),
        blocks=tuple(blocks),
    )


def direct_yard():
    cells = ((0, 0), (0, 1), (0, 2))
    return make_yard(
        cells,
        ((0, 1),),
        ((0, 2),),
        (BlockView("A", (0, 1), 10.0),),
    )


def recovery_state(yard, agent=(0, 0), **kwargs):
    return RecoveryState.from_yard_snapshot(
        yard, agent, **kwargs
    )


def relocation_yard(blocks=None):
    # A and B occupy a one-cell-wide egress corridor.  A must first move into
    # the branch, after which B and then A can be delivered.
    cells = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1))
    return make_yard(
        cells,
        ((0, 1), (0, 2), (1, 1)),
        ((0, 3),),
        blocks
        or (
            BlockView("A", (0, 1), 20.0),
            BlockView("B", (0, 2), 10.0),
        ),
    )


class LegalRecoveryActionTests(unittest.TestCase):
    def test_direct_delivery_is_a_physical_macro_with_complete_paths(self):
        state = recovery_state(direct_yard())

        actions = legal_recovery_actions(state)
        deliveries = [
            action
            for action in actions
            if action.kind is RecoveryActionKind.DELIVERY
        ]

        self.assertEqual(len(deliveries), 1)
        action = deliveries[0]
        self.assertEqual(action.block_label, "A")
        self.assertEqual(action.source, (0, 1))
        self.assertEqual(action.destination, (0, 2))
        self.assertEqual(action.approach_path, ((0, 0), (0, 1)))
        self.assertEqual(action.transport_path, ((0, 1), (0, 2)))
        self.assertEqual(action.steps, 4)

        successor = apply_recovery_action(state, action)
        self.assertEqual(successor.blocks, ())
        self.assertEqual(successor.agent_position, (0, 2))

    def test_relocation_changes_position_without_removing_the_block(self):
        state = recovery_state(relocation_yard())

        actions = legal_recovery_actions(state)

        self.assertEqual(len(actions), 1)
        action = actions[0]
        self.assertIs(action.kind, RecoveryActionKind.RELOCATION)
        self.assertEqual(action.block_label, "A")
        self.assertEqual(action.destination, (1, 1))

        successor = apply_recovery_action(state, action)
        self.assertEqual(successor.block("A").position, (1, 1))
        self.assertEqual(successor.block("B").position, (0, 2))
        self.assertEqual(len(successor.blocks), 2)
        self.assertEqual(successor.agent_position, (1, 1))

    def test_fixed_obstacles_are_not_silently_ignored(self):
        state = recovery_state(
            direct_yard(), fixed_obstacles=((0, 2),)
        )

        actions = legal_recovery_actions(state)
        certificate = analyze_recoverability(
            state,
            max_nodes=None,
        )

        self.assertEqual(actions, ())
        self.assertIs(certificate.status, ViabilityStatus.UNSAFE)
        self.assertEqual(certificate.fixed_obstacles, frozenset({(0, 2)}))
        self.assertEqual(certificate.contract, CLOSED_ADMISSION_CONTRACT)
        self.assertEqual(
            certificate.action_model, STRICT_MACRO_ACTION_MODEL
        )
        self.assertIn("fixed_obstacles", certificate.reason)

    def test_storage_reservation_removes_a_relocation_target(self):
        state = recovery_state(
            relocation_yard(), reserved_cells=((1, 1),)
        )

        actions = legal_recovery_actions(state)
        certificate = analyze_recoverability(state, max_nodes=None)

        self.assertEqual(actions, ())
        self.assertIs(certificate.status, ViabilityStatus.UNSAFE)
        self.assertEqual(certificate.reserved_cells, frozenset({(1, 1)}))
        self.assertIn("storage_reservations", certificate.reason)

    def test_snapshot_adapter_round_trips_geometry_and_inventory(self):
        yard = relocation_yard()
        state = recovery_state(
            yard,
            agent=(1, 1),
            fixed_obstacles=((0, 0),),
        )

        adapted = state.to_yard_snapshot()

        self.assertEqual(adapted, yard)
        self.assertEqual(state.agent_position, (1, 1))
        self.assertEqual(state.fixed_obstacles, frozenset({(0, 0)}))


class RecoverabilitySearchTests(unittest.TestCase):
    def test_empty_yard_is_safe_with_an_empty_witness(self):
        yard = make_yard(
            ((0, 0), (0, 1)),
            ((0, 0),),
            ((0, 1),),
        )

        result = analyze_recoverability(recovery_state(yard))

        self.assertIs(result.status, ViabilityStatus.SAFE)
        self.assertTrue(result.is_safe)
        self.assertEqual(result.witness, ())
        self.assertTrue(result.exhaustive)

    def test_finds_and_replays_a_deterministic_relocation_witness(self):
        yard = relocation_yard()
        state = recovery_state(yard)

        first = analyze_recoverability(
            state, max_depth=None, max_nodes=None
        )
        second = analyze_recoverability(
            recovery_state(
                relocation_yard(
                    blocks=tuple(reversed(yard.blocks))
                )
            ),
            max_depth=None,
            max_nodes=None,
        )

        self.assertIs(first.status, ViabilityStatus.SAFE)
        self.assertEqual(first.witness, second.witness)
        self.assertEqual(
            [action.kind for action in first.witness],
            [
                RecoveryActionKind.RELOCATION,
                RecoveryActionKind.DELIVERY,
                RecoveryActionKind.DELIVERY,
            ],
        )
        self.assertEqual(
            [action.block_label for action in first.witness],
            ["A", "B", "A"],
        )

        current = state
        for action in first.witness:
            current = apply_recovery_action(current, action)
        self.assertEqual(current.blocks, ())

    def test_goal_directed_order_preserves_certificate_semantics(self):
        state = recovery_state(relocation_yard())

        breadth_first = analyze_recoverability(
            state,
            max_depth=None,
            max_nodes=None,
            search_order="breadth_first",
        )
        goal_directed = analyze_recoverability(
            state,
            max_depth=None,
            max_nodes=None,
            search_order="goal_directed",
        )

        self.assertIs(breadth_first.status, ViabilityStatus.SAFE)
        self.assertIs(goal_directed.status, ViabilityStatus.SAFE)
        self.assertTrue(breadth_first.recovery_rank_is_exact)
        self.assertEqual(
            breadth_first.exact_recovery_rank,
            len(breadth_first.witness),
        )
        self.assertFalse(goal_directed.recovery_rank_is_exact)
        self.assertIsNone(goal_directed.exact_recovery_rank)
        self.assertEqual(goal_directed.search_order, "goal_directed")
        current = state
        for action in goal_directed.witness:
            current = apply_recovery_action(current, action)
        self.assertEqual(current.blocks, ())

    def test_goal_directed_depth_cutoff_does_not_hide_shorter_routes(self):
        result = analyze_recoverability(
            recovery_state(relocation_yard()),
            max_depth=2,
            max_nodes=None,
            search_order="goal_directed",
        )

        self.assertIs(result.status, ViabilityStatus.UNKNOWN)
        self.assertIn("depth_budget_exhausted", result.reason)

    def test_exhaustive_cycle_search_proves_unsafe(self):
        # The block can be moved forever inside the left component, but the
        # exit is in a disconnected component.  Memoization must exhaust the
        # finite relocation graph rather than loop or report a false cutoff.
        yard = make_yard(
            ((0, 0), (0, 1), (0, 2), (2, 2)),
            ((0, 0), (0, 1), (0, 2)),
            ((2, 2),),
            (BlockView("A", (0, 1), 10.0),),
        )

        result = analyze_recoverability(
            recovery_state(yard), max_depth=None, max_nodes=None
        )

        self.assertIs(result.status, ViabilityStatus.UNSAFE)
        self.assertTrue(result.is_unsafe)
        self.assertTrue(result.exhaustive)
        self.assertEqual(result.frontier_states, 0)
        self.assertGreater(result.explored_nodes, 1)
        self.assertEqual(result.explored_nodes, result.generated_states)

    def test_depth_cutoff_is_unknown_not_unsafe(self):
        yard = relocation_yard()

        result = analyze_recoverability(
            recovery_state(yard), max_depth=2, max_nodes=None
        )

        self.assertIs(result.status, ViabilityStatus.UNKNOWN)
        self.assertTrue(result.is_unknown)
        self.assertFalse(result.exhaustive)
        self.assertEqual(result.witness, ())
        self.assertIn("depth_budget_exhausted", result.reason)
        self.assertGreater(result.frontier_states, 0)

    def test_node_cutoff_is_unknown_not_unsafe(self):
        yard = relocation_yard()

        result = analyze_recoverability(
            recovery_state(yard), max_depth=None, max_nodes=1
        )

        self.assertIs(result.status, ViabilityStatus.UNKNOWN)
        self.assertFalse(result.exhaustive)
        self.assertEqual(result.explored_nodes, 1)
        self.assertIn("node_budget_exhausted", result.reason)

    def test_dead_end_is_proven_unsafe_even_at_depth_boundary(self):
        # With no spare storage cell, A cannot move aside, B cannot be
        # approached, and neither block has a clear carrying path to the exit.
        yard = make_yard(
            ((0, 0), (0, 1), (0, 2), (0, 3)),
            ((0, 1), (0, 2)),
            ((0, 3),),
            (
                BlockView("A", (0, 1), 20.0),
                BlockView("B", (0, 2), 10.0),
            ),
        )

        result = analyze_recoverability(
            recovery_state(yard), max_depth=0, max_nodes=None
        )

        self.assertIs(result.status, ViabilityStatus.UNSAFE)
        self.assertTrue(result.exhaustive)

    def test_rejects_invalid_budget_and_obstacle_contracts(self):
        state = recovery_state(direct_yard())

        with self.assertRaisesRegex(ValueError, "max_depth"):
            analyze_recoverability(state, max_depth=-1)
        with self.assertRaisesRegex(ValueError, "max_nodes"):
            analyze_recoverability(state, max_nodes=0)
        with self.assertRaisesRegex(ValueError, "search_order"):
            analyze_recoverability(state, search_order="optimistic")
        with self.assertRaisesRegex(ValueError, "overlap stored blocks"):
            recovery_state(
                direct_yard(), fixed_obstacles=((0, 1),)
            )

    def test_primitive_step_horizon_is_part_of_the_certificate(self):
        state = recovery_state(direct_yard())

        too_short = analyze_recoverability(
            state, max_nodes=None, max_primitive_steps=3
        )
        exact = analyze_recoverability(
            state, max_nodes=None, max_primitive_steps=4
        )

        self.assertIs(too_short.status, ViabilityStatus.UNSAFE)
        self.assertTrue(too_short.exhaustive)
        self.assertEqual(too_short.primitive_step_budget, 3)
        self.assertIn("within_primitive_step_budget", too_short.reason)
        self.assertIs(exact.status, ViabilityStatus.SAFE)
        self.assertEqual(exact.witness_primitive_steps, 4)


if __name__ == "__main__":
    unittest.main()
