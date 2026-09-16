import inspect
import unittest

from PSLAP.viability import apply_recovery_action
from vcg_robust_recovery_snapshot_5x5 import (
    ROBUST_CONTRACT,
    OnDemandRecoverySolver,
    SolveStatus,
    canonical_actions,
    certify_snapshot,
    clear_stop_cells,
    execution_envelope,
    make_two_block_5x5_fixture,
    recovery_action_key,
    recovery_state_from_dict,
    recovery_state_to_dict,
)


class FrozenRecoverySnapshotFixtureTests(unittest.TestCase):
    def setUp(self):
        self.state = make_two_block_5x5_fixture()

    def test_bridge_is_closed_admission_recovery_state_only(self):
        payload = recovery_state_to_dict(self.state)

        self.assertEqual(self.state.rows, 5)
        self.assertEqual(self.state.cols, 5)
        self.assertEqual(self.state.exits, ((4, 3),))
        self.assertEqual(recovery_state_from_dict(payload), self.state)
        self.assertEqual(
            set(payload),
            {
                "rows",
                "cols",
                "traversable",
                "storage_cells",
                "exits",
                "blocks",
                "agent_position",
                "fixed_obstacles",
                "reserved_cells",
                "pickup_cells",
                "wait_cells",
            },
        )
        self.assertEqual(
            tuple(inspect.signature(certify_snapshot).parameters),
            ("state", "primitive_budget", "max_expansions", "robust_contract"),
        )

    def test_robust_envelope_is_the_complete_declared_product(self):
        actions = canonical_actions(self.state)
        expected_counts = [6, 6, 8, 8, 6, 8, 10, 8, 6, 10, 6, 8, 10, 8]

        self.assertEqual(len(actions), 14)
        self.assertEqual(
            [recovery_action_key(action) for action in actions],
            sorted(recovery_action_key(action) for action in actions),
        )
        observed_counts = []
        for action in actions:
            outcomes = execution_envelope(
                self.state,
                action,
                remaining_primitive_steps=20,
            )
            nominal_successor = apply_recovery_action(self.state, action)
            stops = clear_stop_cells(
                nominal_successor,
                action.destination,
                include_adjacent=True,
            )
            expected_coordinates = [
                (delay, stop)
                for delay in ROBUST_CONTRACT.delays
                for stop in stops
            ]

            observed_counts.append(len(outcomes))
            self.assertEqual(
                [(outcome.delay_steps, outcome.stop_cell) for outcome in outcomes],
                expected_coordinates,
            )
            self.assertEqual(sum(outcome.nominal for outcome in outcomes), 1)
            self.assertEqual(
                len({outcome.disturbance_id for outcome in outcomes}),
                len(outcomes),
            )
            for outcome in outcomes:
                expected_cost = (
                    action.steps
                    + outcome.delay_steps
                    + int(outcome.stop_cell != action.destination)
                )
                self.assertEqual(outcome.primitive_steps, expected_cost)
                self.assertTrue(outcome.tube_safe)
                self.assertEqual(outcome.horizon_safe, expected_cost <= 20)
                if outcome.horizon_safe:
                    self.assertIsNotNone(outcome.successor)
                    self.assertEqual(
                        outcome.successor.agent_position,
                        outcome.stop_cell,
                    )
                else:
                    self.assertIsNone(outcome.successor)

        self.assertEqual(observed_counts, expected_counts)
        self.assertEqual(sum(observed_counts), 108)

    def test_terminal_state_is_the_depth_zero_base_case(self):
        terminal = type(self.state)(
            rows=self.state.rows,
            cols=self.state.cols,
            traversable=self.state.traversable,
            storage_cells=self.state.storage_cells,
            exits=self.state.exits,
            blocks=(),
            agent_position=self.state.agent_position,
            fixed_obstacles=self.state.fixed_obstacles,
            reserved_cells=self.state.reserved_cells,
            pickup_cells=self.state.pickup_cells,
            wait_cells=self.state.wait_cells,
        )
        solver = OnDemandRecoverySolver(
            contract=ROBUST_CONTRACT,
            max_expansions=1,
        )

        result = solver.solve(terminal, 0)

        self.assertIs(result.status, SolveStatus.WINNING)
        self.assertEqual(result.witness_macro_depth_bound, 0)
        self.assertEqual(solver.expanded_nodes, 0)


class FrozenRecoverySnapshotCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = make_two_block_5x5_fixture()
        cls.certificates = {
            budget: certify_snapshot(
                cls.state,
                primitive_budget=budget,
                max_expansions=5_000,
            )
            for budget in (17, 18, 20)
        }

    def test_exact_nominal_one_step_recursive_separation(self):
        expected = {
            17: (SolveStatus.WINNING, SolveStatus.LOSING, SolveStatus.LOSING),
            18: (SolveStatus.WINNING, SolveStatus.WINNING, SolveStatus.LOSING),
            20: (SolveStatus.WINNING, SolveStatus.WINNING, SolveStatus.WINNING),
        }
        for budget, statuses in expected.items():
            certificate = self.certificates[budget]
            self.assertEqual(
                (
                    certificate.nominal.state.status,
                    certificate.one_step.state.status,
                    certificate.recursive.state.status,
                ),
                statuses,
            )
            self.assertTrue(
                all(
                    method.cutoff_count == 0
                    for method in (
                        certificate.nominal,
                        certificate.one_step,
                        certificate.recursive,
                    )
                )
            )

        self.assertEqual(
            self.certificates[18].one_step.admitted_action_keys,
            ("deliver:A:4:3",),
        )
        self.assertEqual(
            self.certificates[18].recursive.admitted_action_keys,
            (),
        )
        recursive = self.certificates[20].recursive
        self.assertEqual(recursive.admitted_action_keys, ("deliver:A:4:3",))
        self.assertEqual(recursive.state.witness_action_key, "deliver:A:4:3")
        self.assertEqual(recursive.state.witness_macro_depth_bound, 2)

    def test_action_selection_and_report_are_canonical(self):
        certificate = self.certificates[20]
        for method in (
            certificate.nominal,
            certificate.one_step,
            certificate.recursive,
        ):
            keys = [action.action_key for action in method.actions]
            winning = sorted(
                action.action_key
                for action in method.actions
                if action.status is SolveStatus.WINNING
            )
            self.assertEqual(keys, sorted(keys))
            self.assertEqual(method.state.witness_action_key, winning[0])

        self.assertEqual(
            certificate.semantic_digest,
            "d708fe316b95437fc7d86549073f4876ebb733b95a2997737656dbf0e89ed9b5",
        )

    def test_expansion_cutoff_is_unknown_and_never_admitted(self):
        certificate = certify_snapshot(
            self.state,
            primitive_budget=20,
            max_expansions=1,
        )

        for method in (certificate.one_step, certificate.recursive):
            self.assertIs(method.state.status, SolveStatus.UNKNOWN)
            self.assertFalse(method.state.admitted)
            self.assertEqual(method.admitted_action_keys, ())
            self.assertEqual(method.expanded_nodes, 1)
            self.assertGreater(method.cutoff_count, 0)


if __name__ == "__main__":
    unittest.main()
