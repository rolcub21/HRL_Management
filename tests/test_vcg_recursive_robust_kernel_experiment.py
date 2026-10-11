import unittest

from vcg_recursive_robust_kernel_experiment import (
    FiniteRobustGame,
    GameAction,
    GameOutcome,
    HANDLING_LAMBDAS,
    build_duration_game,
    build_endpoint_game,
    build_report,
    compute_kernel,
    compute_one_step_set,
    decreasing_frontier,
    nonincreasing_frontier,
    robust_invariant_frontier,
)


def _small_rank_game(*, include_loop=False):
    states = frozenset({"T", "G", "S", "X", "Y"})

    def action(key, *successors, tube_safe=True):
        return GameAction(
            key=key,
            kind="test",
            nominal_disturbance_id="w0",
            outcomes=tuple(
                GameOutcome(
                    disturbance_id=f"w{index}",
                    successor=successor,
                    tube_safe=tube_safe,
                    primitive_steps=1,
                    uncertainty_kind="unit_test",
                )
                for index, successor in enumerate(successors)
            ),
            q_operational=0.0,
            q_rehandles=0.0,
            physical_rehandles=0,
        )

    actions = {
        "T": (),
        "G": (action("finish", "T"),),
        "S": (action("branch", "T", "G"),),
        "X": (action("x_to_y", "Y"),),
        "Y": (action("y_to_x", "X"),),
    }
    if include_loop:
        actions["G"] = actions["G"] + (action("loop", "G"),)
    return FiniteRobustGame(
        states=states,
        terminal_states=frozenset({"T"}),
        unsafe_states=frozenset(),
        actions=actions,
        state_ids={state: state for state in states},
        uncertainty_contract="unit_test",
    )


class RecursiveKernelStructureTests(unittest.TestCase):
    def test_least_fixed_point_uses_synchronous_robust_ranks(self):
        game = _small_rank_game()
        robust = compute_kernel(game, nominal_only=False)

        self.assertEqual(robust.rank("T"), 0)
        self.assertEqual(robust.rank("G"), 1)
        self.assertEqual(robust.rank("S"), 2)
        self.assertNotIn("X", robust.states)
        self.assertNotIn("Y", robust.states)

    def test_invariance_alone_does_not_imply_completion(self):
        game = _small_rank_game(include_loop=True)
        robust = compute_kernel(game, nominal_only=False)
        invariant = robust_invariant_frontier(game, "G", robust)
        nonincreasing = nonincreasing_frontier(
            invariant, "G", robust, nominal_only=False
        )
        decreasing = decreasing_frontier(
            invariant, "G", robust, nominal_only=False
        )

        self.assertEqual({item.key for item in nonincreasing}, {"finish", "loop"})
        self.assertEqual([item.key for item in decreasing], ["finish"])

    def test_unsafe_tube_fails_closed(self):
        game = _small_rank_game()
        unsafe_action = GameAction(
            key="unsafe_tube",
            kind="test",
            nominal_disturbance_id="w0",
            outcomes=(
                GameOutcome("w0", "T", False, 1, "unit_test"),
            ),
            q_operational=100.0,
            q_rehandles=0.0,
            physical_rehandles=0,
        )
        actions = dict(game.actions)
        actions["G"] = (unsafe_action,)
        unsafe_game = FiniteRobustGame(
            states=game.states,
            terminal_states=game.terminal_states,
            unsafe_states=game.unsafe_states,
            actions=actions,
            state_ids=game.state_ids,
            uncertainty_contract=game.uncertainty_contract,
        )

        self.assertNotIn("G", compute_kernel(unsafe_game, nominal_only=False).states)

    def test_malformed_or_unresolved_outcomes_fail_closed(self):
        with self.assertRaises(TypeError):
            GameOutcome("w0", "T", "false", 1, "unit_test")
        with self.assertRaises(TypeError):
            GameOutcome("w0", "T", True, 1.5, "unit_test")

        game = _small_rank_game()
        unresolved = GameAction(
            key="unresolved",
            kind="test",
            nominal_disturbance_id="w0",
            outcomes=(GameOutcome("w0", "UNKNOWN", True, 1, "unit_test"),),
            q_operational=0.0,
            q_rehandles=0.0,
            physical_rehandles=0,
        )
        actions = dict(game.actions)
        actions["G"] = (unresolved,)
        with self.assertRaises(ValueError):
            FiniteRobustGame(
                states=game.states,
                terminal_states=game.terminal_states,
                unsafe_states=game.unsafe_states,
                actions=actions,
                state_ids=game.state_ids,
                uncertainty_contract=game.uncertainty_contract,
            )

    def test_endpoint_game_has_exact_expected_fixed_points(self):
        game = build_endpoint_game()
        nominal = compute_kernel(game, nominal_only=True)
        one_step = compute_one_step_set(game, nominal)
        robust = compute_kernel(game, nominal_only=False)

        self.assertEqual(len(game.states), 65)
        self.assertEqual(len(nominal.states), 63)
        self.assertEqual(len(one_step), 63)
        self.assertEqual(len(robust.states), 57)
        self.assertEqual(len(one_step - robust.states), 6)

    def test_duration_budget_is_part_of_membership_and_rank(self):
        game, initial = build_duration_game()
        nominal = compute_kernel(game, nominal_only=True)
        one_step = compute_one_step_set(game, nominal)
        robust = compute_kernel(game, nominal_only=False)

        self.assertIn(initial[13], nominal.states)
        self.assertNotIn(initial[13], one_step)
        self.assertIn(initial[14], one_step)
        self.assertNotIn(initial[14], robust.states)
        self.assertIn(initial[15], robust.states)
        self.assertEqual(robust.rank(initial[15]), 2)
        self.assertNotEqual(initial[14], initial[15])

    def test_every_recursive_frontier_is_invariant_and_has_progress(self):
        for game in (build_endpoint_game(), build_duration_game()[0]):
            robust = compute_kernel(game, nominal_only=False)
            for state in robust.states - game.terminal_states:
                invariant = robust_invariant_frontier(game, state, robust)
                self.assertTrue(invariant)
                self.assertTrue(
                    all(
                        outcome.tube_safe
                        and outcome.successor in robust.states
                        for action in invariant
                        for outcome in action.outcomes
                    )
                )
                decreasing = decreasing_frontier(
                    invariant,
                    state,
                    robust,
                    nominal_only=False,
                )
                self.assertTrue(decreasing)
                self.assertTrue(
                    all(
                        robust.rank(outcome.successor) < robust.rank(state)
                        for action in decreasing
                        for outcome in action.outcomes
                    )
                )
                nonincreasing = nonincreasing_frontier(
                    invariant,
                    state,
                    robust,
                    nominal_only=False,
                )
                self.assertTrue(nonincreasing)
                self.assertTrue(
                    all(
                        robust.rank(outcome.successor) <= robust.rank(state)
                        for action in nonincreasing
                        for outcome in action.outcomes
                    )
                )


class RecursiveExperimentReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_native_budget16_comparison_exposes_one_step_failure(self):
        comparison = self.report["duration_game"][
            "common_domain_budget_16_comparison"
        ]

        self.assertFalse(comparison["0.0"]["one_step"]["all_branches_complete"])
        self.assertTrue(
            all(
                comparison[str(value)]["recursive"]["all_branches_complete"]
                for value in HANDLING_LAMBDAS
            )
        )

    def test_recursive_lambda_sweep_changes_rehandles_not_certificate(self):
        duration = self.report["duration_game"]
        sweep = duration["recursive_budget_18_lambda_sweep"]
        digest = duration["summary"]["recursive_kernel_digest"]

        self.assertEqual(sweep["0.0"]["completed_max_rehandles"], 1)
        self.assertEqual(sweep["0.2"]["completed_max_rehandles"], 0)
        self.assertTrue(all(row["all_branches_complete"] for row in sweep.values()))
        self.assertEqual(len(digest), 64)

    def test_out_of_set_delay_is_explicitly_outside_the_guarantee(self):
        scope = self.report["duration_game"]["out_of_set_scope"]

        self.assertFalse(scope["covered_by_declared_guarantee"])
        self.assertTrue(scope["budget_15_in_declared_recursive_kernel"])
        self.assertFalse(scope["budget_15_in_expanded_recursive_kernel"])
        self.assertEqual(scope["expanded_recursive_minimum_budget"], 17)

    def test_report_is_deterministic_and_scoped(self):
        second = build_report()

        self.assertEqual(self.report, second)
        self.assertEqual(self.report["status"], "passed")
        self.assertTrue(all(self.report["checks"].values()))
        self.assertFalse(self.report["training_or_learning"])
        self.assertFalse(self.report["full_dynamic_episode_claim"])
        self.assertFalse(self.report["joint_duration_and_endpoint_kernel"])
        self.assertFalse(
            self.report["completion_policy_contract"]
            ["invariant_frontier_alone_implies_completion"]
        )
        self.assertTrue(
            self.report["uncertainty_enumeration"]
            ["declared_outcomes_are_finite_and_complete"]
        )


if __name__ == "__main__":
    unittest.main()
