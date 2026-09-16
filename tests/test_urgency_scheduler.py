from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from compare_track_b import FIELDNAMES, format_urgency_row
from example.Options.AcceptStoreOption import AcceptStoreOption
from example.Options.RetrieveDeliverOption import (
    RetrieveDeliverOption,
    StrictRetrieveDeliverOption,
)
from example.Options.StrategicDeferOption import StrategicDeferOption
from example.controller_options import build_controller_options
from example.small_rooms_env import SmallRoomsEnv
from example.urgency_scheduler import (
    DURATION_AWARE_METHOD,
    DurationAwareAtomicScheduler,
    SchedulerInfeasibleError,
    URGENCY_FIRST_ACTION_INTERFACE,
    URGENCY_FIRST_METHOD,
    UrgencyFirstAtomicScheduler,
)
from gated_agent import ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE
from options_agent import canonical_manager_options, controller_ids
from PSLAP.checkpoint_identity import selector_deployment_digest
from PSLAP.reg_selector_v5 import REGV5AssignmentSource, REGV5Config
from PSLAP.retrieval_context import retrieval_planning_context
from tests.test_scheduler_options import DummySelector, isolated_retrieval_env
from tests.test_atomic_inbound_scheduler import make_inbound_system
from track_b_urgency_evaluate import (
    evaluate_duration_aware_one,
    evaluate_one,
)


def make_policy_system(
    *,
    inbound=False,
    due=False,
    scheduler_class=UrgencyFirstAtomicScheduler,
    margin_steps=0.0,
):
    env, target = isolated_retrieval_env()
    target_plan = retrieval_planning_context(env).plan(target.label)
    if due:
        target.storage_steps_needed = target_plan.estimated_steps
    if inbound:
        block = env.blocks[1]
        block.position = env.pickup_cell
        block.storage_location = None
        block.carrying = False
        block.stored = False
        block.delivered = False
        block.arrival_step = 0
    selector = DummySelector(env)
    build_controller_options(
        env,
        selector,
        controller_action_interface=URGENCY_FIRST_ACTION_INTERFACE,
        max_defer_steps=5,
    )
    scheduler = (
        scheduler_class(env, margin_steps=margin_steps)
        if scheduler_class is DurationAwareAtomicScheduler
        else scheduler_class(env)
    )
    return env, target, scheduler


class UrgencyInterfaceTests(unittest.TestCase):
    def test_repaired_interface_is_exact_and_legacy_v1_is_unchanged(self):
        repaired = isolated_retrieval_env()[0]
        build_controller_options(
            repaired,
            DummySelector(repaired),
            controller_action_interface=URGENCY_FIRST_ACTION_INTERFACE,
        )
        repaired_manager = canonical_manager_options(repaired.options)
        repaired_retrievals = [
            option
            for option in repaired_manager
            if type(option) is StrictRetrieveDeliverOption
        ]

        legacy = isolated_retrieval_env()[0]
        build_controller_options(
            legacy,
            DummySelector(legacy),
            controller_action_interface=(
                ATOMIC_INBOUND_CONTROLLER_ACTION_INTERFACE
            ),
        )
        legacy_manager = canonical_manager_options(legacy.options)
        legacy_retrievals = [
            option
            for option in legacy_manager
            if type(option) is RetrieveDeliverOption
        ]

        self.assertEqual(len(repaired_manager), 42)
        self.assertEqual(len(repaired_retrievals), 40)
        self.assertEqual(len(legacy_retrievals), 40)
        self.assertEqual(
            controller_ids(repaired_manager), controller_ids(legacy_manager)
        )
        self.assertEqual(
            len(controller_ids(repaired_manager)),
            len(set(controller_ids(repaired_manager))),
        )


class FrozenSelectorPreviewTests(unittest.TestCase):
    def test_preview_rejects_trainable_selector(self):
        env, _, selector, _, _ = make_inbound_system()
        selector.set_learning_enabled(True)

        with self.assertRaisesRegex(RuntimeError, "requires a frozen"):
            selector.preview_assignment(env.blocks[0])

    def test_preview_is_repeatable_and_does_not_mutate_selector_or_env(self):
        env, _, selector, _, _ = make_inbound_system()
        block = env.blocks[0]
        source = selector.source
        before = {
            "time_step": env.time_steps,
            "agent": env.current_state,
            "storage_location": block.storage_location,
            "carrying": block.carrying,
            "selector_decisions": selector.decision_count,
            "selector_seconds": selector.assignment_seconds,
            "source_assignments": source.assignment_count,
            "pending": source.pending,
            "replay": len(source.replay),
            "rng": source.rng.getstate(),
            "store_events": list(env.store_events),
            "network_training": source.network.training,
        }

        first = selector.preview_assignment(
            block, source_cell=block.position
        )
        second = selector.preview_assignment(
            block, source_cell=block.position
        )

        self.assertEqual(first, second)
        self.assertEqual(env.time_steps, before["time_step"])
        self.assertEqual(env.current_state, before["agent"])
        self.assertEqual(block.storage_location, before["storage_location"])
        self.assertEqual(block.carrying, before["carrying"])
        self.assertEqual(selector.decision_count, before["selector_decisions"])
        self.assertEqual(
            selector.assignment_seconds, before["selector_seconds"]
        )
        self.assertEqual(source.assignment_count, before["source_assignments"])
        self.assertIs(source.pending, before["pending"])
        self.assertEqual(len(source.replay), before["replay"])
        self.assertEqual(source.rng.getstate(), before["rng"])
        self.assertEqual(env.store_events, before["store_events"])
        self.assertEqual(source.network.training, before["network_training"])

        chosen = selector.assign_block(block, source_cell=block.position)
        self.assertEqual(chosen, first.chosen_cell)
        self.assertEqual(selector.decision_count, 1)

    def test_duration_estimate_includes_pickup_and_putdown_without_commit(self):
        env, _, selector, _, _ = make_inbound_system()
        accept = next(
            option
            for option in env.options
            if isinstance(option, AcceptStoreOption)
        )
        block = env.blocks[0]

        estimate = accept.estimate_duration()

        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.block_label, block.label)
        self.assertEqual(
            estimate.total_steps,
            estimate.pickup_steps + estimate.storage_steps,
        )
        self.assertGreaterEqual(estimate.pickup_steps, 1)
        self.assertGreaterEqual(estimate.storage_steps, 1)
        self.assertIsNone(block.storage_location)
        self.assertEqual(selector.decision_count, 0)
        self.assertEqual(selector.source.assignment_count, 0)


class UrgencyPolicyTests(unittest.TestCase):
    def test_due_executable_retrieval_precedes_feasible_inbound(self):
        env, target, scheduler = make_policy_system(inbound=True, due=True)

        choice = scheduler.select_action(env.get_current_state())

        self.assertIsInstance(choice, StrictRetrieveDeliverOption)
        self.assertEqual(choice.target_label, target.label)
        self.assertEqual(scheduler.reason_counts["due_retrieval"], 1)
        self.assertEqual(scheduler.decisions[-1]["raw_due_count"], 1)

    def test_positive_slack_inbound_is_accepted(self):
        env, _, scheduler = make_policy_system(inbound=True, due=False)

        choice = scheduler.select_action(env.get_current_state())

        self.assertIsInstance(choice, AcceptStoreOption)
        self.assertEqual(scheduler.reason_counts["accept_inbound"], 1)

    def test_idle_positive_slack_state_defers_to_an_event(self):
        env, _, scheduler = make_policy_system(inbound=False, due=False)

        choice = scheduler.select_action(env.get_current_state())

        self.assertIsInstance(choice, StrategicDeferOption)
        self.assertEqual(scheduler.reason_counts["defer_to_event"], 1)

    def test_infeasible_accept_uses_executable_capacity_release(self):
        env, target, scheduler = make_policy_system(inbound=True, due=False)
        with patch.object(
            scheduler.accept_option, "initiation", return_value=False
        ):
            choice = scheduler.select_action(env.get_current_state())

        self.assertIsInstance(choice, StrictRetrieveDeliverOption)
        self.assertEqual(choice.target_label, target.label)
        self.assertEqual(
            scheduler.reason_counts["capacity_release_retrieval"], 1
        )

    def test_blocked_due_state_fails_instead_of_one_step_defer_loop(self):
        env, _, scheduler = make_policy_system(inbound=False, due=True)
        raw, _ = scheduler._retrieval_candidates(env.get_current_state())
        self.assertTrue(raw)
        with patch.object(
            scheduler, "_retrieval_candidates", return_value=(raw, [])
        ):
            with self.assertRaisesRegex(
                SchedulerInfeasibleError, "no due retrieval"
            ):
                scheduler.select_action(env.get_current_state())

        self.assertEqual(scheduler.reason_counts["no_action_infeasible"], 1)
        self.assertEqual(
            scheduler.reason_counts["blocked_due_infeasible"], 1
        )

    def test_no_inbound_and_no_executable_recovery_is_explicit(self):
        env, _, scheduler = make_policy_system(inbound=True, due=False)
        with patch.object(
            scheduler.accept_option, "initiation", return_value=False
        ), patch.object(
            scheduler, "_retrieval_candidates", return_value=([], [])
        ):
            with self.assertRaisesRegex(
                SchedulerInfeasibleError, "no strict retrieval"
            ):
                scheduler.select_action(env.get_current_state())


class DurationAwarePolicyTests(unittest.TestCase):
    @staticmethod
    def _select(current_slack, projected_slack, *, margin_steps=0.0):
        env, _, scheduler = make_policy_system(
            inbound=True,
            due=False,
            scheduler_class=DurationAwareAtomicScheduler,
            margin_steps=margin_steps,
        )
        raw, executable = scheduler._retrieval_candidates(
            env.get_current_state()
        )
        option, plan = executable[0]
        current = replace(
            plan,
            remaining_time=plan.estimated_steps + current_slack,
        )
        projected = replace(
            plan,
            remaining_time=plan.estimated_steps + projected_slack,
        )
        estimate = SimpleNamespace(
            contract="test_duration_estimate",
            block_label=env.blocks[1].label,
            time_step=env.time_steps,
            preview_storage_cell=env.storage_positions[-1],
            preview_candidate_count=1,
            pickup_steps=3,
            storage_steps=7,
            total_steps=10,
            preview_seconds=0.0,
        )
        candidates = ([(option, current)], [(option, current)])
        with patch.object(
            scheduler, "_retrieval_candidates", return_value=candidates
        ), patch.object(
            scheduler.accept_option,
            "estimate_duration",
            return_value=estimate,
        ), patch.object(
            scheduler, "_projected_plan", return_value=projected
        ):
            choice = scheduler.select_action(env.get_current_state())
        return scheduler, choice

    def test_retrieve_when_now_has_lower_absolute_timing_error(self):
        scheduler, choice = self._select(4.0, -7.0)

        self.assertIsInstance(choice, StrictRetrieveDeliverOption)
        self.assertEqual(scheduler.decisions[-1]["reason"], "lookahead_retrieval")
        self.assertEqual(
            scheduler.decisions[-1]["lookahead_cost_if_retrieve_now"], 4.0
        )
        self.assertEqual(
            scheduler.decisions[-1]["lookahead_cost_if_accept_first"], 7.0
        )

    def test_accept_when_projected_timing_error_is_lower(self):
        scheduler, choice = self._select(8.0, -2.0)

        self.assertIsInstance(choice, AcceptStoreOption)
        self.assertEqual(scheduler.decisions[-1]["reason"], "lookahead_accept")

    def test_exact_cost_tie_and_positive_margin_favor_retrieval(self):
        tied, tied_choice = self._select(4.0, -4.0)
        margin, margin_choice = self._select(
            5.0, -3.0, margin_steps=2.0
        )
        no_margin, no_margin_choice = self._select(5.0, -3.0)

        self.assertIsInstance(tied_choice, StrictRetrieveDeliverOption)
        self.assertEqual(tied.lookahead_tie_count, 1)
        self.assertIsInstance(margin_choice, StrictRetrieveDeliverOption)
        self.assertEqual(margin.lookahead_tie_count, 1)
        self.assertIsInstance(no_margin_choice, AcceptStoreOption)

    def test_due_retrieval_does_not_invoke_assignment_preview(self):
        env, _, scheduler = make_policy_system(
            inbound=True,
            due=True,
            scheduler_class=DurationAwareAtomicScheduler,
        )
        with patch.object(
            scheduler.accept_option,
            "estimate_duration",
            side_effect=AssertionError("due branch must not preview"),
        ):
            choice = scheduler.select_action(env.get_current_state())

        self.assertIsInstance(choice, StrictRetrieveDeliverOption)
        self.assertEqual(scheduler.decisions[-1]["reason"], "due_retrieval")
        self.assertEqual(scheduler.preview_call_count, 0)

    def test_preview_exception_becomes_audited_scheduler_failure(self):
        env, _, scheduler = make_policy_system(
            inbound=True,
            due=False,
            scheduler_class=DurationAwareAtomicScheduler,
        )
        with patch.object(
            scheduler.accept_option,
            "estimate_duration",
            side_effect=ValueError("malformed preview"),
        ):
            with self.assertRaisesRegex(
                SchedulerInfeasibleError, "malformed preview"
            ):
                scheduler.select_action(env.get_current_state())

        self.assertEqual(scheduler.preview_call_count, 1)
        self.assertEqual(scheduler.preview_failure_count, 1)
        self.assertEqual(
            scheduler.reason_counts["lookahead_preview_infeasible"], 1
        )


class UrgencyEvaluationContractTests(unittest.TestCase):
    FLOW_KEYS = {
        "storage_flow_metric_contract",
        "storage_flow_observation_end_step",
        "storage_flow_manifest_count",
        "storage_flow_arrived_count",
        "storage_flow_completed_count",
        "storage_flow_right_censored_count",
        "storage_flow_not_yet_arrived_count",
        "storage_flow_unfinished_count",
        "storage_flow_completion_rate",
        "storage_flow_arrived_completion_rate",
        "storage_flow_fully_observed",
        "mean_storage_flow_time",
        "median_storage_flow_time",
        "p90_storage_flow_time",
        "p95_storage_flow_time",
        "max_storage_flow_time",
        "completed_storage_flow_times",
        "right_censored_storage_flow_ages",
        "storage_flow_records",
    }

    def test_programmatic_evaluator_preserves_instance_and_digest(self):
        env, _ = isolated_retrieval_env()
        config = REGV5Config(
            block_embedding_dim=8,
            candidate_embedding_dim=8,
            context_dim=16,
            replay_size=4,
            batch_size=1,
            min_replay_size=1,
            updates_per_episode=1,
        )
        source = REGV5AssignmentSource(
            env, config, seed=17, learning_enabled=False, device="cpu"
        )
        payload = source.checkpoint(
            training_lambda=0.5,
            training_mu=50.0,
            completed_training_episodes=0,
        )
        evaluation_env = SmallRoomsEnv(
            choose_storage=False, arrival_rate=0.5, proc_mean=50.0
        )
        instance = evaluation_env.sample_episode_instance(17)
        args = SimpleNamespace(
            lam=0.5,
            mu=50.0,
            instance=None,
            device="cpu",
            max_steps=5,
            max_defer_steps=2,
            target_window=20.0,
            save_instances_dir=None,
        )

        run = evaluate_one(
            args, 17, payload, episode_instance=instance
        )

        self.assertEqual(run["method"], URGENCY_FIRST_METHOD)
        self.assertEqual(run["instance_id"], instance.instance_id)
        self.assertTrue(self.FLOW_KEYS.issubset(run))
        self.assertEqual(run["storage_flow_manifest_count"], 40)
        self.assertEqual(
            run["storage_flow_observation_end_step"], run["steps"]
        )
        self.assertEqual(len(run["storage_flow_records"]), 40)
        self.assertEqual(
            run["selector_deployment_digest"],
            selector_deployment_digest(payload),
        )
        self.assertEqual(
            run["urgency_scheduler_audit"][
                "retrieve_deliver_option_version"
            ],
            "retrieve_deliver_option_v3",
        )
        self.assertEqual(
            run["urgency_scheduler_audit"]["retrieval_executor_version"],
            "named_atomic_retrieval_executor_v3",
        )
        row = format_urgency_row(
            run, args, {"block_count": 40}, payload
        )
        self.assertEqual(set(row), set(FIELDNAMES))
        self.assertEqual(row["instance_id"], instance.instance_id)

        args.lookahead_margin_steps = 0.0
        lookahead = evaluate_duration_aware_one(
            args, 17, payload, episode_instance=instance
        )
        self.assertEqual(lookahead["method"], DURATION_AWARE_METHOD)
        self.assertEqual(lookahead["instance_id"], instance.instance_id)
        self.assertTrue(self.FLOW_KEYS.issubset(lookahead))
        self.assertEqual(
            lookahead["urgency_scheduler_audit"][
                "lookahead_cost_contract"
            ],
            "absolute_delivery_error_head_job_one_step_projection_v1",
        )
        scheduler_flow = lookahead["urgency_scheduler_audit"]
        self.assertTrue(self.FLOW_KEYS.issubset(scheduler_flow))
        self.assertEqual(
            scheduler_flow["arrival_to_storage_wait_steps"],
            lookahead["completed_storage_flow_times"],
        )
        self.assertIn(
            "deprecated_alias_of_storage_flow_metrics_v1",
            scheduler_flow["arrival_to_storage_wait_alias_contract"],
        )
        lookahead_row = format_urgency_row(
            lookahead, args, {"block_count": 40}, payload
        )
        self.assertEqual(set(lookahead_row), set(FIELDNAMES))
        self.assertEqual(lookahead_row["method"], DURATION_AWARE_METHOD)


if __name__ == "__main__":
    unittest.main()
