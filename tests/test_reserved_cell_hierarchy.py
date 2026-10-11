from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from example.Options.AcceptStoreOption import (
    AcceptStoreOption,
    ReservedAcceptStoreOption,
)
from example.Options.RetrieveDeliverOption import StrictRetrieveDeliverOption
from example.Options.selector_v5 import StorageSelectOptionV5
from example.controller_options import (
    build_controller_options,
    scheduler_episode_audit,
)
from example.small_rooms_env import SmallRoomsEnv
from example.urgency_scheduler import (
    RESERVED_CELL_ACTION_INTERFACE,
    ReservedCellDurationAwareAtomicScheduler,
)
from options_agent import canonical_manager_options, controller_ids
from PSLAP.reg_selector_v5 import REGV5AssignmentSource, REGV5Config
from PSLAP.retrieval_context import retrieval_planning_context
from tests.test_scheduler_options import isolated_retrieval_env
from track_b_urgency_evaluate import (
    DECISION_EPOCH_RESERVED,
    evaluate_assignment_ablation_one,
)
from PSLAP.track_a import TRACK_A_REG_SELECTOR_V5


SEED = 812
LEGACY_ACCEPT_ID = "option:AcceptStore:v1"
RESERVED_ACCEPT_ID = "option:AcceptStore:v2"


def _config():
    return REGV5Config(
        block_embedding_dim=8,
        candidate_embedding_dim=8,
        context_dim=16,
        replay_size=4,
        batch_size=1,
        min_replay_size=1,
        updates_per_episode=1,
    )


def _one_inbound_instance(env, seed=SEED):
    base = env.sample_episode_instance(seed)
    return replace(
        base,
        arrival_steps=(0,) + (10_000,) * (len(env.blocks) - 1),
    )


def _selector(env, seed=SEED):
    source = REGV5AssignmentSource(
        env,
        _config(),
        seed=seed,
        learning_enabled=False,
        device="cpu",
    )
    return StorageSelectOptionV5(env, source)


def _reserved_system(seed=SEED):
    env = SmallRoomsEnv(
        choose_storage=False,
        arrival_rate=0.5,
        proc_mean=50.0,
    )
    instance = _one_inbound_instance(env, seed)
    selector = _selector(env, seed)
    build_controller_options(
        env,
        selector,
        controller_action_interface=RESERVED_CELL_ACTION_INTERFACE,
        max_defer_steps=10,
    )
    state = env.reset(instance=instance)
    scheduler = ReservedCellDurationAwareAtomicScheduler(
        env, margin_steps=2.0
    )
    return env, state, instance, selector, scheduler


def _execute_macro(env, state, option, limit=200):
    actions = []
    infos = []
    for _ in range(limit):
        action = option.policy(state)
        next_state, _, _, info = env.step(action)
        terminated = option.termination(next_state)
        actions.append(action)
        infos.append(info)
        state = next_state
        if terminated:
            return state, actions, infos
    raise AssertionError("reserved AcceptStore did not terminate")


class ReservedCellExecutionTests(unittest.TestCase):
    def test_no_retrieval_branch_binds_and_commits_exactly_one_model_proposal(self):
        env, state, _, selector, scheduler = _reserved_system()
        source = selector.source

        with patch.object(
            source, "preview", wraps=source.preview
        ) as preview_call, patch.object(
            source, "propose", wraps=source.propose
        ) as propose_call, patch.object(
            source.network, "forward", wraps=source.network.forward
        ) as forward_call:
            option = scheduler.select_action(state)
            self.assertIsInstance(option, ReservedAcceptStoreOption)
            self.assertEqual(
                scheduler.decisions[-1]["reason"],
                "lookahead_accept_no_retrieval",
            )
            reservation = option.active_reservation
            self.assertIsNotNone(reservation)
            proposal = reservation.preview_proposal
            self.assertEqual(
                scheduler.decisions[-1]["preview_proposal_id"],
                proposal.proposal_id,
            )
            self.assertIsNone(env.blocks[0].storage_location)

            _, _, infos = _execute_macro(env, state, option)

        self.assertEqual(preview_call.call_count, 1)
        self.assertEqual(forward_call.call_count, 1)
        self.assertEqual(propose_call.call_count, 0)
        self.assertEqual(selector.decision_count, 1)
        self.assertEqual(len(selector.decisions), 1)
        decision = selector.decisions[0]
        self.assertTrue(decision["valid"])
        self.assertEqual(decision["chosen_cell"], proposal.chosen_cell)
        self.assertEqual(decision["candidate_mask_id"], proposal.candidate_mask_id)
        self.assertEqual(decision["proposal_id"], proposal.proposal_id)
        self.assertEqual(
            decision["commitment_contract"],
            ReservedAcceptStoreOption.COMMITMENT_CONTRACT,
        )
        self.assertEqual(env.blocks[0].storage_location, proposal.chosen_cell)
        self.assertEqual(
            [
                event["proposal_id"]
                for event in env.store_events
                if event.get("phase") == "reserved_chosen"
            ],
            [proposal.proposal_id],
        )
        self.assertEqual(
            [info["stored_block"] for info in infos if info.get("stored_block")],
            [env.blocks[0].label],
        )
        self.assertIsNone(option.active_reservation)
        self.assertEqual(option.episode_reservation_bound_count, 1)
        self.assertEqual(option.episode_reservation_commit_count, 1)
        self.assertEqual(option.episode_reservation_execution_match_count, 1)
        self.assertEqual(option.episode_reservation_invalidation_count, 0)
        self.assertEqual(option.last_outcome["proposal_id"], proposal.proposal_id)
        self.assertEqual(
            option.last_outcome["reserved_storage_cell"], proposal.chosen_cell
        )
        self.assertTrue(option.last_outcome["reservation_execution_match"])

        scheduler_audit = scheduler.audit()
        option_audit = scheduler_episode_audit(env)
        for audit in (scheduler_audit, option_audit):
            self.assertEqual(audit["reservation_bound_count"], 1)
            self.assertEqual(audit["reservation_commit_count"], 1)
            self.assertEqual(audit["reservation_invalidation_count"], 0)
            self.assertEqual(audit["bound_proposal_ids"], [proposal.proposal_id])
            self.assertEqual(
                audit["committed_proposal_ids"], [proposal.proposal_id]
            )

    def test_due_retrieval_bypasses_preview_and_reservation(self):
        env, target = isolated_retrieval_env()
        inbound = env.blocks[1]
        inbound.position = env.pickup_cell
        inbound.storage_location = None
        inbound.carrying = False
        inbound.stored = False
        inbound.delivered = False
        inbound.arrival_step = 0

        selector = _selector(env)
        build_controller_options(
            env,
            selector,
            controller_action_interface=RESERVED_CELL_ACTION_INTERFACE,
            max_defer_steps=10,
        )
        scheduler = ReservedCellDurationAwareAtomicScheduler(env)
        plan = retrieval_planning_context(env).plan(target.label)
        self.assertIsNotNone(plan)
        target.storage_steps_needed = max(1, plan.estimated_steps - 1)

        with patch.object(
            selector.source, "preview", wraps=selector.source.preview
        ) as preview_call, patch.object(
            selector.source, "propose", wraps=selector.source.propose
        ) as propose_call:
            choice = scheduler.select_action(env.get_current_state())

        self.assertIsInstance(choice, StrictRetrieveDeliverOption)
        self.assertEqual(choice.target_label, target.label)
        self.assertEqual(scheduler.decisions[-1]["reason"], "due_retrieval")
        self.assertEqual(preview_call.call_count, 0)
        self.assertEqual(propose_call.call_count, 0)
        self.assertIsNone(scheduler.accept_option.active_reservation)
        self.assertEqual(scheduler.preview_call_count, 0)


class ReservedCellValidationTests(unittest.TestCase):
    def test_tampered_proposal_id_is_rejected_before_binding(self):
        _, _, _, selector, scheduler = _reserved_system()
        option = scheduler.accept_option
        estimate = option.estimate_duration()
        tampered_preview = replace(
            estimate.preview_proposal,
            proposal_id="0" * len(estimate.preview_proposal.proposal_id),
        )
        tampered = replace(estimate, preview_proposal=tampered_preview)

        with self.assertRaisesRegex(ValueError, "proposal_id_mismatch"):
            option.bind_estimate(tampered)

        self.assertIsNone(option.active_reservation)
        self.assertEqual(option.episode_reservation_bound_count, 0)
        self.assertEqual(selector.decision_count, 0)

    def test_bound_proposal_that_is_not_started_at_its_epoch_fails_explicitly(self):
        env, state, _, selector, scheduler = _reserved_system()
        option = scheduler.select_action(state)
        proposal_id = option.active_reservation.preview_proposal.proposal_id
        env.time_steps += 1

        action = option.policy(state)

        self.assertEqual(action, env.ACTION_IDS["WAIT"])
        self.assertTrue(option.termination(state))
        self.assertEqual(
            option.last_outcome["reason"],
            "reserved_assignment_not_started_at_bound_epoch",
        )
        self.assertEqual(option.last_outcome["proposal_id"], proposal_id)
        self.assertEqual(option.episode_reservation_invalidation_count, 1)
        self.assertEqual(option.episode_reservation_commit_count, 0)
        self.assertEqual(selector.decision_count, 0)
        self.assertEqual(selector.source.assignment_count, 0)
        self.assertIsNone(option.active_reservation)

    def test_reset_clears_reservation_and_rejects_previous_instance_token(self):
        env, state, instance, selector, scheduler = _reserved_system()
        option = scheduler.select_action(state)
        old_estimate = option.active_reservation
        old_id = old_estimate.preview_proposal.proposal_id
        next_instance = replace(instance, seed=instance.seed + 1)

        env.reset(instance=next_instance)

        self.assertIsNone(option.active_reservation)
        self.assertEqual(option.episode_reservation_bound_count, 0)
        self.assertEqual(option.episode_reservation_commit_count, 0)
        self.assertEqual(option.episode_bound_proposal_ids, [])
        self.assertEqual(option.episode_committed_proposal_ids, [])
        self.assertEqual(selector.decision_count, 0)
        with self.assertRaisesRegex(ValueError, "preview_instance_mismatch"):
            option.bind_estimate(old_estimate)
        self.assertNotEqual(
            old_id,
            option.estimate_duration().preview_proposal.proposal_id,
        )


class ReservedCellIdentityAndEvaluationTests(unittest.TestCase):
    def test_v1_and_v2_have_separate_stable_controller_identities(self):
        legacy_env = SmallRoomsEnv(
            choose_storage=False, arrival_rate=0.5, proc_mean=50.0
        )
        legacy_selector = _selector(legacy_env)
        build_controller_options(
            legacy_env,
            legacy_selector,
            controller_action_interface="interleaved_atomic_scheduler_v3",
        )
        reserved_env = SmallRoomsEnv(
            choose_storage=False, arrival_rate=0.5, proc_mean=50.0
        )
        reserved_selector = _selector(reserved_env)
        build_controller_options(
            reserved_env,
            reserved_selector,
            controller_action_interface=RESERVED_CELL_ACTION_INTERFACE,
        )

        legacy_ids = controller_ids(canonical_manager_options(legacy_env.options))
        reserved_ids = controller_ids(
            canonical_manager_options(reserved_env.options)
        )
        self.assertEqual(len(legacy_ids), 42)
        self.assertEqual(len(reserved_ids), 42)
        self.assertIn(LEGACY_ACCEPT_ID, legacy_ids)
        self.assertNotIn(RESERVED_ACCEPT_ID, legacy_ids)
        self.assertIn(RESERVED_ACCEPT_ID, reserved_ids)
        self.assertNotIn(LEGACY_ACCEPT_ID, reserved_ids)
        self.assertEqual(
            set(legacy_ids) - {LEGACY_ACCEPT_ID},
            set(reserved_ids) - {RESERVED_ACCEPT_ID},
        )
        legacy_accept = next(
            option
            for option in legacy_env.options
            if type(option) is AcceptStoreOption
        )
        reserved_accept = next(
            option
            for option in reserved_env.options
            if type(option) is ReservedAcceptStoreOption
        )
        self.assertEqual(legacy_accept.VERSION, "accept_store_option_v1")
        self.assertEqual(
            reserved_accept.VERSION, "accept_store_reserved_cell_v2"
        )

    def test_evaluator_joins_scheduler_selector_and_option_provenance(self):
        fixture_env = SmallRoomsEnv(
            choose_storage=False, arrival_rate=0.5, proc_mean=50.0
        )
        instance = _one_inbound_instance(fixture_env, seed=817)
        fixture_selector = _selector(fixture_env, seed=817)
        payload = fixture_selector.source.checkpoint(
            training_lambda=0.5,
            training_mu=50.0,
            completed_training_episodes=0,
        )
        args = SimpleNamespace(
            lam=0.5,
            mu=50.0,
            instance=None,
            device="cpu",
            max_steps=200,
            max_defer_steps=10,
            lookahead_margin_steps=2.0,
            assignment_commitment=DECISION_EPOCH_RESERVED,
            target_window=20.0,
            save_instances_dir=None,
        )

        run = evaluate_assignment_ablation_one(
            args,
            817,
            payload,
            assignment_source=TRACK_A_REG_SELECTOR_V5,
            episode_instance=instance,
        )

        scheduler_audit = run["urgency_scheduler_audit"]
        option_audit = run["scheduler_audit"]
        selector_audit = run["selector_audit"]
        accept_decision = next(
            decision
            for decision in scheduler_audit["decisions"]
            if decision.get("preview_proposal_id") is not None
        )
        proposal_id = accept_decision["preview_proposal_id"]

        self.assertEqual(run["assignment_commitment"], DECISION_EPOCH_RESERVED)
        self.assertEqual(
            run["controller_action_interface"], RESERVED_CELL_ACTION_INTERFACE
        )
        self.assertTrue(run["reservation_integrity"])
        self.assertEqual(option_audit["reservation_bound_count"], 1)
        self.assertEqual(option_audit["reservation_commit_count"], 1)
        self.assertEqual(option_audit["reservation_execution_match_count"], 1)
        self.assertEqual(option_audit["reservation_invalidation_count"], 0)
        self.assertEqual(option_audit["bound_proposal_ids"], [proposal_id])
        self.assertEqual(option_audit["committed_proposal_ids"], [proposal_id])
        self.assertEqual(scheduler_audit["bound_proposal_ids"], [proposal_id])
        self.assertEqual(
            scheduler_audit["committed_proposal_ids"], [proposal_id]
        )
        self.assertEqual(selector_audit["decision_count"], 1)
        self.assertEqual(selector_audit["decisions"][0]["proposal_id"], proposal_id)
        self.assertEqual(
            option_audit["inbound_outcomes"][0]["proposal_id"], proposal_id
        )
        self.assertEqual(
            selector_audit["decisions"][0]["chosen_cell"],
            accept_decision["preview_storage_cell"],
        )
        self.assertEqual(
            scheduler_audit["assignment_commitment_contract"],
            ReservedAcceptStoreOption.COMMITMENT_CONTRACT,
        )
        self.assertEqual(
            scheduler_audit["reservation_contract"],
            ReservedAcceptStoreOption.RESERVATION_CONTRACT,
        )


if __name__ == "__main__":
    unittest.main()
