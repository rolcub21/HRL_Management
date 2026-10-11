from dataclasses import replace
from types import SimpleNamespace
import unittest

from compare_track_b_assignment_sources import (
    paired_comparisons,
    scheduler_invariants,
)
from example.Options.selector_v5 import (
    StrictStorageAssignmentOption,
    TRACK_B_ASSIGNMENT_SOURCES,
    make_track_b_assignment_option,
)
from example.helper.occupancy_pressure import (
    OCCUPANCY_PRESSURE_METRIC_CONTRACT,
)
from example.small_rooms_env import SmallRoomsEnv
from PSLAP.reg_selector_v5 import REGV5AssignmentSource, REGV5Config
from PSLAP.track_a import TRACK_A_REG_SELECTOR_V5
from track_b_urgency_evaluate import evaluate_assignment_ablation_one


def selector_payload():
    env = SmallRoomsEnv(
        choose_storage=False, arrival_rate=0.5, proc_mean=50.0
    )
    source = REGV5AssignmentSource(
        env,
        REGV5Config(
            block_embedding_dim=8,
            candidate_embedding_dim=8,
            context_dim=16,
            replay_size=4,
            batch_size=1,
            min_replay_size=1,
            updates_per_episode=1,
        ),
        seed=9,
        learning_enabled=False,
        device="cpu",
    )
    return source.checkpoint(
        training_lambda=0.5,
        training_mu=50.0,
        completed_training_episodes=0,
    )


def one_inbound_instance(seed=9):
    env = SmallRoomsEnv(
        choose_storage=False, arrival_rate=0.5, proc_mean=50.0
    )
    base = env.sample_episode_instance(seed)
    return replace(
        base,
        arrival_steps=(0,) + (10_000,) * (len(env.blocks) - 1),
    )


class MissingProposalSource:
    VERSION = "missing_proposal_test_v1"

    def __init__(self):
        self.propose_calls = 0
        self.preview_calls = 0

    def propose(self, yard, block, source, valid_candidates):
        self.propose_calls += 1
        return None

    def preview(self, yard, block, source, valid_candidates):
        self.preview_calls += 1
        return None


class AssignmentSourceAdapterTests(unittest.TestCase):
    def test_every_source_has_repeatable_noncommitting_preview_and_strict_commit(self):
        payload = selector_payload()
        instance = one_inbound_instance()
        for source_name in TRACK_B_ASSIGNMENT_SOURCES:
            with self.subTest(source=source_name):
                env = SmallRoomsEnv(
                    choose_storage=False,
                    arrival_rate=0.5,
                    proc_mean=50.0,
                )
                selector = make_track_b_assignment_option(
                    env,
                    source_name,
                    selector_payload=(
                        payload
                        if source_name == TRACK_A_REG_SELECTOR_V5
                        else None
                    ),
                    device="cpu",
                    seed=9,
                )
                env.reset(instance=instance)
                block = next(
                    item
                    for item in env.blocks
                    if item.position == env.pickup_cell
                    and not item.stored
                )
                before = (
                    env.time_steps,
                    env.current_state,
                    block.storage_location,
                    selector.decision_count,
                    selector.assignment_seconds,
                    list(env.store_events),
                )

                first = selector.preview_assignment(block)
                second = selector.preview_assignment(block)

                self.assertEqual(first, second)
                self.assertEqual(
                    (
                        env.time_steps,
                        env.current_state,
                        block.storage_location,
                        selector.decision_count,
                        selector.assignment_seconds,
                        list(env.store_events),
                    ),
                    before,
                )
                chosen = selector.assign_block(
                    block, source_cell=block.position
                )
                self.assertEqual(chosen, first.chosen_cell)
                audit = selector.audit()
                self.assertEqual(audit["assignment_source"], source_name)
                self.assertEqual(audit["decision_count"], 1)
                self.assertEqual(audit["valid_assignment_count"], 1)
                self.assertEqual(audit["invalid_assignment_count"], 0)
                self.assertEqual(audit["fallback_count"], 0)
                self.assertEqual(
                    len(selector.decisions[0]["candidate_mask_id"]), 16
                )
                self.assertEqual(
                    selector.decisions[0]["candidate_mask_id"],
                    first.candidate_mask_id,
                )
                self.assertEqual(
                    selector.decisions[0]["chosen_cell"], first.chosen_cell
                )

    def test_missing_proposal_is_failure_without_fallback(self):
        env = SmallRoomsEnv(
            choose_storage=False, arrival_rate=0.5, proc_mean=50.0
        )
        source = MissingProposalSource()
        selector = StrictStorageAssignmentOption(
            env,
            source,
            assignment_source="missing_test",
            assignment_source_version=source.VERSION,
            preview_contract="missing_test_preview_v1",
            assignment_contract="missing_test_assignment_v1",
        )
        env.reset(instance=one_inbound_instance())
        block = next(
            item
            for item in env.blocks
            if item.position == env.pickup_cell and not item.stored
        )

        with self.assertRaisesRegex(RuntimeError, "no valid shared-mask"):
            selector.preview_assignment(block)
        self.assertIsNone(
            selector.assign_block(block, source_cell=block.position)
        )

        audit = selector.audit()
        self.assertEqual(source.preview_calls, 1)
        self.assertEqual(source.propose_calls, 1)
        self.assertEqual(audit["invalid_assignment_count"], 1)
        self.assertEqual(audit["fallback_count"], 0)
        self.assertEqual(selector.decisions[0]["reason"], "no_proposal")


class AssignmentSourceEvaluationTests(unittest.TestCase):
    def test_same_instance_and_scheduler_contract_are_used_for_all_sources(self):
        payload = selector_payload()
        instance = one_inbound_instance(seed=11)
        args = SimpleNamespace(
            lam=0.5,
            mu=50.0,
            instance=None,
            device="cpu",
            max_steps=20,
            max_defer_steps=10,
            lookahead_margin_steps=2.0,
            target_window=20.0,
            save_instances_dir=None,
        )
        runs = [
            evaluate_assignment_ablation_one(
                args,
                11,
                payload,
                assignment_source=source,
                episode_instance=instance,
            )
            for source in TRACK_B_ASSIGNMENT_SOURCES
        ]

        self.assertEqual(
            {run["instance_id"] for run in runs}, {instance.instance_id}
        )
        self.assertEqual(
            {run["assignment_source"] for run in runs},
            set(TRACK_B_ASSIGNMENT_SOURCES),
        )
        contracts = [scheduler_invariants(run) for run in runs]
        self.assertTrue(all(item == contracts[0] for item in contracts[1:]))
        self.assertEqual(contracts[0]["lookahead_margin_steps"], 2.0)
        self.assertEqual(
            contracts[0]["retrieval_executor_version"],
            "named_atomic_retrieval_executor_v3",
        )
        self.assertTrue(
            all(run["selector_audit"]["fallback_count"] == 0 for run in runs)
        )
        self.assertTrue(
            all(
                run["occupancy_pressure_metric_contract"]
                == OCCUPANCY_PRESSURE_METRIC_CONTRACT
                for run in runs
            )
        )
        self.assertTrue(
            all(
                run["storage_pressure_observed_steps"] == run["steps"]
                for run in runs
            )
        )
        self.assertTrue(
            all(run["storage_capacity_cells"] == 63 for run in runs)
        )
        self.assertTrue(
            all(run["minimum_live_candidate_count"] is None for run in runs)
        )
        self.assertTrue(
            all(
                run["minimum_live_accept_candidate_count"] is not None
                for run in runs
            )
        )

        comparisons = paired_comparisons(
            runs,
            TRACK_B_ASSIGNMENT_SOURCES,
            TRACK_A_REG_SELECTOR_V5,
            10,
        )
        self.assertEqual(len(comparisons), len(TRACK_B_ASSIGNMENT_SOURCES) - 1)
        self.assertTrue(all(item["n"] == 1 for item in comparisons))


if __name__ == "__main__":
    unittest.main()
