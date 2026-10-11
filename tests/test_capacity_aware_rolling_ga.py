import hashlib
import json
from types import SimpleNamespace
import unittest

import numpy as np

from PSLAP.dynamic_yard import BlockView, YardSnapshot
from PSLAP.ga_capacity_aware import (
    CAPACITY_AWARE_ASSIGNER_CLASSES,
    CAPACITY_AWARE_QUEUE_CONTRACT,
    CapacityAwareCompleteRollingGAAssigner,
    CapacityAwareDurationAwareRollingGAAssigner,
    CapacityAwareOperationalRollingGAAssigner,
    CapacityAwareRollingGAStorageAssigner,
)
from PSLAP.ga_optimizer import GAConfig
from PSLAP.neutral_protocol import shared_candidate_mask
from PSLAP.ga_policy import (
    CompleteSingletonDurationAwareRollingGAAssigner,
    DurationAwareRollingGAStorageAssigner,
    OperationalRollingGAStorageAssigner,
    RollingGAStorageAssigner,
)
from example.Options.selector_v5 import StrictStorageAssignmentOption
from example.Options.selector_v5 import (
    TRACK_B_ASSIGNMENT_SOURCES,
    TRACK_B_RESERVED_CAPACITY_AWARE_ASSIGNMENT_SOURCES,
    make_track_b_assignment_option,
)
from PSLAP.track_a import (
    TRACK_A_GA_ROLLING,
    TRACK_A_GA_ROLLING_CAPACITY_AWARE,
    TRACK_A_GA_ROLLING_COMPLETE_CAPACITY_AWARE,
    TRACK_A_GA_ROLLING_DURATION_AWARE_CAPACITY_AWARE,
    TRACK_A_GA_ROLLING_OPERATIONAL_CAPACITY_AWARE,
)


class _Block:
    def __init__(
        self,
        label,
        position,
        *,
        arrival_step,
        remaining=10.0,
        stored=False,
        delivered=False,
        carrying=False,
        storage_location=None,
    ):
        self.label = label
        self.position = position
        self.arrival_step = arrival_step
        self.remaining = remaining
        self.stored = stored
        self.delivered = delivered
        self.carrying = carrying
        self.storage_location = storage_location

    def get_remaining_storage_time(self):
        return self.remaining


class _UnreadFutureBlock(_Block):
    @property
    def arrival_step(self):
        raise AssertionError("unarrived future arrival was read")

    @arrival_step.setter
    def arrival_step(self, _value):
        pass

    def get_remaining_storage_time(self):
        raise AssertionError("unarrived future duration was read")


class _Env:
    pickup_cell = (0, 0)
    waiting_cell = (0, 1)
    grid_rows = 4
    grid_cols = 4
    exit_cells = ((3, 3),)
    storage_positions = frozenset({(1, 1), (1, 2), (2, 1), (2, 2)})

    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.time_steps = 5
        self.rooms = np.full((self.grid_rows, self.grid_cols), ".", dtype="U1")
        self.current_episode_instance = SimpleNamespace(instance_id="unit-instance")
        self.current_episode = 1
        self.store_events = []


def _make_env(pending_count=3, *, future=False):
    blocks = [
        _Block("b0", _Env.pickup_cell, arrival_step=0, remaining=8.0)
    ]
    blocks.extend(
        _Block(
            f"b{index}",
            _Env.waiting_cell,
            arrival_step=index,
            remaining=8.0 + index,
        )
        for index in range(1, pending_count)
    )
    if future:
        blocks.append(
            _UnreadFutureBlock("future", None, arrival_step=999, remaining=999)
        )
    return _Env(blocks)


def _view(env):
    block = env.blocks[0]
    return BlockView(block.label, block.position, block.get_remaining_storage_time())


def _yard(env, candidates=None):
    yard = YardSnapshot.from_env(env)
    if candidates is None:
        return yard
    return YardSnapshot(
        rows=yard.rows,
        cols=yard.cols,
        traversable=yard.traversable,
        storage_cells=frozenset(candidates),
        exits=yard.exits,
        blocks=yard.blocks,
    )


def _mask_id(candidates):
    return hashlib.sha256(
        json.dumps(tuple(candidates), separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _config():
    return GAConfig(
        population_size=6,
        generations=2,
        elite_count=1,
        tournament_size=2,
        seed=41,
    )


def _commit_source(source, chosen, candidates, *, proposal_id="unit-proposal"):
    metadata = {
        "proposal_id": proposal_id,
        "block_label": "b0",
        "chosen_cell": chosen,
        "selection_time_step": 5,
        "candidate_count": len(candidates),
        "candidate_mask_id": _mask_id(candidates),
    }
    source.on_preview_token_minted(**metadata)
    source.on_preview_reserved(**metadata)
    commit = {**metadata, "commit_time_step": 7}
    source.validate_bound_assignment_commit(**commit)
    source.on_bound_assignment_committed(**commit)


class CapacityAwareQueuedRollingGATests(unittest.TestCase):
    def test_new_ids_are_reserved_only_and_old_factory_identity_is_unchanged(self):
        self.assertIn(TRACK_A_GA_ROLLING, TRACK_B_ASSIGNMENT_SOURCES)
        self.assertNotIn(
            TRACK_A_GA_ROLLING_CAPACITY_AWARE, TRACK_B_ASSIGNMENT_SOURCES
        )
        self.assertEqual(
            TRACK_B_RESERVED_CAPACITY_AWARE_ASSIGNMENT_SOURCES,
            (
                TRACK_A_GA_ROLLING_CAPACITY_AWARE,
                TRACK_A_GA_ROLLING_DURATION_AWARE_CAPACITY_AWARE,
                TRACK_A_GA_ROLLING_OPERATIONAL_CAPACITY_AWARE,
                TRACK_A_GA_ROLLING_COMPLETE_CAPACITY_AWARE,
            ),
        )
        env = _make_env(1)
        historical = make_track_b_assignment_option(
            env, TRACK_A_GA_ROLLING, rolling_ga_config=_config()
        )
        repaired = make_track_b_assignment_option(
            env,
            TRACK_A_GA_ROLLING_CAPACITY_AWARE,
            rolling_ga_config=_config(),
        )
        self.assertIsInstance(historical.source, RollingGAStorageAssigner)
        self.assertNotIsInstance(
            historical.source, CapacityAwareRollingGAStorageAssigner
        )
        self.assertIsInstance(
            repaired.source, CapacityAwareRollingGAStorageAssigner
        )
        self.assertFalse(hasattr(historical.source, "on_preview_reserved"))

    def test_pending_above_capacity_is_explicit_overflow_for_all_variants(self):
        candidates = ((1, 1), (1, 2))
        for assigner_type in CAPACITY_AWARE_ASSIGNER_CLASSES:
            with self.subTest(assigner=assigner_type.__name__):
                env = _make_env(3)
                source = assigner_type(env, _config())
                chosen = source.preview(
                    _yard(env, candidates), _view(env), env.pickup_cell, candidates
                )
                self.assertIn(chosen, candidates)
                self.assertEqual(source.decision_count, 0)
                self.assertEqual(source.last_known_labels, ("b0", "b1", "b2"))
                self.assertEqual(source.last_planned_labels, ("b0", "b1"))
                self.assertEqual(source.last_deferred_labels, ("b2",))
                _commit_source(source, chosen, candidates)
                audit = source.audit()
                self.assertEqual(audit["queue_contract"], CAPACITY_AWARE_QUEUE_CONTRACT)
                self.assertEqual(audit["decision_count"], 1)
                self.assertEqual(audit["deferred_membership_count_total"], 1)
                self.assertTrue(audit["all_pending_partitions_exact"])
                self.assertTrue(audit["all_capacity_limits_respected"])
                self.assertTrue(audit["all_current_blocks_are_gene_zero"])
                self.assertTrue(audit["only_gene_zero_ever_committed"])
                self.assertEqual(audit["committed_gene_indices"], (0,))
                record = audit["decisions"][0]
                self.assertEqual(record["queue_overflow_labels"], ("b2",))
                self.assertFalse(record["scheduler_defer_action_introduced"])
                self.assertEqual(record["queue_overflow_virtual_penalty"], 0.0)
                self.assertEqual(
                    tuple(record["best_cost_components"]),
                    record["objective_lexicographic_fields"],
                )
                self.assertEqual(
                    record["best_cost_scalar"],
                    source.last_result.best_cost.scalar,
                )

    def test_one_candidate_still_serves_current_and_audits_full_overflow(self):
        candidates = ((1, 1),)
        for assigner_type in CAPACITY_AWARE_ASSIGNER_CLASSES:
            with self.subTest(assigner=assigner_type.__name__):
                env = _make_env(5)
                source = assigner_type(env, _config())
                chosen = source.preview(
                    _yard(env, candidates), _view(env), env.pickup_cell, candidates
                )
                self.assertEqual(chosen, candidates[0])
                self.assertEqual(source.last_planned_labels, ("b0",))
                self.assertEqual(
                    source.last_deferred_labels, ("b1", "b2", "b3", "b4")
                )
                _commit_source(source, chosen, candidates)
                record = source.audit()["decisions"][0]
                self.assertEqual(record["best_assignment"], (candidates[0],))
                self.assertEqual(
                    record["optimizer_mode"],
                    "exhaustive_singleton"
                    if assigner_type is CapacityAwareCompleteRollingGAAssigner
                    else "genetic_search",
                )

    def test_current_must_be_unique_fifo_pickup_head(self):
        candidates = ((1, 1),)
        env = _make_env(2)
        env.blocks[0].position = env.waiting_cell
        source = CapacityAwareRollingGAStorageAssigner(env, _config())
        with self.assertRaisesRegex(ValueError, "unique observed pickup head"):
            source.preview(_yard(env, candidates), _view(env), env.waiting_cell, candidates)

        env = _make_env(2)
        env.blocks[1].arrival_step = -1
        source = CapacityAwareRollingGAStorageAssigner(env, _config())
        with self.assertRaisesRegex(ValueError, "not the observed FIFO"):
            source.preview(_yard(env, candidates), _view(env), env.pickup_cell, candidates)

    def test_equal_arrival_fifo_order_is_stable_environment_manifest_order(self):
        candidates = ((1, 1), (1, 2), (2, 1))
        env = _make_env(3)
        for block in env.blocks:
            block.arrival_step = 0
        source = CapacityAwareRollingGAStorageAssigner(env, _config())
        source.preview(
            _yard(env, candidates), _view(env), env.pickup_cell, candidates
        )
        self.assertEqual(source.last_known_labels, ("b0", "b1", "b2"))
        self.assertEqual(source.last_planned_labels, ("b0", "b1", "b2"))

    def test_nonpending_statuses_are_excluded_and_explicit_mask_must_be_exact(self):
        env = _make_env(6)
        env.blocks[2].carrying = True
        env.blocks[3].stored = True
        env.blocks[4].delivered = True
        env.blocks[5].storage_location = (2, 2)
        yard = _yard(env)
        expected = shared_candidate_mask(
            yard, _view(env), env.pickup_cell
        )
        source = CapacityAwareRollingGAStorageAssigner(env, _config())
        source.preview(yard, _view(env), env.pickup_cell, expected)
        self.assertEqual(source.last_known_labels, ("b0", "b1"))
        with self.assertRaisesRegex(ValueError, "exact ordered"):
            source.preview(
                yard,
                _view(env),
                env.pickup_cell,
                tuple(reversed(expected)),
            )

    def test_identical_sources_are_deterministic_and_commit_seed_advances_once(self):
        env_a = _make_env(3)
        env_b = _make_env(3)
        yard_a = _yard(env_a)
        yard_b = _yard(env_b)
        candidates_a = shared_candidate_mask(
            yard_a, _view(env_a), env_a.pickup_cell
        )
        candidates_b = shared_candidate_mask(
            yard_b, _view(env_b), env_b.pickup_cell
        )
        source_a = CapacityAwareRollingGAStorageAssigner(env_a, _config())
        source_b = CapacityAwareRollingGAStorageAssigner(env_b, _config())
        chosen_a = source_a.preview(
            yard_a, _view(env_a), env_a.pickup_cell, candidates_a
        )
        chosen_b = source_b.preview(
            yard_b, _view(env_b), env_b.pickup_cell, candidates_b
        )
        self.assertEqual(chosen_a, chosen_b)
        self.assertEqual(source_a.last_result, source_b.last_result)
        _commit_source(source_a, chosen_a, candidates_a)
        self.assertEqual(source_a.decision_count, 1)
        source_a.preview(
            yard_a, _view(env_a), env_a.pickup_cell, candidates_a
        )
        self.assertEqual(
            source_a._unbound_preview_plan["optimizer_seed"],
            _config().seed + 1,
        )

    def test_unarrived_future_state_is_not_read(self):
        candidates = ((1, 1), (1, 2))
        env = _make_env(2, future=True)
        source = CapacityAwareRollingGAStorageAssigner(env, _config())
        chosen = source.preview(
            _yard(env, candidates), _view(env), env.pickup_cell, candidates
        )
        self.assertIn(chosen, candidates)
        self.assertEqual(source.last_known_labels, ("b0", "b1"))

    def test_overflow_duration_does_not_change_current_proposal(self):
        candidates = ((1, 1),)
        env_a = _make_env(3)
        env_b = _make_env(3)
        env_b.blocks[1].remaining = 5000.0
        env_b.blocks[2].remaining = -5000.0
        source_a = CapacityAwareDurationAwareRollingGAAssigner(env_a, _config())
        source_b = CapacityAwareDurationAwareRollingGAAssigner(env_b, _config())
        self.assertEqual(
            source_a.preview(_yard(env_a, candidates), _view(env_a), env_a.pickup_cell, candidates),
            source_b.preview(_yard(env_b, candidates), _view(env_b), env_b.pickup_cell, candidates),
        )

    def test_no_overflow_matches_historical_objective_for_all_variants(self):
        candidates = ((1, 1), (1, 2))
        pairs = (
            (RollingGAStorageAssigner, CapacityAwareRollingGAStorageAssigner),
            (
                DurationAwareRollingGAStorageAssigner,
                CapacityAwareDurationAwareRollingGAAssigner,
            ),
            (
                OperationalRollingGAStorageAssigner,
                CapacityAwareOperationalRollingGAAssigner,
            ),
            (
                CompleteSingletonDurationAwareRollingGAAssigner,
                CapacityAwareCompleteRollingGAAssigner,
            ),
        )
        for historical_type, repaired_type in pairs:
            with self.subTest(repaired=repaired_type.__name__):
                old_env = _make_env(2)
                new_env = _make_env(2)
                old = historical_type(old_env, _config())
                new = repaired_type(new_env, _config())
                old_result, old_labels, old_choice = old._optimize(
                    _yard(old_env, candidates),
                    _view(old_env),
                    old_env.pickup_cell,
                    candidates,
                )
                new_result, new_record, new_choice = new._optimize_capacity_aware(
                    _yard(new_env, candidates),
                    _view(new_env),
                    new_env.pickup_cell,
                    candidates,
                )
                self.assertEqual(old_choice, new_choice)
                self.assertEqual(old_labels, new_record["planned_labels"])
                self.assertEqual(old_result, new_result)
                self.assertEqual(new_record["deferred_labels"], ())

    def test_repeated_discarded_preview_is_idempotent_and_resettable(self):
        candidates = ((1, 1), (1, 2))
        env = _make_env(3)
        source = CapacityAwareRollingGAStorageAssigner(env, _config())
        first = source.preview(_yard(env, candidates), _view(env), env.pickup_cell, candidates)
        _commit_metadata = {
            "proposal_id": "discarded-lookahead",
            "block_label": "b0",
            "chosen_cell": first,
            "selection_time_step": 5,
            "candidate_count": len(candidates),
            "candidate_mask_id": _mask_id(candidates),
        }
        source.on_preview_token_minted(**_commit_metadata)
        second = source.preview(_yard(env, candidates), _view(env), env.pickup_cell, candidates)
        source.on_preview_token_minted(**_commit_metadata)
        self.assertEqual(first, second)
        self.assertEqual(source.decision_count, 0)
        self.assertEqual(source.audit()["minted_unreserved_preview_count"], 1)
        source.on_episode_start()
        audit = source.audit()
        self.assertEqual(audit["decision_count"], 0)
        self.assertEqual(audit["minted_unreserved_preview_count"], 0)
        self.assertEqual(audit["decisions"], [])

    def test_selector_prevalidates_source_before_live_assignment_mutation(self):
        env = _make_env(2)
        source = CapacityAwareRollingGAStorageAssigner(env, _config())
        selector = StrictStorageAssignmentOption(
            env,
            source,
            assignment_source="unit_capacity_aware",
            assignment_source_version=source.VERSION,
            preview_contract="unit_preview",
            assignment_contract="unit_reserved",
        )
        block = env.blocks[0]
        preview = selector.preview_assignment(block, source_cell=block.position)
        selector.bind_preview(block, preview)
        source._reserved_preview_plans[preview.proposal_id]["chosen_cell"] = (9, 9)
        with self.assertRaisesRegex(RuntimeError, "cell differs"):
            selector.commit_preview(block, preview)
        self.assertIsNone(block.storage_location)


if __name__ == "__main__":
    unittest.main()
