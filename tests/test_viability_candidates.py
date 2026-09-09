from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

from example.Options.DirectDeliverOption import DirectDeliverOption
from example.Options.ExplicitDeferOption import ExplicitDeferOption
from example.Options.ExplicitAcceptOption import ExplicitAcceptOption
from example.Options.ReconfigureOption import ReconfigureOption
from example.small_rooms_env import SmallRoomsEnv
from example.yard_geometry import make_shipyard_env
from PSLAP.viability import RecoveryActionKind, ViabilityStatus
from PSLAP.viability_candidates import (
    BoundedEventDeferRule,
    StrictDecisionBoundaryError,
    ViabilityActionType,
    ViabilityCertificateCache,
    ViabilityMode,
    enumerate_viability_candidates,
)
from PSLAP.viability_filter import ViabilitySearchConfig


def compact_env(number_blocks, seed=7201):
    env = SmallRoomsEnv(
        grid_rows=5,
        grid_cols=5,
        number_blocks=number_blocks,
        choose_storage=False,
    )
    env.reset(instance=env.sample_episode_instance(seed))
    return env


def make_stored(block, cell, *, stored_time_step=0):
    block.position = tuple(cell)
    block.storage_location = tuple(cell)
    block.carrying = False
    block.stored = True
    block.delivered = False
    block.stored_time_step = int(stored_time_step)
    block.storage_steps_elapsed = 0


class ViableAcceptFrontierTests(unittest.TestCase):
    def test_physically_reachable_but_unrecoverable_accept_is_rejected(self):
        env = make_shipyard_env(
            arrival_rate=10.0,
            proc_mean=80,
            grid_rows=5,
            grid_cols=5,
            exit_width=1,
            number_blocks=6,
        )
        env.reset(instance=env.sample_episode_instance(7240))
        occupied = ((1, 1), (2, 1), (2, 2), (2, 3), (3, 2))
        for block, cell in zip(env.blocks[:5], occupied):
            make_stored(block, cell)
        inbound = env.blocks[5]
        inbound.position = tuple(env.pickup_cell)
        inbound.storage_location = None
        inbound.stored = False
        inbound.delivered = False
        inbound.carrying = False
        env.current_state = (1, 1)
        env.time_steps = 1

        snapshot = enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            search_config=ViabilitySearchConfig(
                max_nodes=20_000,
                search_order="goal_directed",
            ),
        )

        # (1, 2) is the only physically executable placement.  It creates a
        # closed-admission dead end, so the viability layer removes Accept and
        # exposes only certified capacity-recovery moves.
        self.assertEqual(snapshot.audit.physical_accept_count, 1)
        self.assertEqual(snapshot.audit.executable_accept_count, 1)
        self.assertEqual(snapshot.audit.unsafe_accept_count, 1)
        self.assertEqual(snapshot.audit.safe_accept_count, 0)
        self.assertEqual(snapshot.audit.fail_closed_rejection_count, 1)
        self.assertEqual(
            snapshot.candidates_for_mode(ViabilityMode.ACCEPT), ()
        )
        self.assertGreater(
            len(snapshot.candidates_for_mode(ViabilityMode.RECOVER)), 0
        )

    def test_safe_accepts_are_explicit_bound_options_without_a_ranking_policy(self):
        env = compact_env(1)

        with patch(
            "PSLAP.dynamic_yard.select_storage_location",
            side_effect=AssertionError("dynamic ranking was invoked"),
        ), patch(
            "PSLAP.neutral_protocol.nearest_from_candidates",
            side_effect=AssertionError("nearest ranking was invoked"),
        ):
            snapshot = enumerate_viability_candidates(
                env,
                consecutive_defer_decisions=0,
                search_config=ViabilitySearchConfig(max_nodes=None),
            )

        self.assertGreater(len(snapshot.candidates), 0)
        self.assertEqual(
            snapshot.candidates,
            snapshot.candidates_for_mode(ViabilityMode.ACCEPT),
        )
        self.assertTrue(
            all(
                item.action_type is ViabilityActionType.ACCEPT
                and isinstance(item.option, ExplicitAcceptOption)
                and item.certificate.status is ViabilityStatus.SAFE
                for item in snapshot.candidates
            )
        )
        self.assertEqual(
            {item.destination for item in snapshot.candidates},
            set(env.storage_positions),
        )
        self.assertEqual(
            snapshot.audit.physical_accept_count,
            snapshot.audit.safe_accept_count,
        )
        self.assertEqual(snapshot.audit.unknown_accept_count, 0)
        self.assertEqual(snapshot.audit.unsafe_accept_count, 0)
        self.assertFalse(snapshot.audit.defer_allowed)
        self.assertEqual(
            snapshot.audit.defer_reason,
            "inbound_requires_accept_or_capacity_release",
        )
        self.assertFalse(snapshot.audit.baseline_viability_teacher)
        self.assertEqual(
            snapshot.audit.verifier_authority,
            "exact_recovery_transition_search_v1",
        )

        with self.assertRaises(FrozenInstanceError):
            snapshot.decision_epoch = 99
        with self.assertRaises(FrozenInstanceError):
            snapshot.audit.candidate_count = 0
        with self.assertRaises(FrozenInstanceError):
            snapshot.candidates[0].key = "changed"

    def test_unknown_accept_certificates_are_audited_and_fail_closed(self):
        env = compact_env(2, seed=7202)
        # B1 remains the unique inbound.  Adding B2 means a post-accept proof
        # requires more than the deliberately tiny one-node search budget.
        make_stored(env.blocks[1], (2, 2))

        snapshot = enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            search_config=ViabilitySearchConfig(max_nodes=1),
        )

        self.assertGreater(snapshot.audit.unknown_accept_count, 0)
        self.assertEqual(snapshot.audit.safe_accept_count, 0)
        self.assertEqual(
            snapshot.candidates_for_mode(ViabilityMode.ACCEPT), ()
        )
        self.assertGreaterEqual(
            snapshot.audit.fail_closed_rejection_count,
            snapshot.audit.unknown_accept_count,
        )
        self.assertTrue(
            all(
                item.certificate.status is ViabilityStatus.SAFE
                for item in snapshot.candidates
            )
        )

    def test_complete_state_and_budget_cache_is_reused(self):
        env = compact_env(1, seed=7203)
        cache = ViabilityCertificateCache()
        config = ViabilitySearchConfig(max_nodes=None)

        first = enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            search_config=config,
            cache=cache,
        )
        second = enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            search_config=config,
            cache=cache,
        )

        self.assertGreater(first.audit.cache_misses, 0)
        self.assertEqual(first.audit.cache_hits, 0)
        self.assertEqual(second.audit.cache_misses, 0)
        self.assertEqual(
            second.audit.cache_hits, first.audit.cache_misses
        )
        self.assertEqual(len(cache), first.audit.cache_entries)
        self.assertEqual(second.audit.cache_entries, len(cache))

        different_order = enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            search_config=ViabilitySearchConfig(
                max_nodes=None, search_order="breadth_first"
            ),
            cache=cache,
        )
        self.assertGreater(different_order.audit.cache_misses, 0)
        self.assertEqual(
            different_order.audit.search_order, "breadth_first"
        )
        self.assertEqual(len(cache), 2 * first.audit.cache_entries)


class ViableRecoveryFrontierTests(unittest.TestCase):
    def test_safe_successors_map_to_direct_delivery_and_reconfiguration(self):
        env = compact_env(1, seed=7210)
        target = env.blocks[0]
        make_stored(target, (2, 2))
        env.current_state = (1, 1)
        # Make the block due, so the explicit liveness rule cannot add defer.
        env.time_steps = target.storage_steps_needed

        snapshot = enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            search_config=ViabilitySearchConfig(
                max_nodes=None, search_order="breadth_first"
            ),
        )

        deliveries = snapshot.candidates_of_type(
            ViabilityActionType.DELIVER
        )
        reconfigurations = snapshot.candidates_of_type(
            ViabilityActionType.RECONFIGURE
        )
        self.assertGreater(len(deliveries), 0)
        self.assertGreater(len(reconfigurations), 0)
        self.assertEqual(
            snapshot.candidates_for_mode(ViabilityMode.RECOVER),
            deliveries + reconfigurations,
        )
        self.assertTrue(
            all(
                isinstance(item.option, DirectDeliverOption)
                and item.recovery_action.kind
                is RecoveryActionKind.DELIVERY
                and item.rank_delta == 1
                and item.option.initiation(env.get_current_state())
                for item in deliveries
            )
        )
        self.assertTrue(
            all(
                isinstance(item.option, ReconfigureOption)
                and item.recovery_action.kind
                is RecoveryActionKind.RELOCATION
                and item.rank_delta == 0
                and item.option.initiation(env.get_current_state())
                for item in reconfigurations
            )
        )
        self.assertEqual(snapshot.audit.current_recovery_rank, 1)
        self.assertEqual(
            snapshot.audit.rank_reducing_recovery_count,
            len(deliveries),
        )
        self.assertFalse(snapshot.audit.defer_allowed)
        self.assertEqual(
            snapshot.audit.defer_reason,
            "admitted_work_due_requires_recovery_progress",
        )

    def test_unknown_recovery_successors_and_current_state_fail_closed(self):
        env = compact_env(3, seed=7211)
        for block, cell in zip(
            env.blocks, ((1, 1), (2, 2), (3, 3))
        ):
            make_stored(block, cell)
        env.current_state = (1, 2)
        env.time_steps = 1

        snapshot = enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            search_config=ViabilitySearchConfig(max_nodes=1),
        )

        self.assertIs(
            snapshot.audit.current_recovery_status,
            ViabilityStatus.UNKNOWN,
        )
        self.assertGreater(snapshot.audit.unknown_recovery_count, 0)
        self.assertEqual(snapshot.audit.safe_recovery_count, 0)
        self.assertEqual(snapshot.candidates, ())
        self.assertFalse(snapshot.audit.defer_allowed)
        self.assertEqual(
            snapshot.audit.defer_reason,
            "current_recoverability_not_certified_safe",
        )


class ExplicitDeferLivenessTests(unittest.TestCase):
    def test_known_positive_deadline_bounds_defer_horizon(self):
        env = compact_env(1, seed=7220)
        target = env.blocks[0]
        make_stored(target, (2, 2), stored_time_step=0)
        env.current_state = (1, 1)
        env.time_steps = target.storage_steps_needed - 3
        rule = BoundedEventDeferRule(
            max_option_steps=7,
            max_consecutive_defer_decisions=4,
        )

        snapshot = enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            liveness_rule=rule,
            search_config=ViabilitySearchConfig(
                max_nodes=None, search_order="breadth_first"
            ),
        )

        defers = snapshot.candidates_of_type(ViabilityActionType.DEFER)
        self.assertEqual(len(defers), 1)
        self.assertIsInstance(defers[0].option, ExplicitDeferOption)
        self.assertEqual(defers[0].horizon_steps, 3)
        self.assertEqual(defers[0].rank_delta, 0)
        self.assertEqual(snapshot.audit.defer_horizon_steps, 3)
        self.assertEqual(
            snapshot.audit.defer_reason,
            "bounded_wait_toward_known_positive_deadline",
        )

    def test_bound_defer_executes_without_any_retrieval_context(self):
        env = compact_env(1, seed=7224)
        target = env.blocks[0]
        make_stored(target, (2, 2), stored_time_step=0)
        env.current_state = (1, 1)
        env.time_steps = target.storage_steps_needed - 3
        rule = BoundedEventDeferRule(max_option_steps=7)

        with patch(
            "PSLAP.retrieval_context.retrieval_planning_context",
            side_effect=AssertionError("retrieval context was invoked"),
        ), patch(
            "example.Options.StrategicDeferOption.retrieval_planning_context",
            side_effect=AssertionError("legacy defer planning was invoked"),
        ):
            snapshot = enumerate_viability_candidates(
                env,
                consecutive_defer_decisions=0,
                liveness_rule=rule,
                search_config=ViabilitySearchConfig(max_nodes=None),
            )
            option = snapshot.candidates_of_type(
                ViabilityActionType.DEFER
            )[0].option
            state = env.get_current_state()
            for _ in range(3):
                action = option.policy(state)
                self.assertEqual(action, env.ACTION_IDS["WAIT"])
                state, _, _, _ = env.step(action)
                if option.termination(state):
                    break
            else:
                self.fail("explicit defer did not terminate at its bound")

        self.assertIsInstance(option, ExplicitDeferOption)
        self.assertEqual(option.last_outcome["reason"], "defer_cap")
        self.assertEqual(option.last_outcome["steps"], 3)

    def test_streak_bound_removes_defer_even_before_deadline(self):
        env = compact_env(1, seed=7221)
        make_stored(env.blocks[0], (2, 2))
        env.current_state = (1, 1)
        env.time_steps = 1
        rule = BoundedEventDeferRule(
            max_option_steps=5,
            max_consecutive_defer_decisions=2,
        )

        snapshot = enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=2,
            liveness_rule=rule,
            search_config=ViabilitySearchConfig(max_nodes=None),
        )

        self.assertEqual(
            snapshot.candidates_of_type(ViabilityActionType.DEFER), ()
        )
        self.assertEqual(
            snapshot.audit.defer_reason,
            "consecutive_defer_bound_reached",
        )

    def test_empty_yard_waits_only_for_an_observable_event_under_a_bound(self):
        env = compact_env(1, seed=7222)
        block = env.blocks[0]
        block.position = None
        block.storage_location = None
        block.stored = False
        block.delivered = False
        block.carrying = False
        # The value is intentionally irrelevant to the online liveness rule.
        block.arrival_step = 1_000_000
        rule = BoundedEventDeferRule(max_option_steps=4)

        snapshot = enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            liveness_rule=rule,
            search_config=ViabilitySearchConfig(max_nodes=None),
        )

        defers = snapshot.candidates_of_type(ViabilityActionType.DEFER)
        self.assertEqual(len(defers), 1)
        self.assertEqual(defers[0].horizon_steps, 4)
        self.assertEqual(
            snapshot.audit.defer_reason,
            "bounded_wait_for_observable_arrival_event",
        )

    def test_bound_defer_stops_at_an_observed_arrival_before_its_cap(self):
        env = compact_env(1, seed=7223)
        block = env.blocks[0]
        block.position = None
        block.storage_location = None
        block.stored = False
        block.delivered = False
        block.carrying = False
        block.arrival_step = env.time_steps + 1

        snapshot = enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            liveness_rule=BoundedEventDeferRule(max_option_steps=5),
            search_config=ViabilitySearchConfig(max_nodes=None),
        )
        option = snapshot.candidates_of_type(
            ViabilityActionType.DEFER
        )[0].option

        state = env.get_current_state()
        action = option.policy(state)
        state, _, _, _ = env.step(action)

        self.assertTrue(option.termination(state))
        self.assertEqual(option.last_outcome["reason"], "observed_event")
        self.assertEqual(option.last_outcome["steps"], 1)
        self.assertLess(option.last_outcome["steps"], option.horizon_steps)


class StrictBoundaryTests(unittest.TestCase):
    def test_carrying_or_half_committed_inventory_is_rejected(self):
        carrying = compact_env(1, seed=7230)
        carrying.blocks[0].carrying = True
        with self.assertRaisesRegex(
            StrictDecisionBoundaryError, "cannot carry inventory"
        ):
            enumerate_viability_candidates(
                carrying, consecutive_defer_decisions=0
            )

        committed = compact_env(1, seed=7231)
        committed.blocks[0].storage_location = (2, 2)
        with self.assertRaisesRegex(
            StrictDecisionBoundaryError, "half-committed"
        ):
            enumerate_viability_candidates(
                committed, consecutive_defer_decisions=0
            )

    def test_defer_streak_must_be_explicit_and_nonnegative(self):
        env = compact_env(1, seed=7232)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            enumerate_viability_candidates(
                env, consecutive_defer_decisions=-1
            )
        with self.assertRaises(TypeError):
            # Omitting the liveness state is deliberately a call-site error.
            enumerate_viability_candidates(env)


if __name__ == "__main__":
    unittest.main()
