import unittest
from unittest.mock import patch

from example.small_rooms_env import SmallRoomsEnv
from example.yard_geometry import make_shipyard_env
from PSLAP.viability import RecoveryState, ViabilityStatus
from PSLAP.viability_candidates import (
    CANONICAL_CERTIFICATION_ORDER,
    ViabilityCertificateCache,
    enumerate_viability_candidates,
)
from PSLAP.viability_filter import ViabilitySearchConfig
from PSLAP.viability_prioritizer import (
    VIABILITY_PRIORITY_PROTOCOL,
    StatePriorityBatch,
    StatePriorityEntry,
)


class ReverseOrderPrioritizer:
    """Deterministic non-authoritative fake used to expose order coupling."""

    protocol = VIABILITY_PRIORITY_PROTOCOL
    checkpoint_sha256 = "a" * 64
    threshold = 0.25

    def __init__(self):
        self.calls = []

    def prioritize(self, keyed_states):
        keyed_states = tuple(keyed_states)
        self.calls.append(keyed_states)
        count = len(keyed_states)
        entries = tuple(
            StatePriorityEntry(
                original_index=index,
                key=str(key),
                state=state,
                safety_probability_mean=0.9,
                safety_probability_std=0.0,
                safety_probability_lcb=0.9,
                recovery_rank_mean=1.0,
                recovery_rank_std=0.0,
                primitive_steps_mean=1.0,
                primitive_steps_std=0.0,
                # Intentionally pass every state, including exact UNSAFE and
                # UNKNOWN states.  This flag must never acquire authority.
                priority_pass=True,
            )
            for index, (key, state) in enumerate(keyed_states)
        )
        return StatePriorityBatch(
            entries=entries,
            ordered_indices=tuple(reversed(range(count))),
            inference_seconds=0.125,
            threshold=self.threshold,
            checkpoint_sha256=self.checkpoint_sha256,
        )


def compact_env(number_blocks, *, seed):
    env = SmallRoomsEnv(
        grid_rows=5,
        grid_cols=5,
        number_blocks=number_blocks,
        choose_storage=False,
    )
    env.reset(instance=env.sample_episode_instance(seed))
    return env


def make_stored(block, cell):
    block.position = tuple(cell)
    block.storage_location = tuple(cell)
    block.carrying = False
    block.stored = True
    block.delivered = False
    block.stored_time_step = 0
    block.storage_steps_elapsed = 0


def canonical_frontier(snapshot):
    """Exclude live option identity and wall-clock-only audit values."""

    return tuple(
        (
            candidate.key,
            candidate.mode.value,
            candidate.action_type.value,
            candidate.target_label,
            candidate.source,
            candidate.destination,
            candidate.successor_state,
            candidate.certificate.status.value,
            candidate.certificate.reason,
            candidate.certificate.witness,
            candidate.certificate.exact_recovery_rank,
        )
        for candidate in snapshot.candidates
    )


SEMANTIC_AUDIT_FIELDS = (
    "terminal",
    "current_recovery_status",
    "current_recovery_rank",
    "recovery_rank_exact",
    "physical_accept_count",
    "executable_accept_count",
    "accept_executor_rejections",
    "safe_accept_count",
    "unsafe_accept_count",
    "unknown_accept_count",
    "legal_recovery_count",
    "recovery_executor_rejections",
    "safe_recovery_count",
    "unsafe_recovery_count",
    "unknown_recovery_count",
    "defer_allowed",
    "defer_horizon_steps",
    "defer_reason",
    "candidate_count",
    "accept_candidate_count",
    "deliver_candidate_count",
    "reconfigure_candidate_count",
    "defer_candidate_count",
    "rank_reducing_recovery_count",
    "cache_hits",
    "cache_misses",
    "cache_entries",
    "complete_frontier_exactly_verified",
)


def assert_semantically_identical(test_case, exact_full, ordered_full):
    test_case.assertEqual(
        exact_full.recovery_state,
        ordered_full.recovery_state,
    )
    test_case.assertEqual(
        exact_full.current_certificate,
        ordered_full.current_certificate,
    )
    test_case.assertEqual(
        canonical_frontier(exact_full),
        canonical_frontier(ordered_full),
    )
    for field in SEMANTIC_AUDIT_FIELDS:
        test_case.assertEqual(
            getattr(exact_full.audit, field),
            getattr(ordered_full.audit, field),
            field,
        )
    test_case.assertEqual(
        exact_full.audit.fail_closed_rejection_count,
        ordered_full.audit.fail_closed_rejection_count,
    )


class PermutationOnlyPriorityIntegrationTests(unittest.TestCase):
    def test_reverse_order_preserves_complete_safe_frontier_and_exact_work(self):
        config = ViabilitySearchConfig(
            max_nodes=20_000,
            search_order="goal_directed",
        )
        exact_full = enumerate_viability_candidates(
            compact_env(1, seed=7301),
            consecutive_defer_decisions=0,
            search_config=config,
            cache=ViabilityCertificateCache(),
        )

        prioritizer = ReverseOrderPrioritizer()
        observed_exact_states = []
        from PSLAP import viability_filter

        real_analyze = viability_filter.analyze_recoverability

        def record_exact_order(state, **kwargs):
            observed_exact_states.append(state)
            return real_analyze(state, **kwargs)

        with patch(
            "PSLAP.viability_filter.analyze_recoverability",
            side_effect=record_exact_order,
        ):
            ordered_full = enumerate_viability_candidates(
                compact_env(1, seed=7301),
                consecutive_defer_decisions=0,
                search_config=config,
                cache=ViabilityCertificateCache(),
                state_prioritizer=prioritizer,
            )

        assert_semantically_identical(self, exact_full, ordered_full)
        self.assertEqual(len(prioritizer.calls), 1)
        proposed_states = tuple(state for _, state in prioritizer.calls[0])
        self.assertGreater(len(proposed_states), 1)
        self.assertEqual(observed_exact_states, list(reversed(proposed_states)))

        self.assertEqual(
            exact_full.audit.certification_order,
            CANONICAL_CERTIFICATION_ORDER,
        )
        self.assertEqual(exact_full.audit.priority_states_scored, 0)
        self.assertEqual(exact_full.audit.priority_pass_count, 0)
        self.assertIsNone(exact_full.audit.priority_checkpoint_sha256)

        self.assertEqual(
            ordered_full.audit.certification_order,
            VIABILITY_PRIORITY_PROTOCOL,
        )
        self.assertEqual(
            ordered_full.audit.priority_states_scored,
            len(proposed_states),
        )
        self.assertEqual(
            ordered_full.audit.priority_pass_count,
            len(proposed_states),
        )
        self.assertEqual(ordered_full.audit.priority_inference_seconds, 0.125)
        self.assertEqual(
            ordered_full.audit.priority_checkpoint_sha256,
            prioritizer.checkpoint_sha256,
        )
        self.assertTrue(ordered_full.audit.complete_frontier_exactly_verified)
        self.assertTrue(
            all(
                candidate.certificate.status is ViabilityStatus.SAFE
                for candidate in ordered_full.candidates
            )
        )

    def test_high_priority_unsafe_accept_remains_excluded(self):
        def unsafe_env():
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
            return env

        config = ViabilitySearchConfig(
            max_nodes=20_000,
            search_order="goal_directed",
        )
        exact_full = enumerate_viability_candidates(
            unsafe_env(),
            consecutive_defer_decisions=0,
            search_config=config,
        )
        prioritizer = ReverseOrderPrioritizer()
        ordered_full = enumerate_viability_candidates(
            unsafe_env(),
            consecutive_defer_decisions=0,
            search_config=config,
            state_prioritizer=prioritizer,
        )

        assert_semantically_identical(self, exact_full, ordered_full)
        self.assertEqual(ordered_full.audit.unsafe_accept_count, 1)
        self.assertEqual(ordered_full.audit.safe_accept_count, 0)
        self.assertEqual(ordered_full.audit.accept_candidate_count, 0)
        prioritized_accept_keys = tuple(
            key
            for batch in prioritizer.calls
            for key, _ in batch
            if key.startswith("accept:")
        )
        self.assertEqual(len(prioritized_accept_keys), 1)
        self.assertEqual(
            ordered_full.audit.priority_pass_count,
            ordered_full.audit.priority_states_scored,
        )
        self.assertEqual(
            ordered_full.audit.fail_closed_rejection_count,
            exact_full.audit.fail_closed_rejection_count,
        )
        self.assertTrue(
            all(
                candidate.certificate.status is ViabilityStatus.SAFE
                for candidate in ordered_full.candidates
            )
        )

    def test_high_priority_unknown_states_remain_fail_closed(self):
        def unknown_env():
            env = compact_env(2, seed=7302)
            make_stored(env.blocks[1], (2, 2))
            return env

        config = ViabilitySearchConfig(max_nodes=1)
        exact_full = enumerate_viability_candidates(
            unknown_env(),
            consecutive_defer_decisions=0,
            search_config=config,
        )
        prioritizer = ReverseOrderPrioritizer()
        ordered_full = enumerate_viability_candidates(
            unknown_env(),
            consecutive_defer_decisions=0,
            search_config=config,
            state_prioritizer=prioritizer,
        )

        assert_semantically_identical(self, exact_full, ordered_full)
        self.assertGreater(ordered_full.audit.unknown_accept_count, 0)
        self.assertEqual(ordered_full.audit.safe_accept_count, 0)
        self.assertEqual(ordered_full.audit.accept_candidate_count, 0)
        self.assertGreater(ordered_full.audit.safe_recovery_count, 0)
        self.assertEqual(
            ordered_full.audit.candidate_count,
            ordered_full.audit.safe_recovery_count,
        )
        prioritized_accept_keys = tuple(
            key
            for batch in prioritizer.calls
            for key, _ in batch
            if key.startswith("accept:")
        )
        self.assertEqual(
            len(prioritized_accept_keys),
            ordered_full.audit.unknown_accept_count,
        )
        self.assertEqual(
            ordered_full.audit.priority_pass_count,
            ordered_full.audit.priority_states_scored,
        )
        self.assertEqual(
            ordered_full.audit.fail_closed_rejection_count,
            exact_full.audit.fail_closed_rejection_count,
        )
        self.assertTrue(ordered_full.audit.complete_frontier_exactly_verified)
        self.assertTrue(
            all(
                candidate.certificate.status is ViabilityStatus.SAFE
                for candidate in ordered_full.candidates
            )
        )


if __name__ == "__main__":
    unittest.main()
