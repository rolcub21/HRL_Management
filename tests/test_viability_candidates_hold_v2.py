from dataclasses import replace
import unittest

from example.Options.ExplicitHoldOption import ExplicitHoldOption
from example.small_rooms_env import SmallRoomsEnv
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    ViabilityActionType,
    ViabilityMode,
    enumerate_viability_candidates,
)
from PSLAP.viability_candidates_hold_v2 import (
    CERTIFIED_HOLD_INTERFACE_V2,
    HIDDEN_SCHEDULE_CONTRACT_V2,
    ROBUST_STUTTER_CERTIFICATION_CONTRACT_V2,
    CertifiedHoldRuleV2,
    PrimitiveIdleBudgetStateV2,
    enumerate_viability_candidates_hold_v2,
)
from PSLAP.viability_filter import ViabilitySearchConfig


def compact_env(number_blocks, seed=7800):
    env = SmallRoomsEnv(
        grid_rows=5,
        grid_cols=5,
        number_blocks=number_blocks,
        choose_storage=False,
    )
    env.reset(instance=env.sample_episode_instance(seed))
    return env


def make_stored(block, cell, *, stored_time_step=0, duration=30):
    block.position = tuple(cell)
    block.storage_location = tuple(cell)
    block.carrying = False
    block.stored = True
    block.delivered = False
    block.stored_time_step = int(stored_time_step)
    block.storage_steps_needed = int(duration)
    block.storage_steps_elapsed = 0


def make_outside(block):
    block.position = None
    block.storage_location = None
    block.carrying = False
    block.stored = False
    block.delivered = False


class CertifiedHoldFrontierV2Tests(unittest.TestCase):
    def setUp(self):
        self.config = ViabilitySearchConfig(
            max_nodes=None,
            search_order="breadth_first",
        )

    def test_inbound_positive_slack_adds_bounded_hold_family(self):
        env = compact_env(2, seed=7801)
        make_stored(env.blocks[0], (2, 2), duration=30)
        inbound = env.blocks[1]
        inbound.position = tuple(env.pickup_cell)
        inbound.storage_location = None
        inbound.carrying = False
        inbound.stored = False
        inbound.delivered = False
        env.current_state = (1, 1)
        env.time_steps = 5

        # Frozen V1 denies Defer whenever an inbound block is present.
        frozen = enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            search_config=self.config,
        )
        self.assertEqual(
            frozen.candidates_of_type(ViabilityActionType.DEFER), ()
        )

        budget = PrimitiveIdleBudgetStateV2(
            max_idle_steps=20,
            idle_steps_since_progress=3,
        )
        snapshot = enumerate_viability_candidates_hold_v2(
            env,
            idle_budget=budget,
            remaining_episode_steps=7,
            recovery_witness_forced=False,
            search_config=self.config,
            hold_rule=CertifiedHoldRuleV2(
                max_option_steps=10,
                max_idle_steps=20,
            ),
        )

        holds = snapshot.candidates_of_type(ViabilityActionType.DEFER)
        self.assertEqual(tuple(item.horizon_steps for item in holds), (1, 7))
        self.assertTrue(
            all(
                item.mode is ViabilityMode.DEFER
                and isinstance(item.option, ExplicitHoldOption)
                and item.option.idle_budget_steps == 17
                and item.certificate is snapshot.current_certificate
                and item.successor_state is snapshot.recovery_state
                for item in holds
            )
        )
        self.assertGreater(
            len(snapshot.candidates_for_mode(ViabilityMode.ACCEPT)), 0
        )
        self.assertEqual(snapshot.audit.interface, CERTIFIED_HOLD_INTERFACE_V2)
        self.assertTrue(snapshot.hold_audit.inbound_present)
        self.assertTrue(snapshot.hold_audit.hold_allowed)
        self.assertEqual(
            snapshot.hold_audit.hold_reason,
            "bounded_inbound_hold_with_positive_stored_slack",
        )
        self.assertEqual(snapshot.hold_audit.hidden_schedule_fields_read, ())
        self.assertEqual(
            snapshot.hold_audit.hidden_schedule_contract,
            HIDDEN_SCHEDULE_CONTRACT_V2,
        )
        self.assertEqual(
            snapshot.hold_audit.robust_stutter_contract,
            ROBUST_STUTTER_CERTIFICATION_CONTRACT_V2,
        )

    def test_horizon_is_capped_by_primitive_budget_and_episode_horizon(self):
        env = compact_env(1, seed=7802)
        target = env.blocks[0]
        make_stored(target, (2, 2), duration=30)
        env.current_state = (1, 1)
        env.time_steps = 5

        snapshot = enumerate_viability_candidates_hold_v2(
            env,
            idle_budget=PrimitiveIdleBudgetStateV2(20, 18),
            remaining_episode_steps=1,
            recovery_witness_forced=False,
            search_config=self.config,
            hold_rule=CertifiedHoldRuleV2(10, 20),
        )

        holds = snapshot.candidates_of_type(ViabilityActionType.DEFER)
        self.assertEqual(tuple(item.horizon_steps for item in holds), (1,))
        self.assertEqual(snapshot.hold_audit.idle_budget_remaining_steps, 2)
        self.assertEqual(snapshot.hold_audit.remaining_episode_steps, 1)

    def test_due_unsafe_and_forced_states_fail_closed(self):
        env = compact_env(1, seed=7803)
        target = env.blocks[0]
        make_stored(target, (2, 2), duration=5)
        env.current_state = (1, 1)
        env.time_steps = 5
        budget = PrimitiveIdleBudgetStateV2(20, 0)
        rule = CertifiedHoldRuleV2(10, 20)

        due = enumerate_viability_candidates_hold_v2(
            env,
            idle_budget=budget,
            remaining_episode_steps=100,
            recovery_witness_forced=False,
            search_config=self.config,
            hold_rule=rule,
        )
        self.assertEqual(
            due.candidates_of_type(ViabilityActionType.DEFER), ()
        )
        self.assertEqual(
            due.hold_audit.hold_reason,
            "admitted_work_due_requires_recovery_progress",
        )

        forced = enumerate_viability_candidates_hold_v2(
            env,
            idle_budget=budget,
            remaining_episode_steps=100,
            recovery_witness_forced=True,
            search_config=self.config,
            hold_rule=rule,
        )
        self.assertEqual(
            forced.candidates_of_type(ViabilityActionType.DEFER), ()
        )
        self.assertEqual(
            forced.hold_audit.hold_reason,
            "exact_recovery_witness_forced",
        )

        # Exercise the rule-level fail-closed branch without weakening or
        # mocking the exact verifier used to build the physical snapshot.
        unsafe_certificate = replace(
            due.current_certificate,
            status=ViabilityStatus.UNSAFE,
            witness=(),
            witness_primitive_steps=None,
            reason="test_unsafe_certificate",
        )
        unsafe_frontier = replace(
            due.frontier,
            current_certificate=unsafe_certificate,
        )
        decision = rule.evaluate(
            env,
            unsafe_frontier,
            idle_budget=budget,
            remaining_episode_steps=100,
            recovery_witness_forced=False,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(
            decision.reason,
            "current_recoverability_not_certified_safe",
        )

    def test_empty_yard_external_wait_rule_does_not_read_hidden_schedule(self):
        env = compact_env(1, seed=7804)
        make_outside(env.blocks[0])
        env.blocks[0].arrival_step = 1_000_000
        base = enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            search_config=self.config,
        )
        self.assertIsNone(base.inbound_label)
        self.assertEqual(base.yard.blocks, ())

        class OnlineBlock:
            position = None
            carrying = False
            stored = False
            delivered = False

            @property
            def arrival_step(self):
                raise AssertionError("future arrival schedule was read")

            @property
            def storage_steps_needed(self):
                raise AssertionError("future processing duration was read")

        class OnlineEnv:
            blocks = [OnlineBlock()]

        decision = CertifiedHoldRuleV2(4, 8).evaluate(
            OnlineEnv(),
            base,
            idle_budget=PrimitiveIdleBudgetStateV2(8, 0),
            remaining_episode_steps=10,
            recovery_witness_forced=False,
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.horizons, (1, 4))
        self.assertEqual(
            decision.reason,
            "bounded_hold_for_observable_external_event",
        )


class PrimitiveIdleBudgetV2Tests(unittest.TestCase):
    def test_budget_uses_primitive_duration_and_only_real_progress_resets(self):
        budget = PrimitiveIdleBudgetStateV2(20, 4)
        held = budget.after_macro(ViabilityActionType.DEFER, duration=3)
        self.assertEqual(held.idle_steps_since_progress, 7)
        self.assertAlmostEqual(held.remaining_fraction, 13.0 / 20.0)

        nonprogress = held.after_macro(
            ViabilityActionType.RECONFIGURE,
            duration=8,
        )
        self.assertEqual(nonprogress.idle_steps_since_progress, 7)
        progressed = nonprogress.after_macro(
            ViabilityActionType.RECONFIGURE,
            duration=8,
            exact_rank_progress=True,
        )
        self.assertEqual(progressed.idle_steps_since_progress, 0)
        forced = held.after_macro(
            ViabilityActionType.RECONFIGURE,
            duration=8,
            recovery_witness_forced=True,
        )
        self.assertEqual(forced.idle_steps_since_progress, 0)
        accepted = held.after_macro(ViabilityActionType.ACCEPT, duration=5)
        self.assertEqual(accepted.idle_steps_since_progress, 0)
        delivered = held.after_macro(ViabilityActionType.DELIVER, duration=5)
        self.assertEqual(delivered.idle_steps_since_progress, 0)
        event = held.after_macro(
            ViabilityActionType.DEFER,
            duration=1,
            observed_event=True,
        )
        self.assertEqual(event.idle_steps_since_progress, 0)

        with self.assertRaisesRegex(ValueError, "exceeded"):
            PrimitiveIdleBudgetStateV2(5, 4).after_macro(
                ViabilityActionType.DEFER,
                duration=2,
            )


class ExplicitHoldOptionV2Tests(unittest.TestCase):
    def test_hold_stops_at_observed_arrival_before_cap(self):
        env = compact_env(1, seed=7810)
        block = env.blocks[0]
        make_outside(block)
        block.arrival_step = env.time_steps + 1
        option = ExplicitHoldOption(
            env,
            horizon_steps=5,
            idle_budget_steps=8,
        )

        state = env.get_current_state()
        action = option.policy(state)
        self.assertEqual(action, env.ACTION_IDS["WAIT"])
        state, _, _, _ = env.step(action)

        self.assertTrue(option.termination(state))
        self.assertEqual(option.last_outcome["reason"], "observed_event")
        self.assertEqual(option.last_outcome["steps"], 1)
        self.assertEqual(
            option.last_outcome["robust_stutter_contract"],
            option.ROBUST_STUTTER_CONTRACT,
        )

    def test_no_event_hold_stutters_until_exact_cap(self):
        env = compact_env(1, seed=7811)
        make_stored(env.blocks[0], (2, 2), duration=100)
        env.current_state = (1, 1)
        option = ExplicitHoldOption(
            env,
            horizon_steps=2,
            idle_budget_steps=4,
        )
        initial_position = tuple(env.current_state)
        initial_inventory_position = env.blocks[0].position

        state = env.get_current_state()
        for index in range(2):
            action = option.policy(state)
            state, _, _, _ = env.step(action)
            self.assertEqual(tuple(env.current_state), initial_position)
            self.assertEqual(env.blocks[0].position, initial_inventory_position)
            self.assertEqual(option.termination(state), index == 1)

        self.assertEqual(option.last_outcome["reason"], "hold_cap")
        self.assertEqual(option.last_outcome["steps"], 2)

    def test_option_observable_signature_does_not_read_hidden_schedule(self):
        class OnlineBlock:
            position = None
            carrying = False
            stored = False
            delivered = False

            @property
            def arrival_step(self):
                raise AssertionError("future arrival schedule was read")

            @property
            def storage_steps_needed(self):
                raise AssertionError("future processing duration was read")

        class Instance:
            instance_id = "online-only"

        class OnlineEnv:
            ACTION_IDS = {"WAIT": 6}
            time_steps = 0
            current_state = (1, 1)
            current_episode_instance = Instance()
            blocks = [OnlineBlock()]

            @staticmethod
            def is_state_terminal(state):
                del state
                return False

        env = OnlineEnv()
        option = ExplicitHoldOption(
            env,
            horizon_steps=1,
            idle_budget_steps=1,
        )
        self.assertTrue(option.initiation(env.current_state))
        self.assertEqual(option.policy(env.current_state), 6)
        self.assertTrue(option.termination(env.current_state))
        self.assertEqual(option.last_outcome["reason"], "hold_cap")


if __name__ == "__main__":
    unittest.main()
