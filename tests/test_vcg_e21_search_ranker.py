from types import SimpleNamespace

import pytest

from experiments.conditioned_vcg.E21_search_ranker_92k import run as e21
from PSLAP.viability import RecoveryActionKind
from PSLAP.viability_candidates import ViabilityActionType
from vcg_objective_audit import delivery_reward


def test_dense_macro_reward_uses_frozen_dense_objective():
    execution = SimpleNamespace(
        duration=10,
        action_type=ViabilityActionType.ACCEPT.value,
        option_success=True,
        delivery_deviations=(0.0, 30.0),
    )
    expected = (
        10 * e21.STEP_COST
        + e21.PLACEMENT_REWARD
        + delivery_reward(0.0, e21.pilot.v11.FROZEN_OBJECTIVE_SPEC)
        + delivery_reward(30.0, e21.pilot.v11.FROZEN_OBJECTIVE_SPEC)
    )
    assert e21._dense_macro_reward(execution) == pytest.approx(expected)


def test_future_arrival_mask_preserves_observed_jobs():
    future = SimpleNamespace(
        arrival_step=20,
        position=None,
        carrying=False,
        stored=False,
        delivered=False,
    )
    observed = SimpleNamespace(
        arrival_step=2,
        position=(1, 1),
        carrying=False,
        stored=False,
        delivered=False,
    )
    stored = SimpleNamespace(
        arrival_step=30,
        position=(2, 2),
        carrying=False,
        stored=True,
        delivered=False,
    )
    env = SimpleNamespace(blocks=[future, observed, stored])
    assert e21._mask_unobserved_arrivals(env, root_time=10) == 1
    assert future.arrival_step == e21.MASKED_ARRIVAL_BASE
    assert observed.arrival_step == 2
    assert stored.arrival_step == 30


def test_witness_terminal_value_uses_timing_and_visible_external_work():
    action = SimpleNamespace(
        kind=RecoveryActionKind.DELIVERY,
        block_label="B1",
        steps=6,
    )
    certificate = SimpleNamespace(
        status=e21.ViabilityStatus.SAFE,
        witness=(action,),
        witness_primitive_steps=6,
    )
    successor = SimpleNamespace(
        block=lambda label: SimpleNamespace(remaining_time=4.0)
        if label == "B1"
        else None
    )
    candidate = SimpleNamespace(
        certificate=certificate,
        successor_state=successor,
    )
    visible = SimpleNamespace(
        position=(0, 0), carrying=False, stored=False, delivered=False
    )
    hidden = SimpleNamespace(
        position=None, carrying=False, stored=False, delivered=False
    )
    env = SimpleNamespace(blocks=[visible, hidden])
    expected = (
        6 * e21.STEP_COST
        + delivery_reward(2.0, e21.pilot.v11.FROZEN_OBJECTIVE_SPEC)
        + e21.PLACEMENT_REWARD
        + e21.pilot.v11.FROZEN_OBJECTIVE_SPEC.dense_b
    )
    assert e21._witness_terminal_value(env, candidate) == pytest.approx(expected)


def test_defer_counter_resets_only_on_observed_event_or_progress():
    normal = SimpleNamespace(action_type=ViabilityActionType.ACCEPT, option=None)
    assert e21._next_defer_count(3, normal) == 0

    observed = SimpleNamespace(
        action_type=ViabilityActionType.DEFER,
        option=SimpleNamespace(last_outcome={"reason": "observed_event"}),
    )
    capped = SimpleNamespace(
        action_type=ViabilityActionType.DEFER,
        option=SimpleNamespace(last_outcome={"reason": "defer_cap"}),
    )
    assert e21._next_defer_count(3, observed) == 0
    assert e21._next_defer_count(3, capped) == 4
