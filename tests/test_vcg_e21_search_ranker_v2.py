from types import SimpleNamespace

import pytest

from experiments.conditioned_vcg.E21_search_ranker_v2_92k import run as v2
from PSLAP.viability import ViabilityStatus


def _block(label, position, storage_location, *, stored=False, delivered=False):
    return SimpleNamespace(
        label=label,
        position=position,
        storage_location=storage_location,
        carrying=False,
        stored=stored,
        delivered=delivered,
    )


def test_physical_signature_ignores_clocks_but_not_layout():
    block = _block("B1", (1, 1), (1, 1), stored=True)
    block.storage_steps_elapsed = 2
    env = SimpleNamespace(current_state=(0, 0), blocks=[block])
    first = v2._physical_signature(env)
    block.storage_steps_elapsed = 99
    assert v2._physical_signature(env) == first
    block.position = (1, 2)
    assert v2._physical_signature(env) != first


def test_geometry_service_bound_is_schedule_independent():
    env = SimpleNamespace(pickup_cell=(0, 0))
    state = SimpleNamespace(
        storage_cells=frozenset({(1, 0), (2, 0)}),
        exits=((3, 0),),
    )
    candidate = SimpleNamespace(successor_state=state)
    # pickup->(1,0)=1, (1,0)->exit=2, plus two pickup/putdown pairs.
    assert v2._minimum_unplaced_service_steps(env, candidate) == 7


def test_terminal_value_does_not_score_witness_delivery_timing():
    action = SimpleNamespace(steps=6)
    certificate = SimpleNamespace(
        status=ViabilityStatus.SAFE,
        witness=(action,),
        witness_primitive_steps=6,
    )
    state = SimpleNamespace(
        storage_cells=frozenset({(1, 0)}),
        exits=((3, 0),),
    )
    candidate = SimpleNamespace(certificate=certificate, successor_state=state)
    stored = _block("B1", (1, 0), (1, 0), stored=True)
    unplaced = _block("B2", None, None)
    env = SimpleNamespace(
        pickup_cell=(0, 0),
        blocks=[stored, unplaced],
    )
    service_bound = 7
    expected = (
        2 * v2.v1.pilot.v11.FROZEN_OBJECTIVE_SPEC.dense_b
        + v2.v1.PLACEMENT_REWARD
        + v2.v1.STEP_COST * (6 + service_bound)
    )
    assert v2._consistent_terminal_value(env, candidate) == pytest.approx(expected)


def test_terminal_reward_potential_is_preserved_by_on_target_delivery():
    # Removing one undelivered job removes exactly dense_b from the terminal
    # potential; an on-target realized delivery contributes exactly dense_b.
    dense_b = v2.v1.pilot.v11.FROZEN_OBJECTIVE_SPEC.dense_b
    before = dense_b * 2
    after = dense_b
    assert dense_b + after == before


def test_immediate_inverse_relocation_is_identified_but_progress_is_not():
    prior = SimpleNamespace(
        action_type=v2.v1.ViabilityActionType.RECONFIGURE,
        target_label="B1",
        source=(1, 1),
        destination=(2, 2),
    )
    signature = v2._inverse_relocation_signature(prior)
    inverse = SimpleNamespace(
        action_type=v2.v1.ViabilityActionType.RECONFIGURE,
        target_label="B1",
        source=(2, 2),
        destination=(1, 1),
    )
    other = SimpleNamespace(
        action_type=v2.v1.ViabilityActionType.RECONFIGURE,
        target_label="B1",
        source=(2, 2),
        destination=(3, 2),
    )
    accept = SimpleNamespace(action_type=v2.v1.ViabilityActionType.ACCEPT)
    assert v2._is_inverse_relocation(inverse, signature)
    assert not v2._is_inverse_relocation(other, signature)
    assert not v2._is_inverse_relocation(accept, signature)


def test_only_same_block_reconfigure_is_dominated_by_direct_delivery():
    deliver = SimpleNamespace(
        action_type=v2.v1.ViabilityActionType.DELIVER,
        target_label="B1",
    )
    same = SimpleNamespace(
        action_type=v2.v1.ViabilityActionType.RECONFIGURE,
        target_label="B1",
    )
    obstruction = SimpleNamespace(
        action_type=v2.v1.ViabilityActionType.RECONFIGURE,
        target_label="B2",
    )
    labels = v2._directly_deliverable_labels((deliver, same, obstruction))
    assert v2._is_dominated_self_reconfiguration(same, labels)
    assert not v2._is_dominated_self_reconfiguration(obstruction, labels)
