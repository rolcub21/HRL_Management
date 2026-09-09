"""Focused contracts for the preference-conditioned vector VCG critic."""

import math

import pytest
import torch

from PSLAP.dynamic_yard import BlockView
from PSLAP.viability import RecoveryState
from viability_graph_hierarchy import (
    ACTION_FEATURE_DIM,
    ViabilityGraphCandidateRecord,
    ViabilityGraphConfig,
    ViabilityGraphHierarchyAgent,
)
from viability_graph_preference_conditioned import (
    PreferenceConditionedVectorAgent,
    PreferenceConditionedVectorConfig,
    PreferenceConditionedVectorQNetwork,
    PreferenceConditionedVectorTransition,
    deterministic_mode_candidate_selection,
    scalarized_merit,
    vector_smdp_target,
)


def _state(*, remaining: float = 4.0) -> RecoveryState:
    return RecoveryState(
        rows=2,
        cols=3,
        traversable=frozenset(
            (row, column) for row in range(2) for column in range(3)
        ),
        storage_cells=frozenset({(0, 1), (1, 1)}),
        exits=((0, 2),),
        blocks=(BlockView("B", (0, 1), remaining),),
        agent_position=(0, 0),
        pickup_cells=frozenset({(1, 0)}),
        wait_cells=frozenset({(1, 2)}),
    )


def _tiny_config(**overrides) -> PreferenceConditionedVectorConfig:
    values = dict(
        graph_hidden_dim=8,
        graph_embedding_dim=8,
        message_passing_steps=1,
        action_embedding_dim=8,
        candidate_hidden_dim=16,
        consequence_hidden_dim=16,
        batch_size=2,
        replay_capacity=16,
        target_update_every=1,
        update_every=1,
        lambda_max=0.2,
    )
    values.update(overrides)
    return PreferenceConditionedVectorConfig(**values)


def _network_inputs(count: int = 2):
    current = tuple(_state() for _ in range(count))
    successor = tuple(_state(remaining=3.0) for _ in range(count))
    features = torch.zeros(count, ACTION_FEATURE_DIM)
    return current, successor, features


def test_lambda_bounds_normalization_and_unconditioned_mask():
    conditioned = _tiny_config(condition_on_preference=True)
    assert float(conditioned.normalize_preference_lambda(0.0)) == 0.0
    assert float(conditioned.normalize_preference_lambda(0.1)) == pytest.approx(
        0.5
    )
    assert float(conditioned.normalize_preference_lambda(0.2)) == 1.0
    for invalid in (-1e-6, 0.200001, math.nan, math.inf):
        with pytest.raises(ValueError, match="lambda"):
            conditioned.normalize_preference_lambda(invalid)

    network = PreferenceConditionedVectorQNetwork(
        _tiny_config(condition_on_preference=False)
    )
    inputs = _network_inputs()
    with torch.no_grad():
        at_zero = network(*inputs, preference_lambda=0.0)
        at_max = network(*inputs, preference_lambda=0.2)
    torch.testing.assert_close(at_zero, at_max, rtol=0.0, atol=0.0)


def test_graph_representation_is_independent_of_preference():
    network = PreferenceConditionedVectorQNetwork(
        _tiny_config(condition_on_preference=True)
    )
    graph_batches = []
    conditioned_inputs = []

    def capture(_module, _arguments, output):
        graph_batches.append(output.detach().clone())

    def capture_conditioning(_module, arguments):
        conditioned_inputs.append(arguments[0].detach().clone())

    graph_handle = network.graph_encoder.register_forward_hook(capture)
    conditioning_handle = network.conditioned_trunk.register_forward_pre_hook(
        capture_conditioning
    )
    inputs = _network_inputs()
    try:
        with torch.no_grad():
            network(*inputs, preference_lambda=0.0)
            network(*inputs, preference_lambda=0.2)
    finally:
        graph_handle.remove()
        conditioning_handle.remove()

    assert len(graph_batches) == 2
    torch.testing.assert_close(
        graph_batches[0], graph_batches[1], rtol=0.0, atol=0.0
    )
    assert len(conditioned_inputs) == 2
    torch.testing.assert_close(
        conditioned_inputs[0][:, :-1],
        conditioned_inputs[1][:, :-1],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        conditioned_inputs[0][:, -1], torch.zeros(2)
    )
    torch.testing.assert_close(
        conditioned_inputs[1][:, -1], torch.ones(2)
    )


def test_deterministic_hierarchy_uses_normalized_modes_and_stable_ties():
    # Replicating a complete candidate multiset does not inflate its mode.
    original = deterministic_mode_candidate_selection(
        torch.tensor([2.0, -1.0, 0.5]),
        torch.tensor([0, 0, 1]),
        (0.4, 0.7, 0.9),
        candidate_keys=("b", "a", "c"),
    )
    duplicated = deterministic_mode_candidate_selection(
        torch.tensor([2.0, -1.0, 2.0, -1.0, 0.5]),
        torch.tensor([0, 0, 0, 0, 1]),
        (0.4, 0.7, 0.9),
        candidate_keys=("b", "a", "d", "c", "e"),
    )
    assert original.selected_mode_id == duplicated.selected_mode_id
    assert original.selected_mode_id == 0

    # Equal mode values choose the stable smaller mode ID; equal candidates
    # within that mode choose the lexicographically smaller candidate key.
    mode_tie = deterministic_mode_candidate_selection(
        torch.tensor([1.0, 1.0]),
        torch.tensor([1, 0]),
        (0.3, 0.3, 0.3),
        candidate_keys=("z", "b"),
    )
    assert mode_tie.selected_mode_id == 0
    assert mode_tie.selected_index == 1

    candidate_tie = deterministic_mode_candidate_selection(
        torch.tensor([1.0, 1.0]),
        torch.tensor([0, 0]),
        (0.3, 0.3, 0.3),
        candidate_keys=("b", "a"),
    )
    assert candidate_tie.selected_index == 1

    singleton = deterministic_mode_candidate_selection(
        torch.tensor([-100.0]),
        torch.tensor([2]),
        (0.3, 0.3, 0.3),
        candidate_keys=("forced-witness",),
    )
    assert singleton.selected_index == 0


def test_old_scalar_checkpoint_family_is_rejected():
    old = ViabilityGraphHierarchyAgent(
        config=ViabilityGraphConfig(
            graph_hidden_dim=8,
            graph_embedding_dim=8,
            message_passing_steps=1,
            action_embedding_dim=8,
            head_hidden_dim=16,
            batch_size=2,
            replay_capacity=16,
        ),
        seed=4,
        epsilon=0.0,
    )
    with pytest.raises(ValueError, match="incompatible"):
        PreferenceConditionedVectorAgent.from_checkpoint(old.checkpoint())


def test_vector_target_scalarizes_exactly_with_common_gamma_and_reward_scale():
    targets = vector_smdp_target(
        operational_returns=torch.tensor([5.0]),
        discounted_rehandles=torch.tensor([2.0]),
        durations=torch.tensor([3]),
        dones=torch.tensor([False]),
        next_vectors=torch.tensor([[30.0, 4.0]]),
        gamma=1.0,
        reward_scale=0.1,
    )
    torch.testing.assert_close(targets, torch.tensor([[30.5, 6.0]]))
    scalarized = targets[0, 0] - targets[0, 1]
    expected = 0.1 * 5.0 - 2.0 + (30.0 - 4.0)
    assert float(scalarized) == pytest.approx(expected)


def _record(key: str, mode_id: int) -> ViabilityGraphCandidateRecord:
    return ViabilityGraphCandidateRecord(
        key=key,
        mode_id=mode_id,
        action_type="deliver",
        current_state=_state(),
        successor_state=_state(remaining=3.0),
        action_features=(0.0,) * ACTION_FEATURE_DIM,
    )


def test_double_q_target_uses_one_online_selected_action_for_both_heads(
    monkeypatch,
):
    agent = PreferenceConditionedVectorAgent(
        config=_tiny_config(
            gamma=1.0,
            reward_scale=0.1,
            lambda_max=1.0,
            preference_relabels=1,
        ),
        seed=2,
        epsilon=0.0,
    )
    chosen = _record("chosen", 0)
    next_records = (_record("wrong-for-op", 0), _record("shared", 0))
    transition = PreferenceConditionedVectorTransition(
        chosen=chosen,
        operational_return=5.0,
        discounted_rehandles=2.0,
        raw_rehandles=2,
        duration=3,
        next_candidates=next_records,
        done=False,
        behavior_preference_lambda=1.0,
    )

    online_next = torch.tensor(
        [
            [10.0, 10.0],  # merit 0 at lambda=1
            [7.0, 1.0],    # merit 6: online-selected action
        ]
    )
    target_next = torch.tensor(
        [
            [100.0, 0.0],  # target-network merit would prefer this action
            [30.0, 4.0],
        ]
    )

    def fake_score(records, _preference, *, network=None, grad=False):
        keys = tuple(record.key for record in records)
        if keys == ("chosen",):
            return torch.zeros((1, 2), requires_grad=grad)
        assert keys == ("wrong-for-op", "shared")
        return online_next if network is agent.Q_local else target_next

    monkeypatch.setattr(agent, "_score_records", fake_score)
    _prediction, targets = agent._td_batch(
        (transition,), preference_lambdas=(1.0,)
    )
    # The online scalar merit selects index 1.  Both target components must
    # therefore come from target_next[1], rather than selecting per head or
    # reselecting with the target network.
    torch.testing.assert_close(targets, torch.tensor([[30.5, 6.0]]))


def test_lambda_zero_selection_is_independent_of_handling_head():
    operational = torch.tensor([1.0, 3.0, 2.0])
    modes = torch.tensor([0, 0, 1])
    first = torch.stack((operational, torch.tensor([0.0, 0.0, 0.0])), dim=1)
    second = torch.stack(
        (operational, torch.tensor([-1.0e6, 1.0e6, -5.0e5])), dim=1
    )
    choices = []
    for vectors in (first, second):
        choices.append(
            deterministic_mode_candidate_selection(
                scalarized_merit(vectors, 0.0),
                modes,
                (0.1, 0.1, 0.1),
                candidate_keys=("a", "b", "c"),
            ).selected_index
        )
    assert choices[0] == choices[1]
